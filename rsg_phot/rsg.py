import argparse
import pandeia.engine
import stsynphot
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob, sys
import numpy as np
from astropy.io import fits, ascii
import astropy.units as u
import astropy.constants as const
from scipy import interpolate
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from rsg_phot.dust import dustgen
from astropy.io.misc.hdf5 import read_table_hdf5
import pickle

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Interpolate DUSTY grids')
    parser.add_argument('--subdir', type=str, help='Directory containing DUSTY grids')
    parser.add_argument('--outdir', type=str, default='data/interpolate', help='Output directory of picke file')
    parser.add_argument('--modelname', type=str, default='rsg', help='Name of output pickle file')
    parser.add_argument('--ntau', type=int, default=27, help='Number of points in tau grid')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize spectra?')
    return parser

class rsg_phot(object):
    def __init__(self, verbose=True, interpolate=True):
        self.filename = ''

        try:
            self.pandeia = os.environ['pandeia_refdata']
        except:
            print('WARNING: if using JWST filters, set PANDEIA env variable')
            self.pandeia = None

        # Magnitude system of input data (AB is usually best)
        self.magsystem = 'abmag'

        # Storing model data and magnitudes for models
        self.models = None
        self.model_mags = {}

        self.model_size = 1000
        self.ages_shape = [0, 0, self.model_size]
        self.mass_shape = [0, 0, self.model_size]

        self.phottable = None
        self.backend = None
        self.pickles_data = None
        self.verbose = verbose

        self.all_ages = []
        self.all_masses = []

        self.model_type = ''
        self.model_fit_params = []
        self.model_fit_blobs = []

        self.mist_masses = list(np.arange(7.5, 40.0, 0.5))
        add_masses = list(np.arange(8.0,15.0,0.1))
        add_masses = [int(m*10.0)*1.0/10.0 for m in add_masses]
        self.mist_masses = np.array(np.unique(self.mist_masses+add_masses))*u.M_sun
        self.metallicity = 0.014 # Metallicity in terms of Z

        # Distance and uncertainty in Mpc
        self.distance = [10, 0.5] * u.Mpc
        self.dm = 5.0 * np.log10(self.distance[0].value) + 25.0
        self.flux_scale = ((1 * u.L_sun) / (4 * np.pi * (self.distance[0])**2)).to(u.erg/u.s/u.cm**2)

        # For extinction likelihood values to use as a prior for modeling
        self.extinction = {
            'Av': None, 'Rv': None, 'pdf': None, 'cdf': None,
            'interpolation': None, 'likelihood': False,
            # For the extinction in a particular bandpass given Av,Rv
            'function': {}
        }

        # A list of rv values for input filters
        self.rv = []

        # Default wavelength binset in angstroms for pysynphot
        self.waves = (3500.0 + 10.0*np.arange(9650)) * u.Angstrom
        self.bandpasses = None
        self.bandpass_dir = os.environ['BANDPASS']

        self.nrc_filts = ['F070W','F090W','F115W','F140M','F150W','F162M',
                          'F164N','F150W2','F182M','F187N','F200W','F210M',
                          'F212N','F250M','F277W','F300M','F322W2','F323N',
                          'F335M','F356W','F360M','F405N','F410M','F430M',
                          'F444W','F460M','F466N','F470N','F480M']
        
    def extinction_law(self, wave, Av, Rv, deredden=False):

        # Inverse wavelength dependent quantities
        x=1./(wave*1.0e-4)
        y=x-1.82

        # Variables proportional to extinction
        a=np.zeros(len(wave))
        b=np.zeros(len(wave))
        Fa=np.zeros(len(wave))
        Fb=np.zeros(len(wave))

        # Masks to apply piecewise Cardelli et al. function
        mask1 = np.where(x < 1.1)
        mask2 = np.where((x > 1.1) & (x < 3.3))
        mask3 = np.where(x > 3.3)
        mask4 = np.where(x > 5.9)

        x=1./(wave[mask1]*1.0e-4)
        y=x-1.82

        a[mask1]=0.574 * x**1.61
        b[mask1]=-0.527 * x**1.61

        x=1./(wave[mask2]*1.0e-4)
        y=x-1.82

        a[mask2]=1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 +\
            0.72085 * y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b[mask2]=1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 -\
            5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7

        x=1./(wave[mask4]*1.0e-4)
        y=x-1.82

        Fa[mask4]=-0.04473 * (x-5.9)**2 - 0.009779 * (x-5.9)**3
        Fb[mask4]=0.2130 * (x-5.9)**2 + 0.1207 * (x-5.9)**3

        x=1./(wave[mask3]*1.0e-4)
        y=x-1.82

        a[mask3]=1.752 - 0.316 * x - 0.104/((x-4.67)**2 + 0.341) + Fa[mask3]
        b[mask3]=-3.090 + 1.825 * x + 1.206/((x-4.62)**2 + 0.263) + Fb[mask3]

        Alam = Av*(a+b/Rv)
        elam = 10**(-0.4 * Alam)
        if deredden: elam = 1./elam

        return(elam)

    def get_jwst_filters(self, filtnam):

        inst, filt = filtnam.split(',')
        filt = filt.lower()
        inst = inst.lower()

        if inst=='nircam':
            file = f'{filt.lower()}.txt'
            fullfile = os.path.join(self.bandpass_dir, inst, file)

            table = ascii.read(fullfile)
            wave = table['Microns']*1e4 * u.Angstrom
            trans = table['Throughput']

            bp = SpectralElement(Empirical1D, points=wave, lookup_table=trans)
            return(bp)
        
        globstr = os.path.join(self.pandeia, 'jwst', inst, 'filters', f'*{filt}_trans*')
        filts = glob.glob(globstr)
        
        if len(filts)!=1:
            return None
        
        f = filts[0]
        
        base = os.path.split(f)[1]
        name = inst.lower()+'_'+filt.lower()
        hdu = fits.open(f)
        wave = np.array([float(el[0])*1.0e4 for el in hdu[1].data]) * u.Angstrom
            
        tran = np.array([float(el[1]) for el in hdu[1].data])
            
        # Only consider wavelengths < 30 microns
        mask = wave < 30.0*1.0e4 * u.Angstrom
        wave = wave[mask]
        tran = tran[mask]
            
        bp = SpectralElement(Empirical1D, points=wave, lookup_table=tran)
            
        return bp
    
    def get_bandpasses(self, filt, inst='nircam', redo=False, save=True):

        inst_filt = [f'{inst},{f}' for f in filt]
        if (self.bandpasses and not redo and isinstance(inst_filt, list) and
            len(inst_filt)==len(self.bandpasses)):
            return(self.bandpasses)

        # Get bandpasses for each filter in phottable
        bandpasses = []
        for flt in inst_filt:
            bandpasses.append([self.get_jwst_filters(flt)])

        if save:
            self.bandpasses = bandpasses

        return bandpasses

    def compute_synphot_mag(self, spectrum, bandpasses):
        # Get bandpasses for each filter in phottable
        mags = []
        kwargs = {'force': 'taper', 'binset': self.waves}
        for bp in bandpasses:

            if len(bp)==1:
                obs = synphot.Observation(spectrum, bp[0], **kwargs)

                try:
                    mag = obs.effstim(self.magsystem)
                    mags.append(mag)
                except ValueError:
                    mags.append(np.nan)

            elif len(bp)==2:
                obs1 = synphot.Observation(spectrum, bp[0], **kwargs)
                obs2 = synphot.Observation(spectrum, bp[1], **kwargs)

                try:
                    mag1 = obs1.effstim(self.magsystem)
                    mag2 = obs2.effstim(self.magsystem)
                    mags.append(mag1-mag2)
                except ValueError:
                    mags.append(np.nan)

        return(mags)
        
    def create_rsg(self, tau_V, lum, temp, dust_temp, dist, dust_model, sptype='all'):
        # RSG model takes luminosity in Lsol as input
        rsg_model = dustgen(dist=dist)
        scaled_lum = 10**(lum.to(u.Lsun)).value
        spec = rsg_model.get_ext_bb((tau_V, scaled_lum, temp, dust_temp), dust_model=dust_model, sptype=sptype)

        return(spec)
        
    def compute_rsg_mag(self, inst_filt, tau_V, lum, temp, dust_temp, dist, dust_model):

        # Scale spectrum up to input luminosity
        sp = self.create_rsg(tau_V, lum, temp, dust_temp, dist, dust_model)

        # Get bandpasses and compute mags
        bandpasses = self.get_bandpasses(inst_filt)
        mags = self.compute_synphot_mag(sp, bandpasses)
        mags = dict(zip(inst_filt, mags))

        return(mags)
    
    def create_rsg_grid(self, subdir, modelname, outdir='data/interpolate', ntau=27, norm=False, loglums=None, av=None, rv=None):

        all_grids = sorted(glob.glob(subdir + '/rsg*'))
        grid_temps = np.unique([float(os.path.basename(i).split('_')[1]) for i in all_grids])
        grid_dust_temps = np.unique([float(os.path.basename(i).split('_')[2]) for i in all_grids])
        outfile_ = os.path.join(subdir, f'rsg_{grid_temps[0]}_{grid_dust_temps[0]}', f'rsg_{grid_temps[0]}_{grid_dust_temps[0]}.out')
        grid_taus = np.loadtxt(outfile_, skiprows=42, max_rows = ntau)[:, 1]

        if loglums is None:
            loglums = np.array([3, 6])
        if rv is None:
            rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        if av is None:
            av = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])

        mags = {}
        for flt in self.nrc_filts:
            mags[flt] = np.zeros((len(grid_temps), len(grid_dust_temps), len(grid_taus), len(loglums), len(rv), len(av)))

        for i, te in enumerate(grid_temps):
            for j, td in enumerate(grid_dust_temps):
                sd = f'rsg_{te}_{td}'
                specfile = os.path.join(subdir, sd, sd+'.hdf5')
                spectable = read_table_hdf5(specfile)
                wv = spectable['lambda'].to(u.Angstrom)
                _, mask = np.unique(wv, return_index=True)
                wv = wv[mask]
                energy = (const.h*const.c/wv).to(u.erg)
                print(specfile)

                for k, col in enumerate(spectable.columns[1:]):
                    if norm:
                        flux = spectable[col][mask]
                        normalize = simpson(flux, x=wv.to(u.um).value)
                        flux = self.flux_scale * flux/normalize / u.micron
                        flux = flux.to(u.erg/u.s/u.cm**2/u.Angstrom)
                    else:
                        try:
                            flux = spectable[col][mask].to(u.erg/u.s/u.cm**2/u.Angstrom)
                        except Exception as e:
                            print(e)
                            print('Incorrect flux units in model grid')

                    for m, Rv_ in enumerate(rv):
                        for n, Av_ in enumerate(av):
                            host_ext = self.extinction_law(wv.value, Av_, Rv_)
                            flux_ext = flux*host_ext
                            sp = synphot.SourceSpectrum(Empirical1D, points=wv, lookup_table=(flux_ext/energy).value)

                            for flt in self.nrc_filts:
                                bp = self.get_jwst_filters('NIRCAM,' + flt)
                                kwargs = {'force': 'taper', 'binset': wv}
                                obs = synphot.Observation(sp, bp, **kwargs)
                                mag = obs.effstim(self.magsystem)
                                
                                for l, logl in enumerate(loglums):
                                    logl = logl - 4
                                    scale_mag = mag.value-2.5*logl                                    
                                    mags[flt][i, j, k ,l, m, n] = scale_mag

        models = {}
        params = (grid_temps, grid_dust_temps, grid_taus, loglums, rv, av)

        for flt in self.nrc_filts:
            models[flt] = interpolate.RegularGridInterpolator(params, mags[flt], method='linear', bounds_error=True)

        pfile = os.path.join(outdir, f'{modelname}.pkl')
        pickle.dump(models, open(pfile, 'wb'))

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    subdir = args.subdir
    outdir = args.outdir
    modelname = args.modelname
    ntau = args.ntau
    norm = args.norm

    rsg = rsg_phot()
    rsg.create_rsg_grid(subdir, modelname, outdir, ntau, norm)