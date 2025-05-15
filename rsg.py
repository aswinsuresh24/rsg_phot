import pandeia.engine
import stsynphot
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob, sys
import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from scipy import interpolate
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from dust import dustgen

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
        self.distance = [12.3, 1.8] * u.Mpc
        self.dm = 5.0 * np.log10(self.distance[0].value) + 25.0

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

        self.bounds = {
            'luminosity': [-1.0, 7.0],
            'temperature': [800., 100000.] * u.K,
            'tau_V': [0.01, 6.0],
            'dust_temp': [200., 2000.] * u.K,
            'mass': [0.1, 120.0] * u.M_sun,
            'age': [1.0e4, 13.0e9] * u.yr,
            'period': [0.0, 4.0], # in log10(days)
            'ratio': [0.1, 0.9],
            'Av': [0.0, 6.0],
            'Rv': [2.0, 6.0]
        }

    def get_jwst_filters(self, filtnam):

        inst, filt = filtnam.split(',')
        filt = filt.lower()
        inst = inst.lower()
        
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
        
    def create_rsg(self, tau_V, lum, temp, dust_temp, dist, sptype='all'):
        # RSG model takes luminosity in Lsol as input
        dust_model = dustgen(dist=dist)
        scaled_lum = 10**(lum.to(u.Lsun)).value
        spec = dust_model.get_ext_bb((tau_V, scaled_lum, temp, dust_temp), sptype=sptype)

        return(spec)
        
    def compute_rsg_mag(self, inst_filt, tau_V, lum, temp, dust_temp, dist):

        # Scale spectrum up to input luminosity
        sp = self.create_rsg(tau_V, lum, temp, dust_temp, dist)

        # Get bandpasses and compute mags
        bandpasses = self.get_bandpasses(inst_filt)
        mags = self.compute_synphot_mag(sp, bandpasses)
        mags = dict(zip(inst_filt, mags))

        return(mags)