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
from astropy.table import QTable
from astropy.io import ascii
from scipy import interpolate
from scipy.integrate import simpson
import subprocess
import tempfile
import argparse
from pathlib import Path

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Create DUSTY grids')
    parser.add_argument('--logz', type=float, default=0.00, help='log metallicity (0.0, -0.25 and -0.5 for MARCS; 0.0 and -0.5 for NewEra)')
    parser.add_argument('--comp', type=str, default='sil', help='Dust composition (sil or grf)')
    parser.add_argument('--outdir', type=str, default='data/dusty_sil_grid', help='Output directory for spectra')
    parser.add_argument('--modeldir', type=str, default='data/marcs/m1.00_g0.00', help='Model directory for DUSTY input spectra')
    parser.add_argument('--rewrite_lambda_grid', default=False, action=argparse.BooleanOptionalAction, help='Rewrite the wavelength grid in DUSTY and recompile')
    return parser
class dusty_gen(object):
    def __init__(self, 
                 dist=10*u.Mpc, 
                 logz=0.00,
                 logg=0.00,
                 interp_method='pchip', 
                 rewrite_lambda_grid=False,
                 modeldir='/Users/aswin/rsg_phot/data/marcs/m1.00_g0.00',
                 outdir='/Users/aswin/rsg_phot/data/dusty_grid'):

        self.dist = dist

        # This is the total integrated flux of a source with Lum = 1 Lsun in units of
        # erg/s/cm2.  Used to renormalize the RSG spectrum.  This is equivalent to
        # 3.839e33/(4 * pi * (10 * 3.08568025e18)^2) for 1 Lsol at 10 pc.
        self.FLUX_SCALE = ((1 * u.L_sun) / (4 * np.pi * self.dist**2)).to(u.erg/u.s/u.cm**2)

        # Dust-to-gas mass ratio assumption
        self.DUST_TO_GAS = 0.01

        # Read MARCS models
        self.rsg_modeldir = modeldir
        self.rsg_logz = logz
        sgn = '+' if self.rsg_logz > -1e-5 else '-'
        if 'marcs' in self.rsg_modeldir:
            self.modeltype='MARCS'
        elif 'newera' in self.rsg_modeldir:
            self.modeltype='NewEra'
        else:
            raise ValueError('Model directory must have either MARCS or NewEra models')
        
        model_spec = np.loadtxt(os.path.join(self.rsg_modeldir, 
                                             '{modeltype}_models_g{logg:.2f}_Z{sgn}{met:.2f}.dat'.format(modeltype=self.modeltype, 
                                                                                                         sgn=sgn, 
                                                                                                         logg=logg, 
                                                                                                         met=np.abs(self.rsg_logz))),
                                             dtype=float)
        self.rsg_temp = model_spec[0] * u.K
        self.rsg_data = model_spec[1:]
        self.rsg_minflux = np.min(self.rsg_data)
        self.rsg_wavelength = np.logspace(np.log10(1300.), np.log10(200000.), num=5036) * u.Angstrom
        self.rsg_interp = self.setup_temp_interp(interp_method)

        # DUSTY setup 
        self.dusty_basedir = os.environ['DUSTY_PATH'] #full path
        self.dusty_datadir = outdir # full path
        self.dusty_lambda_grid = self.make_lambda_grid(npts=500)
        self.dusty_n_taugrid = 27
        self.validate_dusty_dir(rewrite_lambda_grid)
        self.dust_comp = 'sil'
        self.shell_thickness = 2

    def rebin(self, a, newshape):
        newarray = np.zeros(newshape)
        curr = 0
        for i in np.arange(newshape):
            s = slice(curr, curr+int(a.shape[0]/newshape), 1)
            newarray[i] = np.mean(a[s])
            curr += int(a.shape[0]/newshape)
        return newarray
    
    def setup_temp_interp(self, interp_method='pchip'):
        # All model spectra share one wavelength grid, so interpolate in temperature
        # only, in log flux. This is exact at the grid temperatures and, with PCHIP,
        # monotone between them.
        logflux = np.log10(np.clip(self.rsg_data, self.rsg_data[self.rsg_data > 0].min(), None))
        if interp_method in ('pchip', 'cubic'):
            return interpolate.PchipInterpolator(self.rsg_temp.value, logflux, axis=1, extrapolate=False)
        elif interp_method == 'linear':
            return interpolate.interp1d(self.rsg_temp.value, logflux, axis=1, bounds_error=False)
        else:
            raise ValueError('interp_method must be "pchip" (or "cubic") or "linear"')

    def get_rsg(self, temp, model='g00'):

        if model == 'g00': 
            flux = 10**self.rsg_interp(np.atleast_1d(temp.to(u.K).value)[0])
            flux = flux.flatten()
            flux[flux<0] = self.rsg_minflux
        else: 
            raise NotImplementedError('Only model "g00" is currently available')

        return flux
    
    def make_lambda_grid(self, npts=500):
        # DUSTY wavelength grid in three log-spaced segments
        n_out = max(22, int(round(0.175*npts)))
        n_mid = npts - 2*n_out
        uv = np.logspace(np.log10(0.01), np.log10(0.6), n_out, endpoint=False)
        mid = np.logspace(np.log10(0.6), np.log10(30), n_mid, endpoint=False)
        fir = np.logspace(np.log10(30), np.log10(3.6e4), n_out)
        # pin the end points exactly; logspace rounding otherwise fails DUSTY's checks
        uv[0], fir[-1] = 0.01, 3.6e4
        grid = np.concatenate([uv, mid, fir])
        if (grid[1:]/grid[:-1]).max() >= 1.5:
            raise ValueError(f'{npts} points is too few for DUSTY (step ratio >= 1.5)')

        return grid

    def read_lambda_grid(self):
        # the wavelength grid the compiled DUSTY binary actually uses
        grid = np.loadtxt(os.path.join(self.dusty_basedir, 'lambda_grid.dat'), skiprows=1)
        return np.unique(grid)

    def validate_dusty_dir(self, rewrite_lambda=False):
        if rewrite_lambda:
            gridfile = os.path.join(self.dusty_basedir, 'lambda_grid.dat')
            parfile = os.path.join(self.dusty_basedir, 'userpar.inc')
            nline = f'npL = {len(self.dusty_lambda_grid)}\n'

            # edit wavelength grid file
            with open(gridfile, 'w') as f:
                f.write(nline)
            with open(gridfile, 'ab') as f:
                np.savetxt(f, self.dusty_lambda_grid)

            # update user parameter file
            with open(parfile, 'r') as f:
                par_lines = f.readlines()
            par_lines[15] = f'      PARAMETER (npL={len(self.dusty_lambda_grid)})\n'
            with open(parfile, 'w') as f:
                f.writelines(par_lines)

            # recompile dusty
            curdir = os.getcwd()
            os.chdir(self.dusty_basedir)
            subprocess.run(['gfortran', 'dustyV2.f', '-std=legacy', '-o', 'dusty'], check=True)
            os.chdir(curdir)

        default_tau = np.array(list(np.arange(0.0, 1.1, 0.1)) + list(np.arange(1.5, 6.0, 0.5)) + list(np.arange(6.0, 13.0, 1.0)) +\
                        list(np.arange(15.0, 33.0, 2.0)) + list(np.arange(35.0, 61.0, 5.0))) 
        # If the first entry is 0, dusty will throw a non-invertible matrix error. Minimum value is 1e-4
        default_tau[0] = 1e-4
        taufile = os.path.join(self.dusty_basedir, 'taugrid.dat')
        exist_tau = np.loadtxt(taufile, skiprows=1, dtype=float)
        if len(exist_tau) != len(default_tau):
            replace_tau = True
        elif (np.abs(exist_tau - default_tau) > 1e-5).any():
            replace_tau = True
        else:
            replace_tau = False

        if replace_tau:
            with open(taufile, 'w') as f:
                f.write('lambda0 = 0.55 micron\n')
                for t in default_tau:
                    f.write(f'   {t}\n')
            self.dusty_n_taugrid = len(default_tau)

        # make sure all required files are present in the dusty directory
        req_files = ['dusty', 'dustyV2.f', 'userpar.inc', 'lambda_grid.dat', 'dusty.inp']
        for fl_ in req_files:
            if not os.path.exists(os.path.join(self.dusty_basedir, fl_)):
                raise ValueError(f"{fl_} not found in {self.dusty_basedir}")

    def extend_bb_tail(self, wv, flux, temp, lam_join=20.0, lam_match=(10.0, 20.0), lam_max=3.6e4, npts=300):
        # Extend the model atmosphere beyond lam_join (micron) with a blackbody at
        # temp, scaled to the model's continuum over lam_match so the join does not
        # hinge on a single flux point. Otherwise DUSTY extrapolates a 
        # lambda*F_lambda ~ lambda^-3 tail from the last input point.
        bb = lambda w: 1.0 / (w**5 * np.expm1(14387.77 / (w * temp)))   # B_lambda, w in micron
        m = (wv >= lam_match[0]) & (wv <= lam_match[1])
        scale = np.median(flux[m] / bb(wv[m]))

        keep = wv <= lam_join
        tail_wv = np.logspace(np.log10(lam_join), np.log10(lam_max), npts)[1:]
        return np.concatenate([wv[keep], tail_wv]), np.concatenate([flux[keep], scale * bb(tail_wv)])

    def cell_average(self, wv, flux, grid):
        # Average F_lambda over each DUSTY grid cell (edges at the geometric midpoints
        # between grid points). DUSTY linearly interpolates the input spectrum at its
        # grid points, so passing cell averages makes it flux-conserving instead of
        # point-sampling the molecular bands. Cells only partly covered by the model
        # are averaged over the covered part; cells outside it are dropped (DUSTY sets
        # the stellar flux to zero below the shortest input wavelength).
        edges = np.concatenate([[grid[0]], np.sqrt(grid[1:]*grid[:-1]), [grid[-1]]])
        cum = np.concatenate([[0.0], np.cumsum(0.5*(flux[1:] + flux[:-1])*np.diff(wv))])
        lo, hi = np.clip(edges[:-1], wv[0], wv[-1]), np.clip(edges[1:], wv[0], wv[-1])
        covered = hi > lo
        integral = np.interp(hi, wv, cum) - np.interp(lo, wv, cum)

        return grid[covered], integral[covered] / (hi - lo)[covered]

    def setup_input_spec(self, temp, outdir, redo=False):
        grid = self.read_lambda_grid()
        # the file depends on the DUSTY grid it was averaged onto
        specfilename = os.path.join(outdir, f'{self.modeltype.lower()}_{temp.value}_npL{len(grid)}.dat')
        if redo or not os.path.exists(specfilename):
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # get marcs model flux, extend with a blackbody beyond 20 micron and
            # average onto the DUSTY wavelength grid
            rsg_flux = self.get_rsg(temp, model='g00')
            rsg_wv = self.rsg_wavelength.to(u.um).value
            rsg_wv, rsg_flux = self.extend_bb_tail(rsg_wv, rsg_flux, temp.to(u.K).value)
            spec_input = np.array(self.cell_average(rsg_wv, rsg_flux, grid)).T

            # write marcs model as input spectrum for dusty
            with open(specfilename, 'w') as f:
                f.write(f'{self.modeltype} model atmosphere for T={temp.value} K\n')
                f.write(f'  lambda    L_lambda\n')
                f.write(f' (micron)  (arbitrary)\n')
            with open(specfilename, 'ab') as f:
                np.savetxt(f, spec_input)

        return specfilename

    def setup_dusty(self, outdir, p):
        #p - specfilename (path), temp (in K), dust_temp (in K), dust_comp ('sil' or 'grf'), shell_thickness (float), tau (at 0.55 micron)
        #setup input file for dusty
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # input file - outdir/rsg_temp_dusttemp.inp
        inp_file = os.path.join(outdir, f'rsg_{p['temp'].value}_{p['dust_temp'].value}.inp')

        #write input file
        with open(inp_file, 'w') as f:
            f.write('  I PHYSICAL PARAMETERS\n')
            f.write('     1) External radiation:\n')
            f.write('                Spectrum = 5\n')
            f.write(f'                {p['input_spec']}\n')
            f.write('     2) Dust Properties\n\n')
            f.write('       2.1 Chemical composition\n')
            f.write('           Optical properties index = 1\n')
            f.write('           Abundances for supported grain types:\n')
            f.write('               Sil-Ow  Sil-Oc  Sil-DL  grf-DL  amC-Hn  SiC-Pg\n')
            if p['dust_comp'].lower() == 'sil' or p['dust_comp'].lower() == 'silicate': # use Draine and Lee silicate dust
                f.write('           x =  0.00    0.00    1.00    0.00    0.00    0.00\n\n')
            elif p['dust_comp'].lower() == 'grf' or p['dust_comp'].lower() == 'graphite': # use Draine and Lee graphite dust
                f.write('           x =  0.00    0.00    0.00    1.00    0.00    0.00\n\n')
            f.write('       2.2 Grain size distribution\n\n')
            f.write('          Size distribution = 2 % arbitrary MRN\n')
            f.write('          q = 3.0, a(min) = 0.05 micron, a(max) = 1.00 micron\n\n')
            f.write('       2.3 Dust temperature on inner boundary:\n\n')
            f.write(f'        - temperature = {p['dust_temp'].value} K\n\n')
            f.write('     3) Density Distribution\n')
            f.write('        - density type = 1\n')
            f.write('        - number of powers = 1\n')
            f.write(f'        - shell\'s relative thickness = {p['shell_thickness']}\n')
            f.write('        - power = 2\n\n')
            f.write('     4) Optical Depth\n')
            f.write('        - grid type = 3\n')
            f.write('            taugrid.dat\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  II NUMERICS\n\n')
            f.write('     - accuracy for flux conservation = 0.05\n\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  III OUTPUT PARAMETERS\n\n')
            f.write('        FILE DESCRIPTION                               FLAG\n')
            f.write('       ------------------------------------------------------------\n')
            f.write('       - verbosity flag;                               verbose = 1\n')
            f.write('       - properties of emerging spectra;             fname.spp = 0\n')
            f.write('       - detailed spectra for each model;           fname.s### = 2\n')
            f.write('       - images at specified wavelengths;           fname.i### = 0\n')
            f.write('       - visibility function at spec. wavelengths;  fname.v### = 0\n')
            f.write('       - radial profiles for each model;            fname.r### = 0\n')
            f.write('       - detailed run-time messages;                fname.m### = 0\n')
            f.write('       -------------------------------------------------------------\n\n')
            f.write('  The end of the input parameters listing.\n')


    def scale_flux(self, wv, flux, scale):
        # Renormalize the RSG spectrum so it's in units of erg/s/cm2/angstrom for
        # synphot to interpret
        normalize = simpson(flux, x=wv.to(u.um).value)
        flux = scale.value * self.FLUX_SCALE * flux/normalize / u.um
        flux = flux.to(u.erg/u.s/u.cm**2/u.Angstrom)

        return flux

    def run_dusty(self, temp, dust_temp,
                  dust_comp='sil', 
                  shell_thickness=2, 
                  filedir=None, 
                  redo_input=False,
                  redo_dusty=False,
                  tb_overwrite=False):
        curdir = os.getcwd()

        self.dust_comp, self.shell_thickness = dust_comp, shell_thickness
        if filedir is None:
            filedir = self.dusty_datadir

        # create input spectrum
        input_spec = self.setup_input_spec(temp=temp, outdir=os.path.join(filedir, 'model_spec'), redo=redo_input)
        # parameter list
        p = {'input_spec': input_spec, 
             'temp' : temp, 
             'dust_temp': dust_temp, 
             'dust_comp': dust_comp, 
             'shell_thickness' : shell_thickness}
        
        # directory structure
        # - root (filedir)
        #   - model_spec
        #       - spectra
        #   - dusty_out
        #       - basename: rsg_temp_dusttemp (outdir)
        #           - basename.inp (inpfile)
        #           - bsaename.out (outfile)
        #           - basename.s### (spec_files)
        #           - basename.hdf5 (dusty_tb_file)

        basename = f'rsg_{p['temp'].value}_{p['dust_temp'].value}'
        outdir = os.path.join(filedir, f'dusty_{dust_comp}_out', basename)
        outfile = os.path.join(outdir, basename + '.out')

        # run dusty
        if redo_dusty or not os.path.exists(outfile):
            self.setup_dusty(outdir=outdir, p=p)

            # write input file in dusty directory
            os.chdir(self.dusty_basedir)
            with open('dusty.inp', 'w') as f:
                f.write(os.path.join(outdir, basename))

            try:
                _ = subprocess.run(['./dusty'], check=True, capture_output=True).stdout.decode()
            except subprocess.CalledProcessError as e:
                raise e
            os.chdir(curdir)

        # read outfile and spectra files
        # BUG: self.dusty_n_taugrid is not updated if taugrid.dat is changed, so we read the number of rows in the outfile instead
        outfile_rows = np.loadtxt(outfile, skiprows=42, max_rows=self.dusty_n_taugrid)
        idx, taus = outfile_rows[:, 0], outfile_rows[:, 1]
        spec_files = [f"{os.path.join(outdir, basename)}.s{int(i):03}" for i in idx]
        taus = outfile_rows[:, 1]

        # create output hdf5 table for spectrum at each tau
        dusty_tb = QTable()
        dusty_tb_file = os.path.join(outdir, basename) + '.hdf5'

        for i, fl_ in enumerate(spec_files):
            t_ = ascii.read(fl_,  names=['lambda', 'fTot', 'xAtt', 'xDs', 'xDe', 'fInp', 'tauT', 'albedo'], 
                            format='basic', data_start=0)
            if i == 0:
                dusty_tb['lambda'] = t_['lambda'] * u.um
                dusty_tb[f'flam_{taus[i]}'] = t_['fTot']/t_['lambda']
            else:
                dusty_tb[f'flam_{taus[i]}'] = t_['fTot']/t_['lambda']

        dusty_tb.write(dusty_tb_file, path='data', serialize_meta=True, overwrite=tb_overwrite)

        return dusty_tb, dusty_tb_file
    
    def dust_spec(self, lum, temp, dust_temp, **kwargs):
        dusty_tb, dusty_tb_file = self.run_dusty(temp, dust_temp, **kwargs)
        scale = 10**lum * u.Lsun
        wv = dusty_tb['lambda']

        # scale spectra from dusty output to the correct luminosity
        for col_ in dusty_tb.colnames[1:]:
            dusty_tb[col_] = self.scale_flux(wv, dusty_tb[col_], scale)

        dusty_tb.write(dusty_tb_file, path='data', serialize_meta=True, overwrite=True)

        return dusty_tb
    
    def gen_grid(self, grid_temps=None, grid_dust_temps=None, grid_taus='default', **kwargs):
        print(f'Creating a {self.dust_comp} DUSTY grid at log(Z)={self.rsg_logz} using {self.modeltype} input spectra')
        if grid_temps is None:
            if self.modeltype=='MARCS':
                grid_temps = np.array(list(np.arange(2500., 4100., 100.)) + list(np.arange(4250., 5250., 250.)))
            if self.modeltype=='MARCS15':
                grid_temps = np.array(list(np.arange(3300., 4100., 100.)) + list(np.array([4250, 4500])))
            if self.modeltype=='NewEra':
                grid_temps = np.arange(2500, 5100, 100)
                
        if grid_dust_temps is None:
            grid_dust_temps = np.array([200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 1800.0])
        
        if grid_taus != 'default':
            if not (isinstance(grid_taus, list) or isinstance(grid_taus, np.ndarray)):
                raise ValueError('grid_taus has to be a list or numpy array, if not "default"')
            taufile = os.path.join(self.dusty_basedir, 'taugrid.dat')
            with open(taufile, 'w') as f:
                f.write('lambda0 = 0.55 micron\n')
                for t in grid_taus:
                    f.write(f'   {t}\n')
            self.dusty_n_taugrid = len(grid_taus)

        for rsg_temp in grid_temps:
            for dust_temp in grid_dust_temps:
                print(f'Generating grid for {rsg_temp}, {dust_temp}')
                tb = self.dust_spec(4.0, rsg_temp*u.K, dust_temp*u.K, dust_comp=self.dust_comp, 
                                    shell_thickness=self.shell_thickness, **kwargs)

class DustyError(RuntimeError):
    '''
    Raised when a DUSTY model fails or does not reach the requested flux
    conservation accuracy, so that a sampler can catch it and reject the point.
    '''
    pass

class dusty_likelihood(dusty_gen):
    '''
    Run single DUSTY models on the fly, so that the output SED can be used directly
    as the model in an MCMC / nested sampling fit.

    Unlike dusty_gen, which writes a full tau grid per (temp, dust_temp) pair, every
    call to model_sed runs exactly one DUSTY model at the requested tau and shell
    thickness. Each process gets its own DUSTY working directory (dusty.inp and
    lambda_grid.dat are read from the cwd), so the model can be evaluated from a
    multiprocessing pool.

    Parameters (theta), in order of PARAM_NAMES, minus any that are fixed:
        logl            : log10(L/Lsun)
        temp            : photospheric Teff (K)
        dust_temp       : dust temperature at the inner boundary T_in (K)
        tau             : optical depth at 0.55 micron
        shell_thickness : relative shell thickness Y = R_out/R_in (fixed at 1e4 by default;
                          pass shell_thickness=None to sample it)
    '''
    PARAM_NAMES = ['logl', 'temp', 'dust_temp', 'tau', 'shell_thickness']

    # Built-in DUSTY grain types, in the column order of the input file
    DUSTY_GRAINS = ['Sil-Ow', 'Sil-Oc', 'Sil-DL', 'grf-DL', 'amC-Hn', 'SiC-Pg']
    GRAIN_ALIASES = {'sil': 'Sil-DL', 'silicate': 'Sil-DL',
                     'grf': 'grf-DL', 'graphite': 'grf-DL',
                     'amc': 'amC-Hn', 'carbon': 'amC-Hn',
                     'sic': 'SiC-Pg'}

    # DUSTY reports r1 for a source with L = 1e4 Lsun
    DUSTY_LUM = 1e4 * u.Lsun

    def __init__(self,
                 dist=10*u.Mpc,
                 logz=0.00,
                 logg=0.00,
                 comp=None,
                 nk_comp=None,
                 size_dist='mrn',
                 size_par=None,
                 density='rdw',
                 density_power=2,
                 shell_thickness=1e4,
                 fixed=None,
                 interp_method='pchip',
                 workdir=None,
                 keep_files=False,
                 accuracy=0.05,
                 outdir='../data/dusty_sil_grid',
                 modeldir='../data/marcs/m1.00_g0.00'):

        super().__init__(dist=dist, logz=logz, logg=logg, interp_method=interp_method,
                         rewrite_lambda_grid=False, modeldir=modeldir, outdir=outdir)
        self.logz = logz
        self.outdir = Path(outdir)
        self.modeldir = Path(modeldir)

        # Dust properties are fixed per fit 
        # comp sets the built-in grains; nk_comp optionally adds grains from
        # optical-constant (.nk) files, e.g. comp={'sic': 0.2}, nk_comp={'amC-zb2.nk': 0.8}.
        # With no composition given at all, fall back to pure silicate.
        if comp is None and not nk_comp:
            comp = 'sil'
        self.comp = comp
        self.abundances, self.nk_abundances = self.parse_comp(comp, nk_comp)
        self.size_dist = size_dist.lower()
        if self.size_dist == 'mrn':
            # modified MRN; amin = amax gives a single grain size
            self.size_par = {'q': 3.0, 'amin': 0.05, 'amax': 1.00}
        elif self.size_dist == 'kmh':
            self.size_par = {'q': 3.5, 'amin': 0.005, 'a0': 0.2}
        else:
            raise ValueError('size_dist must be "mrn" or "kmh"')
        if size_par is not None:
            self.size_par.update(size_par)
        self.density = density.lower()
        if self.density not in ('rdw', 'powerlaw'):
            raise ValueError('density must be "rdw" (analytic radiatively driven wind) or "powerlaw"')
        self.density_power = density_power

        # required fractional accuracy of bolometric flux conservation (DUSTY q_acc).
        # DUSTY reports errors < 1% as 0, so the post-run check only bites for accuracy >= 0.01
        self.accuracy = accuracy

        # Fixed parameters are removed from theta, e.g. fixed={'temp': 3000*u.K}
        self.fixed = {} if shell_thickness is None else {'shell_thickness': shell_thickness}
        for k, v in (fixed or {}).items():
            if k not in self.PARAM_NAMES:
                raise ValueError(f'Unknown parameter {k}, must be one of {self.PARAM_NAMES}')
            self.fixed[k] = v.value if isinstance(v, u.Quantity) else v
        self.free_params = [p for p in self.PARAM_NAMES if p not in self.fixed]

        # Per-process DUSTY working directories are created lazily (after a pool forks)
        self.workdir_base = workdir
        self.keep_files = keep_files
        self._workdirs = {}
        self._input_spec = {}

    def parse_comp(self, comp, nk_comp=None):
        # comp is either a single built-in grain type ('sil', 'amc', ...) or a dict
        # of abundances, e.g. {'amc': 0.9, 'sic': 0.1}. nk_comp is a dict of
        # {.nk file: abundance}. Abundances are normalized over both together, as
        # DUSTY does.
        if comp is None:
            comp = {}
        elif isinstance(comp, str):
            comp = {comp: 1.0}

        abundances = dict.fromkeys(self.DUSTY_GRAINS, 0.0)
        for grain, frac in comp.items():
            name = self.GRAIN_ALIASES.get(grain.lower(), grain)
            if name not in abundances:
                raise ValueError(f'Unknown grain type {grain}, must be one of '
                                 f'{self.DUSTY_GRAINS} or {list(self.GRAIN_ALIASES)}')
            if frac < 0:
                raise ValueError(f'Abundance of {grain} must be non-negative')
            abundances[name] += frac

        nk_abundances = {}
        for nkfile, frac in (nk_comp or {}).items():
            if frac < 0:
                raise ValueError(f'Abundance of {nkfile} must be non-negative')
            path = self.find_nk_file(nkfile)
            nk_abundances[path] = nk_abundances.get(path, 0.0) + frac
        # DUSTY reads at most 10 additional components (nameNK(10))
        if len(nk_abundances) > 10:
            raise ValueError('DUSTY supports at most 10 additional (.nk) components')

        total = sum(abundances.values()) + sum(nk_abundances.values())
        if total <= 0:
            raise ValueError('At least one grain type needs a non-zero abundance')

        return ({k: v/total for k, v in abundances.items()},
                {k: v/total for k, v in nk_abundances.items()})

    def find_nk_file(self, nkfile):
        # DUSTY opens .nk files relative to its working directory (a temp dir here),
        # so resolve to an absolute path: as given, then $DUSTY_PATH/Lib_nk, then $DUSTY_PATH
        for path in [Path(nkfile),
                     Path(self.dusty_basedir) / 'Lib_nk' / nkfile,
                     Path(self.dusty_basedir) / nkfile]:
            if path.is_file():
                path = str(path.resolve())
                # FileMSG in dustyV2.f reads file names into a CHARACTER*230
                if len(path) >= 230:
                    raise ValueError(f'Path to {nkfile} is too long for DUSTY (>= 230 characters)')
                return path

        raise FileNotFoundError(f'{nkfile} not found as given, in {self.dusty_basedir}/Lib_nk '
                                f'or in {self.dusty_basedir}')

    def get_params(self, theta):
        # map the free parameter vector onto the full parameter dictionary
        if len(theta) != len(self.free_params):
            raise ValueError(f'Expected {len(self.free_params)} free parameters '
                             f'{self.free_params}, got {len(theta)}')
        p = dict(self.fixed)
        p.update(zip(self.free_params, theta))

        if not (self.rsg_temp.value.min() <= p['temp'] <= self.rsg_temp.value.max()):
            raise ValueError(f'temp = {p['temp']} K is outside the {self.modeltype} grid '
                             f'({self.rsg_temp.value.min()}-{self.rsg_temp.value.max()} K)')
        if p['tau'] <= 0:
            raise ValueError('tau must be positive (DUSTY fails at tau = 0)')
        if p['shell_thickness'] <= 1:
            raise ValueError('shell_thickness (Y = R_out/R_in) must be > 1')

        return p

    def get_workdir(self):
        # one DUSTY working directory per process, since DUSTY reads dusty.inp and
        # lambda_grid.dat from its cwd
        pid = os.getpid()
        if pid not in self._workdirs:
            if self.workdir_base is not None:
                os.makedirs(self.workdir_base, exist_ok=True)
            workdir = tempfile.mkdtemp(prefix=f'dusty_{pid}_', dir=self.workdir_base)
            for fl_ in ['dusty', 'lambda_grid.dat']:
                os.symlink(os.path.join(self.dusty_basedir, fl_), os.path.join(workdir, fl_))
            self._workdirs[pid] = workdir

        return self._workdirs[pid]

    def get_input_spec(self, temp, workdir):
        # input spectra are cached per process, since temp is usually fixed or
        # tightly priored
        key = (os.getpid(), float(temp))
        if key not in self._input_spec:
            self._input_spec[key] = self.setup_input_spec(temp=temp*u.K, outdir=workdir)

        return self._input_spec[key]

    def dusty_comp_block(self):
        # 2.1 chemical composition. Built-in grains only -> optical properties
        # index 1; with additional .nk files -> index 2, which appends the file
        # list and their abundances after the built-in abundances (DUSTY manual 3.2.1)
        top = 2 if self.nk_abundances else 1
        lines = ['       2.1 Chemical composition\n',
                 f'           Optical properties index = {top}\n',
                 '           Abundances for supported grain types:\n',
                 '               Sil-Ow  Sil-Oc  Sil-DL  grf-DL  amC-Hn  SiC-Pg\n',
                 '           x = ' + '  '.join(f'{self.abundances[g]:.4f}' for g in self.DUSTY_GRAINS) + '\n\n']

        if self.nk_abundances:
            lines.append(f'           Number of additional components = {len(self.nk_abundances)}, '
                         'properties listed in files\n')
            for path in self.nk_abundances:
                lines.append(f'               {path}\n')
            lines.append('           Abundances for these components = ' +
                         ', '.join(f'{x:.4f}' for x in self.nk_abundances.values()) + '\n\n')

        return ''.join(lines)

    def dusty_size_block(self):
        # 2.2 grain size distribution
        lines = ['       2.2 Grain size distribution\n\n']
        if self.size_dist == 'mrn':
            lines.append('          Size distribution = 2 % arbitrary MRN\n')
            lines.append(f'          q = {self.size_par['q']}, a(min) = {self.size_par['amin']} micron, '
                         f'a(max) = {self.size_par['amax']} micron\n\n')
        else:
            lines.append('          Size distribution = 3 % KMH\n')
            lines.append(f'          q = {self.size_par['q']}, a(min) = {self.size_par['amin']} micron, '
                         f'a0 = {self.size_par['a0']} micron\n\n')

        return ''.join(lines)

    def dusty_density_block(self, p):
        # 3) density distribution. DUSTY input density type 4 is the analytic
        # radiatively driven wind and only takes the relative thickness Y; 
        # type 1 is a single power law
        lines = ['     3) Density Distribution\n']
        if self.density == 'rdw':
            lines.append('        - density type = 4\n')
            lines.append(f'        - Y = {p['shell_thickness']}\n\n')
        else:
            lines.append('        - density type = 1\n')
            lines.append('        - number of powers = 1\n')
            lines.append(f'        - shell\'s relative thickness = {p['shell_thickness']}\n')
            lines.append(f'        - power = {self.density_power}\n\n')

        return ''.join(lines)

    def setup_dusty_single(self, inp_file, p):
        # setup input file for a single DUSTY model at p['tau']
        with open(inp_file, 'w') as f:
            f.write('  I PHYSICAL PARAMETERS\n')
            f.write('     1) External radiation:\n')
            f.write('                Spectrum = 5\n')
            f.write(f'                {p['input_spec']}\n')
            f.write('     2) Dust Properties\n\n')
            f.write(self.dusty_comp_block())
            f.write(self.dusty_size_block())
            f.write('       2.3 Dust temperature on inner boundary:\n\n')
            f.write(f'        - temperature = {p['dust_temp']} K\n\n')
            f.write(self.dusty_density_block(p))
            f.write('     4) Optical Depth\n')
            f.write('        - grid type = 1\n')
            f.write('        - lambda0 = 0.55 micron\n')
            f.write(f'        - tau(min) = {p['tau']}; tau(max) = {p['tau']}\n')
            f.write('        - number of models = 1\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  II NUMERICS\n\n')
            f.write(f'     - accuracy for flux conservation = {self.accuracy}\n\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  III OUTPUT PARAMETERS\n\n')
            f.write('        FILE DESCRIPTION                               FLAG\n')
            f.write('       ------------------------------------------------------------\n')
            f.write('       - verbosity flag;                               verbose = 0\n')
            f.write('       - properties of emerging spectra;             fname.spp = 0\n')
            f.write('       - detailed spectra for each model;           fname.s### = 2\n')
            f.write('       - images at specified wavelengths;           fname.i### = 0\n')
            f.write('       - visibility function at spec. wavelengths;  fname.v### = 0\n')
            f.write('       - radial profiles for each model;            fname.r### = 0\n')
            f.write('       - detailed run-time messages;                fname.m### = 2\n')
            f.write('       -------------------------------------------------------------\n\n')
            f.write('  The end of the input parameters listing.\n')

    def read_dusty_results(self, outfile):
        # read the single-model row of the RESULTS table in the .out file, keyed
        # by the column names in the header (e.g. 'r1(cm)', 'Td(Y)', 'err')
        with open(outfile) as f:
            lines = f.readlines()

        for n, line in enumerate(lines):
            if line.lstrip().startswith('###') and 'tau0' in line:
                names = line.split()[1:]
                break
        else:
            raise DustyError(f'No RESULTS table in {outfile}, the DUSTY run probably failed')

        for line in lines[n+1:]:
            s = line.split()
            if s and s[0].isdigit():
                return dict(zip(names, [float(v) for v in s[1:len(names)+1]]))

        raise DustyError(f'RESULTS table in {outfile} has no models')

    def check_dusty_status(self, outfile, msgfile, results, p):
        # DUSTY flags problems in the .out file and details them in the .m### file.
        # Errors and failed flux conservation raise DustyError; warnings are returned.
        with open(outfile) as f:
            out = f.read()
        messages = ''
        if os.path.exists(msgfile):
            with open(msgfile) as f:
                messages = f.read().strip()

        if 'FATAL ERROR' in out or 'error messages for model' in out:
            raise DustyError(f'DUSTY reported errors for {p}:\n{messages}')

        # err is the maximum flux conservation error in %, (1-r)/(1+r) with r the
        # ratio of smallest to largest bolometric flux in the shell (< 1% is listed as 0)
        if results.get('err', 0.0) > 100*self.accuracy:
            raise DustyError(f'DUSTY flux conservation error {results['err']}% exceeds '
                             f'the required {100*self.accuracy}% for {p}')

        warnings = messages if 'warnings for model' in out else ''
        return warnings

    def run_dusty_single(self, p):
        workdir = self.get_workdir()
        p['input_spec'] = self.get_input_spec(p['temp'], workdir)

        basename = os.path.join(workdir, 'model')
        inp_file, outfile, specfile = basename + '.inp', basename + '.out', basename + '.s001'
        msgfile = basename + '.m001'
        for fl_ in [outfile, specfile, msgfile]:
            if os.path.exists(fl_):
                os.remove(fl_)

        self.setup_dusty_single(inp_file, p)
        with open(os.path.join(workdir, 'dusty.inp'), 'w') as f:
            f.write(basename)

        run = subprocess.run(['./dusty'], cwd=workdir, capture_output=True)
        if run.returncode != 0 or not os.path.exists(specfile):
            raise DustyError(f'DUSTY failed for {p}:\n{run.stdout.decode()}{run.stderr.decode()}')

        results = self.read_dusty_results(outfile)
        results['warnings'] = self.check_dusty_status(outfile, msgfile, results, p)
        spec = ascii.read(specfile, names=['lambda', 'fTot', 'xAtt', 'xDs', 'xDe', 'fInp', 'tauT', 'albedo'],
                          format='basic', data_start=0)

        if not self.keep_files:
            for fl_ in [inp_file, outfile, specfile, msgfile]:
                if os.path.exists(fl_):
                    os.remove(fl_)

        return spec, results

    def model_sed(self, theta):
        '''
        Primary model function: free parameters -> observed SED at self.dist.

        Returns a QTable with 'lambda' (micron), 'flam' (erg/s/cm2/A) and 'fnu' (uJy)
        for the total (attenuated star + scattered + dust emission) spectrum. The
        DUSTY results are stored in the table meta, with r_in scaled to logl.
        '''
        p = self.get_params(theta)
        spec, results = self.run_dusty_single(p)

        # the stitched DUSTY wavelength grid repeats the boundary wavelengths
        wv, idx = np.unique(np.asarray(spec['lambda']), return_index=True)
        wv = wv * u.um

        # fTot is lambda*F_lambda/F_bol; scale to the requested luminosity
        flam = np.asarray(spec['fTot'])[idx] / wv.value
        flam = self.scale_flux(wv, flam, 10**p['logl'] * u.Lsun)
        fnu = flam.to(u.uJy, equivalencies=u.spectral_density(wv))

        sed = QTable()
        sed['lambda'] = wv
        sed['flam'] = flam
        sed['fnu'] = fnu

        sed.meta = {k: p[k] for k in self.PARAM_NAMES}
        sed.meta['comp'] = self.abundances
        sed.meta['nk_comp'] = self.nk_abundances
        sed.meta['size_dist'] = self.size_dist
        sed.meta['size_par'] = self.size_par
        sed.meta['density'] = self.density
        if self.density == 'powerlaw':
            sed.meta['density_power'] = self.density_power
        sed.meta['dusty_results'] = results
        # R_in scales as L^1/2 at fixed T_in
        r1 = results.get('r1(cm)', np.nan) * u.cm
        sed.meta['r_in'] = r1 * np.sqrt((10**p['logl'] * u.Lsun / self.DUSTY_LUM).decompose().value)
        sed.meta['dusty_err'] = results.get('err', 0)
        sed.meta['dusty_warnings'] = results['warnings']

        return sed

    def model_phot(self, theta, bandpasses, sed=None):
        '''
        Synthetic photometry (uJy) of model_sed(theta) through a dict of
        {filter name: synphot.SpectralElement}.
        '''
        if sed is None:
            sed = self.model_sed(theta)

        wv = sed['lambda'].to(u.Angstrom)
        sp = synphot.SourceSpectrum(Empirical1D, points=wv, lookup_table=sed['flam'])

        phot = {}
        for filt, bp in bandpasses.items():
            # taper the bandpass to zero at its ends: curves that stop at non-zero
            # transmission (pandeia MIRI F2550W ends at 30 um with T ~ 0.6) are otherwise
            # extrapolated over the whole long-wavelength model grid
            obs = synphot.Observation(sp, bp.taper(), force='taper', binset=wv)
            phot[filt] = obs.effstim(u.uJy).value

        return phot


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    logz = args.logz
    comp = args.comp
    outdir = args.outdir
    modeldir = args.modeldir
    rewrite_lambda_grid = args.rewrite_lambda_grid

    dustgen = dusty_gen(dist = 10*u.Mpc, logz=logz, outdir=outdir, modeldir=modeldir, rewrite_lambda_grid=rewrite_lambda_grid)
    dustgen.dust_comp = comp

    dustgen.gen_grid(redo_dusty=False, redo_input=False, tb_overwrite=True)