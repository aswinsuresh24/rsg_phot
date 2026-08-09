import os

if __name__ == '__main__':
    for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_var, '1')

import argparse
import pandeia.engine
import stsynphot
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
from synphot.exceptions import SynphotError
import glob, sys, shutil, traceback
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
from rsg_phot.utils import logger

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
    parser.add_argument('--norm', default=False, action=argparse.BooleanOptionalAction,
                        help='Normalize spectra?')
    parser.add_argument('--nproc', type=int, default=None,
                        help='Number of worker processes (default: cores allocated to the job)')
    return parser

def read_dusty_taus(outfile):
    '''
    Read the tau0 grid from the RESULTS table of a DUSTY .out file.

    Parameters
    ----------
    outfile : str
        Path to a DUSTY .out file

    Returns
    -------
    taus : np.ndarray
        Strictly ascending array of tau0 values

    Raises
    ------
    ValueError
        If the results table is missing (e.g. the DUSTY run crashed) or the
        tau values are not strictly ascending.
    '''
    with open(outfile) as f:
        lines = f.readlines()

    # Locate the table header rather than hard-coding a line offset
    start = None
    for n, line in enumerate(lines):
        if line.lstrip().startswith('###') and 'tau0' in line:
            start = n
            break

    if start is None:
        raise ValueError(f'No RESULTS table found in {outfile} - the DUSTY '
                          'run probably failed, check the file for errors')

    taus = []
    for line in lines[start+1:]:
        s = line.strip()
        # Skip the second '###' header row and the opening '=====' rule;
        # the matching closing rule terminates the table.
        if not s or s.startswith('###') or set(s) == {'='}:
            if taus:
                break
            continue
        parts = s.split()
        try:
            int(parts[0])
            tau = float(parts[1])
        except (ValueError, IndexError):
            break
        taus.append(tau)

    if not taus:
        raise ValueError(f'RESULTS table in {outfile} contains no tau values')

    taus = np.array(taus)

    # RegularGridInterpolator requires strictly ascending axes
    if not np.all(np.diff(taus) > 0):
        raise ValueError(f'tau values in {outfile} are not strictly ascending: {taus}')

    return taus

def available_cpus():
    '''
    Number of CPUs actually available to this process.

    os.cpu_count() reports the whole machine, which oversubscribes badly on a
    scheduler-allocated cluster node, so prefer the job allocation.
    '''
    n = os.environ.get('SLURM_CPUS_PER_TASK')
    if n:
        return int(n)
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1

# Magnitude recorded when a model is extinguished to identically zero flux
# across a whole bandpass, so effstim has no finite magnitude to return. The
# grid has to stay finite for RegularGridInterpolator, and mag > 90 makes the
# affected corner trivial to spot downstream.
FAINT_MAG = 99.0

# Worker-process state, populated once per process by the pool initializer so
# that the rsg_phot instance and the bandpasses are not re-pickled per task.
_WORKER = {}

def _init_worker(rsg_obj, static):
    # Rebuild the bandpasses in the worker rather than shipping them from the
    # parent. Pickling a synphot Empirical1D silently changes its behaviour
    # outside the waveset from holding the edge value to returning NaN, which
    # makes Observation.effstim raise 'Integrated flux is NaN' for any filter
    # whose support extends past the model spectrum (e.g. MIRI F2550W).
    _WORKER['rsg'] = rsg_obj
    _WORKER['static'] = static
    _WORKER['bp_cache'] = {flt: rsg_obj.get_jwst_filters(inst + ',' + flt)
                           for inst, flts in rsg_obj.jwst_filts.items() for flt in flts}

def _pair_worker(task):
    i, j, te, td = task
    s = _WORKER['static']
    pair_mags = _WORKER['rsg']._compute_pair_mags(
        s['subdir'], te, td, s['ntau'], s['norm'], s['loglums'],
        s['rv'], s['av'], _WORKER['bp_cache'], s['all_filts'])
    return i, j, pair_mags

class rsg_phot(object):
    def __init__(self, verbose=True, interpolate=True):
        self.filename = ''

        try:
            self.pandeia = os.environ['pandeia_refdata']
        except:
            logger.warning('if using JWST filters, set PANDEIA env variable')
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
        self.miri_filts = ['F0560W','F0770W','F1000W','F1130W','F1280W','F1500W',
                           'F1800W','F2100W','F2550W']
        self.jwst_filts = {'NIRCAM': self.nrc_filts, 'MIRI': self.miri_filts}
                
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
    
    def _compute_pair_mags(self, subdir, te, td, ntau, norm, loglums, rv, av, bp_cache, all_filts):
        '''
        Compute the (tau, loglum, rv, av) magnitude cube for a single
        (temp, dust_temp) grid point. Independent of every other grid
        point, so this is the unit of work handed to worker processes.
        '''
        sd = f'rsg_{te}_{td}'
        specfile = os.path.join(subdir, sd, sd+'.hdf5')
        spectable = read_table_hdf5(specfile)
        wv = spectable['lambda'].to(u.Angstrom)
        _, mask = np.unique(wv, return_index=True)
        wv = wv[mask]
        energy = (const.h*const.c/wv).to(u.erg)
        logger.info(specfile)

        # One flux column per tau; a mismatch means the spectra and the .out
        # table disagree, which would silently mis-index the tau axis.
        cols = list(spectable.columns[1:])
        if len(cols) != ntau:
            raise ValueError(f'{specfile} has {len(cols)} flux columns but the '
                             f'DUSTY .out table lists {ntau} optical depths')

        # precompute host extinction
        host_ext_grid = {
            (m, n): self.extinction_law(wv.value, Av_, Rv_)
            for m, Rv_ in enumerate(rv) for n, Av_ in enumerate(av)
        }

        pair_shape = (ntau, len(loglums), len(rv), len(av))
        pair_mags = {flt: np.zeros(pair_shape) for flt in all_filts}
        nfaint = 0

        for k, col in enumerate(cols):
            if norm:
                flux = spectable[col][mask]
                normalize = simpson(flux, x=wv.to(u.um).value)
                flux = self.flux_scale * flux/normalize / u.micron
                flux = flux.to(u.erg/u.s/u.cm**2/u.Angstrom)
            else:
                # Fail loudly: falling through here would silently reuse the
                # previous tau's flux and write wrong mags into the grid.
                flux = spectable[col][mask]
                try:
                    flux = flux.to(u.erg/u.s/u.cm**2/u.Angstrom)
                except u.UnitConversionError as e:
                    raise ValueError(f'Incorrect flux units in {specfile} column '
                                     f'{col!r}: cannot convert {flux.unit} to '
                                     'erg/s/cm2/Angstrom') from e

            for m in range(len(rv)):
                for n in range(len(av)):
                    host_ext = host_ext_grid[(m, n)]
                    flux_ext = flux*host_ext
                    sp = synphot.SourceSpectrum(Empirical1D, points=wv, lookup_table=(flux_ext/energy).value)

                    for flt in all_filts:
                        bp = bp_cache[flt]
                        kwargs = {'force': 'taper', 'binset': wv}
                        obs = synphot.Observation(sp, bp, **kwargs)
                        try:
                            mag = obs.effstim(self.magsystem).value
                        except SynphotError:
                            # The coolest models at the highest tau are
                            # extinguished to exactly zero flux across the
                            # bluest bands, so there is no magnitude to
                            # compute. That is a property of the model, not a
                            # failure, so record the sentinel and carry on
                            # rather than killing the whole grid run.
                            mag = FAINT_MAG
                            nfaint += 1

                        for l, logl in enumerate(loglums):
                            logl = logl - 4
                            scale_mag = mag-2.5*logl
                            pair_mags[flt][k, l, m, n] = scale_mag

        if nfaint:
            ncell = ntau * len(rv) * len(av) * len(all_filts)
            logger.warning(f'{os.path.basename(specfile)}: {nfaint}/{ncell} (tau, Rv, Av, filter) '
                           f'cells had zero flux in band, set to mag {FAINT_MAG}')

        return pair_mags

    def create_rsg_grid(self, subdir, modelname, outdir='data/interpolate', norm=False,
                         loglums=None, av=None, rv=None, nproc=None):

        all_grids = sorted(glob.glob(subdir + '/rsg*'))
        if not all_grids:
            raise ValueError(f'No rsg* model directories found in {subdir}')

        grid_temps = np.unique([float(os.path.basename(i).split('_')[1]) for i in all_grids])
        grid_dust_temps = np.unique([float(os.path.basename(i).split('_')[2]) for i in all_grids])

        # The tau axis is a property of the DUSTY runs, so read it from the
        # .out tables instead of assuming a count. Every pair must agree,
        # otherwise there is no single tau axis to interpolate over.
        grid_taus = None
        for i, te in enumerate(grid_temps):
            for j, td in enumerate(grid_dust_temps):
                sd = f'rsg_{te}_{td}'
                taus = read_dusty_taus(os.path.join(subdir, sd, sd+'.out'))
                if grid_taus is None:
                    grid_taus, tau_ref = taus, sd
                elif not np.array_equal(taus, grid_taus):
                    raise ValueError(f'tau grid in {sd} ({len(taus)} values) does not '
                                     f'match {tau_ref} ({len(grid_taus)} values); all '
                                     'DUSTY runs must share one tau grid')
        ntau = len(grid_taus)
        logger.info(f'Read {ntau} optical depths from the DUSTY .out tables')

        if loglums is None:
            loglums = np.array([3, 8])
        if rv is None:
            rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        if av is None:
            av = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])

        all_filts = self.nrc_filts + self.miri_filts

        # Cache bandpasses for the serial path. Workers rebuild their own (see
        # _init_worker), but building them here first also fails fast if the
        # filter files are missing, before any pairs are computed.
        bp_cache = {}
        for inst in self.jwst_filts.keys():
            for flt in self.jwst_filts[inst]:
                bp_cache[flt] = self.get_jwst_filters(inst + ',' + flt)

        grid_shape = (len(grid_temps), len(grid_dust_temps), ntau, len(loglums), len(rv), len(av))

        # Checkpoint as one small file per (temp, dust_temp) pair. Rewriting
        # the whole grid after every pair would cost O(grid) I/O per pair,
        # which for a full run is tens of GB of redundant writes.
        ckpt_dir = os.path.join(outdir, f'{modelname}_checkpoint')
        os.makedirs(ckpt_dir, exist_ok=True)

        # A resumed run must describe the same grid, or the reused pair files
        # would be silently mixed into a grid they do not belong to.
        meta = {'grid_temps': grid_temps, 'grid_dust_temps': grid_dust_temps,
                'grid_taus': grid_taus, 'loglums': loglums, 'rv': rv, 'av': av,
                'norm': bool(norm), 'filters': all_filts}
        meta_file = os.path.join(ckpt_dir, 'meta.pkl')
        if os.path.exists(meta_file):
            with open(meta_file, 'rb') as f:
                old = pickle.load(f)
            for key, val in meta.items():
                if not np.array_equal(np.asarray(old.get(key)), np.asarray(val)):
                    raise ValueError(
                        f'Checkpoint in {ckpt_dir} was built with a different {key} '
                        f'({old.get(key)!r} vs {val!r}). Delete the directory to '
                        'rebuild from scratch.')
        else:
            with open(meta_file, 'wb') as f:
                pickle.dump(meta, f)

        def pair_path(i, j):
            return os.path.join(ckpt_dir, f'pair_{i:04d}_{j:04d}.npz')

        def save_pair(i, j, pair_mags):
            tmp = pair_path(i, j) + '.tmp'
            with open(tmp, 'wb') as f:
                np.savez(f, **pair_mags)
            os.replace(tmp, pair_path(i, j))

        all_pairs = [(i, j, te, td) for i, te in enumerate(grid_temps)
                     for j, td in enumerate(grid_dust_temps)]
        pending = [p for p in all_pairs if not os.path.exists(pair_path(p[0], p[1]))]

        ndone = len(all_pairs) - len(pending)
        if ndone:
            logger.info(f'Resuming: {ndone}/{len(all_pairs)} (temp, dust_temp) pairs already done')

        if nproc is None:
            nproc = available_cpus()
        nproc = max(1, min(nproc, len(pending) or 1))
        logger.info(f'Computing {len(pending)} pairs on {nproc} process(es)')

        # Each (temp, dust_temp) pair is independent, so farm them out across
        # processes. Each result is checkpointed as it lands, so a timeout
        # only loses the pairs still in flight.
        failed, nsaved = [], 0
        if nproc > 1:
            import concurrent.futures as cf
            static = {'subdir': subdir, 'ntau': ntau, 'norm': norm, 'loglums': loglums,
                      'rv': rv, 'av': av, 'all_filts': all_filts}
            # Set up each worker once via the initializer rather than
            # re-pickling the configuration with every task.
            with cf.ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker,
                                        initargs=(self, static)) as pool:
                futures = {pool.submit(_pair_worker, (i, j, te, td)): (i, j, te, td)
                           for i, j, te, td in pending}
                try:
                    for fut in cf.as_completed(futures):
                        i, j, te, td = futures[fut]
                        try:
                            _, _, pair_mags = fut.result()
                        except Exception:
                            # A pair that raises must not discard the pairs that
                            # did finish, so record it and keep collecting.
                            logger.error(f'pair rsg_{te}_{td} failed:\n{traceback.format_exc()}')
                            failed.append(f'rsg_{te}_{td}')
                            continue
                        save_pair(i, j, pair_mags)
                        nsaved += 1
                        logger.info(f'saved rsg_{te}_{td} ({nsaved}/{len(pending)})')
                finally:
                    # shutdown() defaults to cancel_futures=False, which keeps
                    # feeding every queued task to the workers before returning.
                    # Leaving this loop early would otherwise run the whole
                    # remaining grid for results that nobody collects.
                    pool.shutdown(wait=False, cancel_futures=True)
        else:
            for i, j, te, td in pending:
                try:
                    pair_mags = self._compute_pair_mags(subdir, te, td, ntau, norm,
                                                        loglums, rv, av, bp_cache, all_filts)
                except Exception:
                    logger.error(f'pair rsg_{te}_{td} failed:\n{traceback.format_exc()}')
                    failed.append(f'rsg_{te}_{td}')
                    continue
                save_pair(i, j, pair_mags)
                nsaved += 1
                logger.info(f'saved rsg_{te}_{td} ({nsaved}/{len(pending)})')

        # There is no grid to interpolate with pairs missing, but the ones that
        # did finish stay checkpointed so a rerun only redoes the failures.
        if failed:
            raise RuntimeError(
                f'{len(failed)}/{len(pending)} pairs failed: {", ".join(failed)}. '
                f'{nsaved} pair(s) are checkpointed in {ckpt_dir}; fix the errors '
                'above and rerun to resume from there.')

        # Assemble the full grid from the per-pair checkpoints
        mags = {flt: np.zeros(grid_shape) for flt in all_filts}
        for i, j, _, _ in all_pairs:
            with np.load(pair_path(i, j)) as pair:
                for flt in all_filts:
                    mags[flt][i, j] = pair[flt]

        models = {}
        params = (grid_temps, grid_dust_temps, grid_taus, loglums, rv, av)

        for flt in all_filts:
            models[flt] = interpolate.RegularGridInterpolator(params, mags[flt], method='linear', bounds_error=True)

        os.makedirs(outdir, exist_ok=True)
        pfile = os.path.join(outdir, f'{modelname}.pkl')
        pickle.dump(models, open(pfile, 'wb'))
        logger.info(f'Wrote {pfile}')

        shutil.rmtree(ckpt_dir, ignore_errors=True)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    subdir = args.subdir
    outdir = args.outdir
    modelname = args.modelname
    norm = args.norm
    nproc = args.nproc

    rsg = rsg_phot()
    rsg.create_rsg_grid(subdir, modelname, outdir, norm, nproc=nproc)