import numpy as np
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob, re, argparse
import rsg_phot.dust
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import math
import pickle
import random
import time
from scipy.stats import truncexpon
from scipy.special import log_ndtr
from pathlib import Path
from rsg_phot.mcmc import NRC_FILTS, MIRI_FILTS, MODE_FILTS

# AB magnitude of 1 uJy: m_AB = 23.9 - 2.5 log10(F_nu / uJy)
AB_ZP_UJY = 23.9

def mu_to_dist(dm, dmerr):
    # distance and its uncertainty (Mpc) from a distance modulus, as in mcmc.mu_to_dist
    d = 10**(dm/5+1.0) * u.pc
    de = (10**((dm+dmerr)/5+1.0) - 10**((dm-dmerr)/5+1.0)) * u.pc
    return [d.to(u.Mpc).value, de.to(u.Mpc).value]

def mag_to_flux(mag, magerr=None):
    '''
    AB magnitudes to F_nu in uJy, with first-order errors sigma_F = 0.921 F sigma_m.
    '''
    flux = 10**((AB_ZP_UJY - np.asarray(mag, dtype=float)) / 2.5)
    if magerr is None:
        return flux
    return flux, flux * np.log(10) / 2.5 * np.asarray(magerr, dtype=float)

def flux_to_mag(flux):
    return AB_ZP_UJY - 2.5 * np.log10(flux)

class nest(object):
    '''
    Nested sampling (dynesty) counterpart of rsg_phot.mcmc.mcmc.

    model = 'grid' samples the interpolated magnitude grids that mcmc uses (same
    pickles, parameters, bounds and priors). model = a rsg_phot.dust.dusty_likelihood
    instance runs DUSTY on the fly instead: its free parameters are sampled (under the
    mcmc names below), extinction is applied to the model SED, and synthetic photometry
    uses the JWST bandpasses (or the bandpasses passed in, e.g. for HST filters).
    model = a rsg_phot.dust.DustyGrid (or the path to its .npz) samples the precomputed
    DUSTY grid instead, in (temperature, dust_temp, tau_V, s, Y, luminosity), with
    bounds from the grid axes and log L in DustyGrid.LOGL_RANGE; a wind + shell grid
    (rsg_phot.dust_twoshell.TwoShellGrid, chosen from the .npz metadata) is sampled in its
    own axes (dust_temp, tau_V, f_wind, y_shell, gap, Y_shell, luminosity). The grid band names
    must match phot['inst_filt']. Its photometry is unreddened, so extinction is
    applied per band at the bandpass pivot wavelength, which differs from reddening
    the SED by < 0.01 mag per magnitude of A_V for these bands.

    space = 'mag' fits AB magnitudes with a Gaussian likelihood in magnitude, like mcmc.
    space = 'flux' fits F_nu in uJy: Gaussian for detections and a censored term,
    log Phi((F_lim - F_model) / sigma_lim), for upper limits. uJy keeps fluxes and
    variances O(1e-3 - 10) for these sources, so nothing underflows even in float32
    (F_lambda variances are ~1e-44 in erg/s/cm2/A); it is also what model_phot returns.

    Photometry dict, as for mcmc: phot['index'], phot['inst_filt'] and either
        phot['mag'], phot['magerr']        (AB; converted when space='flux'), or
        phot['flux'], phot['fluxerr']      (uJy; space='flux' only, negative fluxes allowed)
    plus optionally phot['uplim'], a boolean mask of upper limits. For a limit, mag (or
    flux) holds the limiting value, and its sigma is F_lim / nsigma_lim unless fluxerr
    gives it.

    reduced_chi2 = True reproduces mcmc's log-likelihood, -chi^2/2/(N-1) (magnitude
    space only). That tempering broadens the posterior by ~sqrt(N-1) and makes the
    evidence meaningless, so the default is the proper likelihood.

    sys_frac adds a fractional model-flux error in quadrature (calibration, model
    mismatch), e.g. 0.03 for 3%.
    '''
    # dusty_likelihood parameter names -> mcmc parameter names
    DUSTY_NAMES = {'logl': 'luminosity', 'temp': 'temperature', 'dust_temp': 'dust_temp',
                   'tau': 'tau_V', 'shell_thickness': 'shell_thickness'}
    # DustyGrid axis names -> mcmc parameter names (s: power-law index of rho ~ r^-s;
    # Y: relative shell thickness R_out/R_in)
    GRID_NAMES = {'temp': 'temperature', 'dust_temp': 'dust_temp', 'tau': 'tau_V', 's': 's', 'Y': 'Y'}
    # value returned for failed/out-of-grid models (dynesty prefers finite values to -inf)
    LOGL_FAIL = -1e300

    def __init__(self, model='grid', model_type='MARCS', ext=None, comp='sil', z=0.00, dm=30.0, dmerr=0.5,
                 mode='nircam', bandpasses=None, space='mag', nsigma_lim=3.0, sys_frac=0.0,
                 reduced_chi2=False):
        self.bounds = {
            'luminosity': [3.0, 6.0],
            'temperature': [2600.0, 5000.0],
            'tau_V': [1e-4, 12.0],
            'dust_temp': [200.0, 1800.0],
            'Av': [0.0, 5.0],
            'Rv': [2.0, 6.0],
            'shell_thickness': [2.0, 1e4],
            # DustyGrid power-law index (rho ~ r^-s) and relative thickness Y = R_out/R_in,
            # over the default grid; with a DustyGrid model all the grid parameters take
            # the grid's own axis ranges
            's': [1.0, 3.5],
            'Y': [10.0, 1e4],
        }
        self.mode = mode.lower()
        if self.mode not in MODE_FILTS:
            raise ValueError(f'Mode {mode} is not valid - use one of {list(MODE_FILTS.keys())}')
        self.space = space.lower()
        if self.space not in ('mag', 'flux'):
            raise ValueError('space must be "mag" or "flux"')
        if reduced_chi2 and self.space != 'mag':
            raise ValueError('reduced_chi2 reproduces mcmc and is only defined in magnitude space')
        self.model_type, self.comp = model_type, comp
        self.dm = dm - 30.0
        self.distance = mu_to_dist(dm, dmerr)
        self.nsigma_lim, self.sys_frac = nsigma_lim, sys_frac
        self.reduced_chi2 = reduced_chi2
        self.verbose = False
        self.phot, self.results = None, None

        if (ext is not None) and not ((isinstance(ext, tuple) | isinstance(ext, list)) & (len(ext) == 2)):
            raise ValueError(f'Extinction {ext} is not valid - needs to be array-like with Rv and Av')
        self.ext = ext
        ext_params = [] if ext is not None else ['Rv', 'Av']

        sgn = '+' if z > -1e-5 else '-'
        self.dirs = {
            'model_grid': Path("..") / 'data' / 'interpolate' / self.mode / f'{model_type}_Z{sgn}{np.abs(z):.2f}_{comp}.pkl',
            'backends': Path("..") / 'data' / 'backends'
        }

        if isinstance(model, str) and model == 'grid':
            self.model_kind = 'grid'
            with open(self.dirs['model_grid'], 'rb') as f:
                self.model = pickle.load(f)
            self.model_fit_params = ['temperature', 'dust_temp', 'tau_V', 'luminosity'] + ext_params
            self.tag = f'{model_type}_{comp}_{self.mode}'
        elif isinstance(model, (str, Path, rsg_phot.dust.DustyGrid)):
            self.model_kind = 'dustygrid'
            if not isinstance(model, rsg_phot.dust.DustyGrid):
                from rsg_phot.dust_twoshell import load_grid
                model = load_grid(model)
            self.model = model
            # axes without an mcmc name (the wind + shell geometry) keep their own
            self.grid_params = [self.GRID_NAMES.get(k, k) for k in self.model.AXES]
            self.model_fit_params = self.grid_params + ['luminosity'] + ext_params
            # the bounds of the grid parameters are the grid's axes (the defaults above
            # are for the other models; e.g. tau_V <= 12 would cut the grid's 0.01-60).
            # Narrow them afterwards through n.bounds if needed.
            for k, par in zip(self.model.AXES, self.grid_params):
                self.bounds[par] = [float(self.model.axes[k].min()), float(self.model.axes[k].max())]
            self.bounds['luminosity'] = list(self.model.LOGL_RANGE)
            self.bandpasses = bandpasses
            self._rp = None
            self.tag = f'dustygrid_{Path(self.model.gridfile).stem}'
        else:
            self.model_kind = 'dusty'
            self.model = model
            self.dusty_params = list(model.free_params)
            self.model_fit_params = [self.DUSTY_NAMES[p] for p in self.dusty_params] + ext_params
            # keep the temperature prior inside the photospheric grid
            if 'temperature' in self.model_fit_params:
                tmin, tmax = model.rsg_temp.value.min(), model.rsg_temp.value.max()
                self.bounds['temperature'] = [max(self.bounds['temperature'][0], tmin),
                                              min(self.bounds['temperature'][1], tmax)]
            self.bandpasses = bandpasses
            self._rp = None
            # model fluxes are computed at the dusty_likelihood distance; shift to dm
            self.dm_model = 5*np.log10(model.dist.to(u.pc).value) - 5
            self.tag = f'dusty_{model.rsg_tag}'
        self.tag += f'_{self.space}'

        # prior type per parameter: 'uniform', 'log' (log-uniform) or 'truncexpon' (mcmc's
        # truncated exponential with scale 1 and upper edge at the bound). Defaults follow
        # mcmc: truncated exponential on tau_V and Av when extinction is fitted.
        self.priors = {p: 'uniform' for p in self.model_fit_params}
        if ext is None:
            self.priors['tau_V'] = 'truncexpon'
            self.priors['Av'] = 'truncexpon'
        if 'shell_thickness' in self.priors:
            self.priors['shell_thickness'] = 'log'
        if self.model_kind == 'dustygrid':
            # the grid spans 0.01-60 in tau and 10-1e4 in Y, so sample both in log
            self.priors['tau_V'] = 'log'
            self.priors['Y'] = 'log'

    def prior_transform(self, u_):
        # map the unit cube onto the parameters
        theta = np.empty(len(u_))
        for i, par in enumerate(self.model_fit_params):
            lo, hi = self.bounds[par]
            kind = self.priors[par]
            if kind == 'uniform':
                theta[i] = lo + u_[i] * (hi - lo)
            elif kind == 'log':
                theta[i] = 10**(np.log10(lo) + u_[i] * (np.log10(hi) - np.log10(lo)))
            elif kind == 'truncexpon':
                dist = truncexpon(hi)
                c0, c1 = dist.cdf(lo), dist.cdf(hi)
                theta[i] = dist.ppf(c0 + u_[i] * (c1 - c0))
            else:
                raise ValueError(f'Unknown prior type {kind} for {par}')
        return theta

    def get_bandpasses(self, filts):
        # JWST bandpasses from the filter names; anything else must be passed in
        from rsg_phot.rsg import rsg_phot
        if self._rp is None:
            self._rp = rsg_phot(verbose=False)
        bps = dict(self.bandpasses or {})
        for f in filts:
            if f in bps:
                continue
            # MIRI filters also by their short names (F560W for F0560W), as in the DUSTY grids
            miri = f'F{int(f[1:-1]):04d}W' if re.fullmatch(r'F\d{3,4}W', f) else f
            if f in NRC_FILTS:
                bps[f] = self._rp.get_jwst_filters(f'nircam,{f}')
            elif miri in MIRI_FILTS:
                bps[f] = self._rp.get_jwst_filters(f'miri,{miri}')
            else:
                raise ValueError(f'No bandpass for {f}: pass it in bandpasses (e.g. HST filters from stsynphot.band)')
        return {f: bps[f] for f in filts}

    def model_mag(self, theta):
        # model AB magnitudes in phot['inst_filt'] at the fitted distance
        p = dict(zip(self.model_fit_params, theta))
        rv, av = self.ext if self.ext is not None else (p['Rv'], p['Av'])
        if self.model_kind == 'grid':
            params = np.meshgrid(*[p[k] for k in ['temperature', 'dust_temp', 'tau_V', 'luminosity']], rv, av,
                                 indexing='ij', sparse=True)
            return np.array([self.model[f](params).flatten()[0] for f in self.phot['inst_filt']]) + self.dm
        if self.model_kind == 'dustygrid':
            flux = self.model.phot(p['luminosity'], *[p[k] for k in self.grid_params],
                                   dist=self.distance[0] * u.Mpc)[self._gidx]
            if av > 0:
                flux = flux * self._rp.extinction_law(self._pivots, av, rv)
            return flux_to_mag(flux)

        sed = self.model.model_sed([p[self.DUSTY_NAMES[k]] for k in self.dusty_params])
        if av > 0:
            sed['flam'] = sed['flam'] * self._rp.extinction_law(sed['lambda'].to(u.Angstrom).value, av, rv)
        flux = self.model.model_phot(None, self._bps, sed=sed)
        return flux_to_mag(np.array([flux[f] for f in self.phot['inst_filt']])) + (self.dm + 30.0 - self.dm_model)

    def model_flux(self, theta):
        # model F_nu (uJy) in phot['inst_filt']
        return mag_to_flux(self.model_mag(theta))

    def log_likelihood(self, theta):
        try:
            model_mag = self.model_mag(theta)
        except (rsg_phot.dust.DustyError, ValueError):
            return self.LOGL_FAIL
        if not np.all(np.isfinite(model_mag)):
            return self.LOGL_FAIL

        if self.space == 'mag':
            chi2 = np.sum((self.phot['mag'] - model_mag)**2 / self.phot['magerr']**2)
            if self.reduced_chi2:
                return -0.5 * chi2 / (len(model_mag) - 1)
            return -0.5 * chi2 - np.sum(np.log(np.sqrt(2*np.pi) * self.phot['magerr']))

        model = mag_to_flux(model_mag)
        sig = np.sqrt(self.phot['fluxerr']**2 + (self.sys_frac * model)**2)
        det, lim = ~self.phot['uplim'], self.phot['uplim']
        logl = (-0.5 * np.sum(((self.phot['flux'][det] - model[det]) / sig[det])**2)
                - np.sum(np.log(np.sqrt(2*np.pi) * sig[det])))
        if lim.any():
            # probability that a source with the model flux falls below the limit
            logl += np.sum(log_ndtr((self.phot['flux'][lim] - model[lim]) / sig[lim]))
        return logl

    def setup_phot(self, phot, set_limmask=False):
        '''
        Validate the photometry and, for space='flux', put it in uJy: phot['flux'],
        phot['fluxerr'] and phot['uplim'] are added if the input was in magnitudes.
        set_limmask drops missing measurements (mag or magerr > 90 or NaN), as in
        mcmc; flagged upper limits are kept.
        '''
        phot = dict(phot)
        phot['inst_filt'] = np.asarray(phot['inst_filt'])
        n = len(phot['inst_filt'])
        uplim = np.zeros(n, bool) if phot.get('uplim') is None else np.asarray(phot['uplim'], bool)
        if set_limmask and 'mag' in phot:
            mag, magerr = np.asarray(phot['mag'], float), np.asarray(phot['magerr'], float)
            bad = (mag > 90.0) | np.isnan(mag) | (~uplim & ((magerr > 90.0) | np.isnan(magerr)))
            for k in [k for k in ['mag', 'magerr', 'flux', 'fluxerr', 'inst_filt'] if k in phot]:
                phot[k] = np.asarray(phot[k])[~bad]
            uplim = uplim[~bad]
        phot['uplim'] = uplim

        if self.space == 'mag':
            if uplim.any():
                raise ValueError('upper limits need space="flux"')
            phot['mag'], phot['magerr'] = np.asarray(phot['mag'], float), np.asarray(phot['magerr'], float)
        else:
            if 'flux' in phot:
                flux = np.asarray(phot['flux'], float)
                fluxerr = np.asarray(phot.get('fluxerr', np.full(len(flux), np.nan)), float)
            else:
                flux, fluxerr = mag_to_flux(phot['mag'], phot['magerr'])
                fluxerr = np.where(uplim, np.nan, fluxerr)
            # limits without an explicit sigma: F_lim is nsigma_lim sigma
            fluxerr = np.where(uplim & ~np.isfinite(fluxerr), flux / self.nsigma_lim, fluxerr)
            if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(fluxerr)) or np.any(fluxerr <= 0):
                raise ValueError('flux and fluxerr must be finite with fluxerr > 0 (use set_limmask to drop missing data)')
            phot['flux'], phot['fluxerr'] = flux, fluxerr

        if self.model_kind == 'grid':
            missing = [f for f in phot['inst_filt'] if f not in self.model]
            if missing:
                raise ValueError(f'Filters {missing} are not in the {self.mode} model grid '
                                 f'{self.dirs["model_grid"]}; use mode="miri" to fit MIRI photometry')
        elif self.model_kind == 'dustygrid':
            missing = [f for f in phot['inst_filt'] if f not in self.model.bands]
            if missing:
                raise ValueError(f'Filters {missing} are not in the DUSTY grid {self.model.gridfile} '
                                 f'(bands {self.model.bands})')
            self._gidx = np.array([self.model.bands.index(f) for f in phot['inst_filt']])
            # pivot wavelengths (Angstrom) for the per-band extinction
            if self.ext is None or self.ext[1] > 0:
                bps = self.get_bandpasses(list(phot['inst_filt']))
                self._pivots = np.array([bps[f].pivot().to(u.Angstrom).value for f in phot['inst_filt']])
        else:
            self._bps = self.get_bandpasses(list(phot['inst_filt']))
        self.phot = phot
        return phot

    def run_nested(self, phot, nlive=500, dlogz=0.1, bound='multi', sample='rwalk', npool=None,
                   maxcall=None, seed=None, set_limmask=False, save=True, return_params=True):
        import dynesty
        import dynesty.pool

        self.setup_phot(phot, set_limmask=set_limmask)
        ndim = len(self.model_fit_params)
        kw = dict(nlive=nlive, bound=bound, sample=sample, rstate=np.random.default_rng(seed))
        run_kw = dict(dlogz=dlogz, maxcall=maxcall, print_progress=self.verbose)

        if npool and npool > 1:
            # the likelihood (and any dusty_likelihood) is sent to each worker once.
            # On macOS, scripts that use a pool need an if __name__ == '__main__' guard.
            with dynesty.pool.Pool(npool, self.log_likelihood, self.prior_transform) as pool:
                sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool, **kw)
                sampler.run_nested(**run_kw)
        else:
            sampler = dynesty.NestedSampler(self.log_likelihood, self.prior_transform, ndim, **kw)
            sampler.run_nested(**run_kw)
        self.results = sampler.results

        if save:
            os.makedirs(self.dirs['backends'], exist_ok=True)
            with open(self.results_file(self.phot), 'wb') as f:
                pickle.dump({'results': self.results, 'params': self.model_fit_params, 'bounds': self.bounds,
                             'priors': self.priors, 'ext': self.ext, 'space': self.space,
                             'nsigma_lim': self.nsigma_lim, 'sys_frac': self.sys_frac,
                             'reduced_chi2': self.reduced_chi2, 'tag': self.tag, 'phot': self.phot}, f)
        if return_params:
            return self.read_params()

    def results_file(self, phot):
        return self.dirs['backends'] / f'{phot["index"]}_{self.tag}_nest.pkl'

    def posterior_samples(self, results=None):
        # equally weighted posterior samples
        results = self.results if results is None else results
        return results.samples_equal()

    def read_params(self, results=None):
        '''
        Posterior median and 16th/84th-percentile offsets for each parameter, as
        {param: (median, lower, upper)} like mcmc.read_params, plus the evidence in
        self.evidence = (logZ, logZerr). The distance uncertainty is added in quadrature
        to the luminosity errors.
        '''
        results = self.results if results is None else results
        samples = self.posterior_samples(results)
        params = {}
        for i, par in enumerate(self.model_fit_params):
            p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
            lo, hi = p50 - p16, p84 - p50
            if par == 'luminosity':
                logL_unc = 2.17 * self.distance[1] / self.distance[0]
                lo, hi = np.hypot(lo, logL_unc), np.hypot(hi, logL_unc)
            params[par] = (p50, lo, hi)
            if self.verbose:
                print(f'{par:<18}: {p50:>12.4g} + {hi:>10.3g} - {lo:>10.3g}')
        self.evidence = (results.logz[-1], results.logzerr[-1])
        if self.verbose:
            print(f'log Z = {self.evidence[0]:.3f} +/- {self.evidence[1]:.3f}')
        return params

def sn2026sqf_phot(path, exclude=('F187N', 'F405N')):
    '''
    Photometry dict for nest (space='flux') from an SN 2026sqf photometry table
    (data/dolphot/sn2026sqf/sn2026sqf_photometry_*_ujy.txt): detections with err_ujy,
    and the non-detections as upper limits at ul_ujy (sigma = ul / nsigma_lim, set by
    nest). exclude drops bands, by default the Pa alpha and Br alpha narrow bands.
    '''
    from astropy.io import ascii
    tb = ascii.read(path, format='csv', comment=r'\s*#', header_start=0, data_start=1)
    filt = np.array([str(f) for f in tb['filter']])
    det = np.asarray(tb['detected']).astype(str) == 'True'
    flux = np.where(det, np.ma.filled(np.ma.asarray(tb['flux_ujy'], float), np.nan),
                    np.ma.filled(np.ma.asarray(tb['ul_ujy'], float), np.nan))
    err = np.where(det, np.ma.filled(np.ma.asarray(tb['err_ujy'], float), np.nan), np.nan)
    keep = ~np.isin(filt, list(exclude)) & np.isfinite(flux)
    return {'index': 'sn2026sqf', 'inst_filt': filt[keep], 'flux': flux[keep], 'fluxerr': err[keep], 'uplim': ~det[keep]}

def chi2_radmc(phot, model, floor=0.05, nsigma_lim=3.0):
    '''
    chi^2 as used to score the RADMC-3D models (2026sqf_tmp/evol_fit.chi2): detections
    with sigma = err (+) floor * observed flux, and limits one-sided, penalised only
    above the limit with sigma = limit / nsigma_lim. For comparing the best fit here
    with those models; the nested sampling likelihood itself is nest.log_likelihood.
    '''
    det, lim = ~phot['uplim'], phot['uplim']
    sig = np.hypot(phot['fluxerr'][det], floor * phot['flux'][det])
    over = np.clip(model[lim] - phot['flux'][lim], 0, None) / (phot['flux'][lim] / nsigma_lim)
    return np.sum(((phot['flux'][det] - model[det]) / sig)**2) + np.sum(over**2)

if __name__ == '__main__':
    # nested sampling of an SN 2026sqf SED with a DUSTY grid from dust.py (e.g. the single-shell
    # grid), interpolated on the fly by DustyGrid (tensor-product PCHIP). Run from the repo root:
    #   python -m rsg_phot.mcmc_nest --npool 8
    ap = argparse.ArgumentParser(description='Nested sampling of SN 2026sqf with a DUSTY grid')
    ap.add_argument('--grid', default='data/dusty_grids/2026sqf/shell_amC_SiC20.npz', help='DustyGrid .npz')
    ap.add_argument('--phot', default='data/dolphot/sn2026sqf/sn2026sqf_photometry_new_ujy.txt', help='photometry table')
    ap.add_argument('--exclude', nargs='*', default=['F187N', 'F405N'], help='bands left out of the fit')
    ap.add_argument('--dist', type=float, nargs=2, default=[19.0, 1.5], metavar=('D', 'ERR'), help='distance and error (Mpc)')
    ap.add_argument('--ext', type=float, nargs=2, default=[3.1, 0.125], metavar=('RV', 'AV'),
                    help='fixed extinction, MW + host (applied per band at the bandpass pivots)')
    ap.add_argument('--sys_frac', type=float, default=0.05, help='fractional model error added in quadrature')
    ap.add_argument('--nlive', type=int, default=500)
    ap.add_argument('--dlogz', type=float, default=0.1)
    ap.add_argument('--npool', type=int, default=1, help='processes for the likelihood')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--backends', default='data/backends', help='directory for the results pickle')
    args = ap.parse_args()

    d, de = args.dist
    dm, dmerr = 5*np.log10(d*1e6) - 5, 5/np.log(10) * de/d
    n = nest(model=args.grid, ext=tuple(args.ext), dm=dm, dmerr=dmerr, space='flux', sys_frac=args.sys_frac)
    n.verbose = True
    n.dirs['backends'] = Path(args.backends)
    # T_in nodes are geometric (150-1500 K), so sample it in log like tau_V and Y
    n.priors['dust_temp'] = 'log'
    phot = sn2026sqf_phot(args.phot, exclude=args.exclude)
    print(f'grid {args.grid}: {n.model_fit_params}')
    print(f'bands {", ".join(phot["inst_filt"])}; limits {", ".join(phot["inst_filt"][phot["uplim"]])}')
    print('bounds ' + ', '.join(f'{p} {n.bounds[p]} ({n.priors[p]})' for p in n.model_fit_params), flush=True)

    t0 = time.time()
    params = n.run_nested(phot, nlive=args.nlive, dlogz=args.dlogz, npool=args.npool, seed=args.seed)
    res = n.results
    best = res.samples[np.argmax(res.logl)]
    model = n.model_flux(best)
    print(f'\n{res.ncall.sum() if hasattr(res.ncall, "sum") else res.ncall} likelihood calls, {(time.time() - t0)/60:.1f} min')
    print('maximum likelihood: ' + ', '.join(f'{p} {v:.4g}' for p, v in zip(n.model_fit_params, best)))
    print(f'  log L_max {res.logl.max():.2f}; RADMC-style chi2 {chi2_radmc(n.phot, model):.1f} '
          f'({(~n.phot["uplim"]).sum()} detections + {n.phot["uplim"].sum()} limits)')
    for f, o, e, m, ul in zip(n.phot['inst_filt'], n.phot['flux'], n.phot['fluxerr'], model, n.phot['uplim']):
        print(f'  {f:>7} obs {o:8.3f} {"(limit)" if ul else f"+/- {e:6.3f}":>10}  model {m:8.3f}')
    print(f'results: {n.results_file(n.phot)}')
