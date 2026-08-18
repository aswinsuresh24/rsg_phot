import numpy as np
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob
import emcee
import rsg_phot.dust
import progressbar
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import math
import pickle
import random
from astropy.stats import sigma_clipped_stats as scs
import time
import re
from scipy.stats import truncexpon
from pathlib import Path

NRC_FILTS = np.array(['F070W','F090W','F115W','F140M','F150W','F150W2','F162M',
                      'F164N','F182M','F187N','F200W','F210M','F212N','F250M',
                      'F277W','F300M','F322W2','F323N','F335M','F356W','F360M',
                      'F405N','F410M','F430M','F444W','F460M','F466N','F470N','F480M'])
MIRI_FILTS = np.array(['F0560W','F0770W','F1000W','F1130W','F1280W','F1500W',
                       'F1800W','F2100W','F2550W'])
# 'miri' fits NIRCam and MIRI photometry together, so it carries both filter sets
MODE_FILTS = {'nircam': NRC_FILTS,
              'miri': np.concatenate([NRC_FILTS, MIRI_FILTS])}

def filt_wavelength(filts):
    '''
    Pivot wavelength in microns from a JWST filter name.  The digits following
    the leading F are hundredths of a micron for both NIRCam (F070W -> 0.70,
    F150W2 -> 1.50) and MIRI (F0560W -> 5.60, F2550W -> 25.50).
    '''
    single = isinstance(filts, str)
    if single: filts = [filts]
    wv = np.array([float(re.match(r'F(\d+)', f.upper()).group(1))/100 for f in filts])
    return wv[0] if single else wv

DUST_BB_MASS = 2.4319771e-12
RSG_V_WIND = 50.0 * u.km/u.s

d=const.R_sun.to('cm') * 1.0 * u.km/u.s / (const.M_sun.to('g') / (1.0 * u.year) * 1.0 * u.cm**2/u.g)
DUST_BB_WIND = d.to(u.Unit(1)).value

class mcmc(object):
    def __init__(self, model_type='MARCS', ext=None, comp='sil', z=0.00, shell=2, dm=30.0, dmerr=0.5,
                 mode='nircam'):
        self.bounds = {
            'luminosity': [3.0, 6.0],
            'temperature': [2600.0, 5000.0], 
            'tau_V': [1e-4, 12.0],
            'dust_temp': [200.0, 1800.0],
            'Av': [0.0, 5.0],
            'Rv': [2.0, 6.0]
        }

        self.model_type = model_type
        self.ext = ext
        self.mode = mode.lower()
        if self.mode not in MODE_FILTS:
            raise ValueError(f'Mode {mode} is not valid - use one of {list(MODE_FILTS.keys())}')
        self.filts = MODE_FILTS[self.mode]
        self.wv = filt_wavelength(self.filts)

        if self.model_type=='MARCS15':
            if z!=0.0:
                raise ValueError('15Msun MARCS model is only avaiable at Z=0.0')
            self.bounds['temperature'] = [3300.0, 4500.0]

        if (self.model_type=='NewEra') & (z==-0.25):
            raise ValueError('NewEra grid not available at Z = -0.25 Zsun')

        if self.ext is None:
            self.model_fit_params = ['temperature', 'dust_temp', 'tau_V', 'luminosity', 'Rv', 'Av']
            self.blobs_dtype = None
            self.log_likelihood_fn = self.log_likelihood
        elif (isinstance(self.ext, tuple) | isinstance(self.ext, list)) & (len(self.ext)==2):
            self.model_fit_params = ['temperature', 'dust_temp', 'tau_V', 'luminosity']
            self.blobs_dtype = None
            self.log_likelihood_fn = self.log_likelihood_tau
        else:
            raise ValueError(f'Extinction {self.ext} is not valid - needs to be array-like with Rv and Av')

        self.backend = None
        self.dm = dm - 30.0
        self.verbose = False
        self.comp = comp
        sgn = '+' if z > -1e-5 else '-'
        self.dirs = {
            'bandpass':Path("..") / 'data' / 'bandpass',
            'model_grid':Path("..") / 'data' / 'interpolate' / self.mode / f'{self.model_type}_Z{sgn}{np.abs(z):.2f}_{self.comp}.pkl',
            'backends':Path("..") / 'data' / 'backends'
        }
        with open(self.dirs['model_grid'], 'rb') as f:
            self.model = pickle.load(f)

        self.significant_figures = 3
        self.distance = self.mu_to_dist(dm, dmerr)
        self.phot = None

    def reset_bounds(self):
        self.bounds = {
            'luminosity': [3.0, 6.0],
            'temperature': [2600.0, 5000.0], 
            'tau_V': [0.01, 12.0],
            'dust_temp': [200.0, 1800.0],
            'Av': [0.0, 5.0],
            'Rv': [2.0, 6.0]
        }

    def mu_to_dist(self, dm, dmerr):
        d = 10**(dm/5+1.0) * u.pc
        de = (10**((dm+dmerr)/5+1.0) - 10**((dm-dmerr)/5+1.0)) * u.pc
        return [d.to(u.Mpc).value, de.to(u.Mpc).value]

    def get_guess(self, guess_type='params'):
        if self.backend:
            try:
                if self.backend.iteration>0:
                    flat_samples = np.array(self.backend.get_chain(flat=True))
                    flat_prob = np.array(self.backend.get_log_prob(flat=True))
                    flat_blobs = np.array(self.backend.get_blobs(flat=True))

                    mask = np.isinf(np.abs(flat_prob))
                    flat_samples = flat_samples[~mask]
                    flat_prob = -1.0 * flat_prob[~mask]
                    flat_blobs = flat_blobs[~mask]

                    if len(flat_prob)>0:
                        best = np.argmin(flat_prob)
                        if guess_type=='params': return(flat_samples[best])
                        if guess_type=='blobs': return(flat_blobs[best])
            except (OSError, KeyError):
                print(traceback.format_exc())

        else:
            guess = np.array([3200, 800, 0.02, 1.6e4, 3.1, 0.125])

        return(guess)
    
    def load_backend(self, phot):
        objname = phot['index']
        name = ''+objname
        for i,m,e in zip(phot['inst_filt'],phot['mag'],phot['magerr']):
            name += i+'='+str('%7.4f'%m)+'+/-'+str('%7.4f'%e)
        name = name+self.comp
        name = name.replace(' ','')

        newname = ''
        for c in name:
            newname += str(ord(c))
        newname = str(int(newname)%100207100213100237100267)

        # keep the nircam filenames as they were so existing backends stay usable
        mode_suffix = '' if self.mode=='nircam' else '_'+self.mode
        backfile = self.dirs['backends'] / str(objname+'_'+self.model_type+mode_suffix+'.h5')
        if self.verbose:
            print('Backend file:',backfile)
            print('Backend name:',newname)
        backend = emcee.backends.HDFBackend(backfile, name=newname)

        return backend

    def get_init_pos(self, nwalkers, seed=None):
        ndim = len(self.model_fit_params)
        init_pos = np.zeros((nwalkers, ndim))

        if seed is None:
            for i,par in enumerate(self.model_fit_params):
                init_pos[:,i] = np.random.uniform(self.bounds[par][0], self.bounds[par][1], nwalkers)
        else:
            rng = np.random.default_rng(seed)
            for i,par in enumerate(self.model_fit_params):
                init_pos[:,i] = rng.uniform(self.bounds[par][0], self.bounds[par][1], nwalkers)

        return init_pos
    
    def check_bounds(self, theta):
        for i,par in enumerate(self.model_fit_params):
            if theta[i] < self.bounds[par][0] or theta[i] > self.bounds[par][1]:
                return(True)
        return(False)
    
    def run_emcee(self, phot, ext=None, nsteps=350, nwalkers=64, burn_in=75, set_limmask=False, return_params=True, calculate_chi=False):
        if set_limmask:
            limmask = (phot['mag'] > 90.0) | (phot['magerr'] > 90.0) | (np.isnan(phot['mag'])) | (np.isnan(phot['magerr']))
            phot['mag'] = phot['mag'][~limmask]
            phot['magerr'] = phot['magerr'][~limmask]
            phot['inst_filt'] = phot['inst_filt'][~limmask]

        missing = [f for f in phot['inst_filt'] if f not in self.model]
        if missing:
            raise ValueError(f'Filters {missing} are not in the {self.mode} model grid '
                             f'{self.dirs["model_grid"]}; use mode="miri" to fit MIRI photometry')
        self.phot = phot

        if ext is not None:
            self.ext = ext
            if (isinstance(self.ext, tuple) | isinstance(self.ext, list)) & (len(self.ext)==2):
                self.model_fit_params = ['temperature', 'dust_temp', 'tau_V', 'luminosity']
                self.blobs_dtype = None
                self.log_likelihood_fn = self.log_likelihood_tau
            else:
                print(f'WARNING: invalid format for extinction {ext}; Extinction will be fit for in MCMC.')

        ndim = len(self.model_fit_params)

        #load backend
        backend = self.load_backend(self.phot)
        use_backend_pos = False
        try:
            if os.path.exists(backend.filename):
                try:
                    if backend.iteration>0:
                        use_backend_pos = True
                except KeyError:
                    pass
        except:
            print(traceback.format_exc())

        if use_backend_pos:
            walkers_backend = backend.shape[0]
            if nwalkers!=walkers_backend:
                print('Inconsistent walkers, cannot use backend')
                use_backend_pos = False

        if use_backend_pos:
            if self.verbose: print('Current number of iterations on backend: ',backend.iteration)
            init_pos = backend.get_last_sample()
        else:
            init_pos = self.get_init_pos(nwalkers)

        # Construct emcee sampler with parameters derived above
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_likelihood_fn, 
                                        backend=backend, moves=[(emcee.moves.KDEMove(), 1.0)], 
                                        blobs_dtype=self.blobs_dtype)

        # Run MCMC step
        sampler.run_mcmc(init_pos, nsteps, progress=self.verbose)

        if return_params:
            params = self.read_params(self.phot, burn_in=burn_in, calculate_chi=calculate_chi)
            return params
    
    def read_params(self, phot, burn_in=75, thin=1, calculate_chi=False):
        ndim = len(self.model_fit_params)
        # Read current model probabilities, samples, and blobs from backend
        reader = self.load_backend(phot)
        sample = np.array(reader.get_chain(flat=True))
        prob = np.array(reader.get_log_prob(flat=True))
        blob = np.array(reader.get_blobs(flat=True))

        if self.verbose:
            print('Current number of samples in backend:', len(sample))
            mask = ~np.isinf(prob)
            print('Minimum chi^2 is:','%.7f'%(-1.0*np.max(prob)))
            print('\n\n')

        params = dict.fromkeys(self.model_fit_params)
        converged_sample = np.array(reader.get_chain(flat=True, discard=burn_in, thin=thin))
        converged_prob = np.array(reader.get_log_prob(flat=True, discard=burn_in, thin=thin))
        converged_blob = np.array(reader.get_blobs(flat=True, discard=burn_in, thin=thin))

        for i,param in enumerate(self.model_fit_params):
            p, p_elow, p_eup = self.calculate_param_best_fit(converged_sample[:,i], converged_prob, ndim, param, 
                                                             verbose=self.verbose, return_uncertainty=True)
            params[param] = (p, p_elow, p_eup)

        if calculate_chi:
            fit_pe = np.array(list(params.values()))
            min_chi_params = converged_sample[np.argmax(converged_prob)]

            posterior_params = np.meshgrid(*fit_pe[:, 0], indexing='ij', sparse=True)
            model_mag = np.array([self.model[f](posterior_params).flatten()[0] for f in phot['inst_filt']])+self.dm
            chi_posterior = np.sum((phot['mag'] - model_mag)**2)/(len(model_mag)-1)

            mc_params = np.meshgrid(*min_chi_params, indexing='ij', sparse=True)
            chi_mag = np.array([self.model[f](mc_params).flatten()[0] for f in phot['inst_filt']])+self.dm
            chi_best = 0.5*np.sum((phot['mag'] - chi_mag)**2)/(len(chi_mag)-1)
            return (params, min_chi_params, chi_posterior, chi_best)
        return params
    
    def log_prior(self, theta):
        tau, av = theta[2], theta[5]
        tau_pdf = truncexpon(self.bounds['tau_V'][-1]).logpdf(tau)
        av_pdf = truncexpon(self.bounds['Av'][-1]).logpdf(av)
        return tau_pdf + av_pdf

    def log_likelihood(self, theta):
        if self.check_bounds(theta):
            return(-np.inf)
        
        params = np.meshgrid(*theta, indexing='ij', sparse=True)
        model_mag = np.array([self.model[f](params).flatten()[0] for f in self.phot['inst_filt']])+self.dm

        if any(np.isnan(model_mag)):
            return(-np.inf)
        
        chi2 = -0.5*np.sum((self.phot['mag']-model_mag)**2/self.phot['magerr']**2) / (len(model_mag) - 1)
        if np.isnan(chi2):
            print(f'likelihood is nan for {theta}')
            return(-np.inf)
        
        logp = chi2 + self.log_prior(theta)

        return logp
    
    def log_likelihood_tau(self, theta):
        if self.check_bounds(theta):
            return(-np.inf)

        params = np.meshgrid(*theta, self.ext[0], self.ext[1], indexing='ij', sparse=True)
        model_mag = np.array([self.model[f](params).flatten()[0] for f in self.phot['inst_filt']])+self.dm

        if any(np.isnan(model_mag)):
            return(-np.inf)

        chi2 = -0.5*np.sum((self.phot['mag']-model_mag)**2/self.phot['magerr']**2) / (len(model_mag) - 1)
        if np.isnan(chi2):
            print(f'likelihood is nan for {theta}')
            return(-np.inf)

        return chi2
    
    def sample_params(self, params, prob, ndim, nsamples=None, downsample=1.0):

        mask = np.isinf(np.abs(prob)) | np.isnan(prob)
        _, prob_median, prob_sig = scs(prob)
        mask = mask & (prob > prob_median - prob_sig)
        if all(mask):
            print('WARNING: all probabilities are bad.  Try wider param range')
            return(params[0])
        if len(params.shape)==1:
            params = params[~mask]
        else:
            params = params[~mask,:]

        prob = -1.0 * prob[~mask]
        prob = prob / np.min(prob)

        chi_limit = [1.00, 2.30, 3.50, 4.72, 5.89, 7.04]
        mask = prob < 1.0 + downsample * chi_limit[ndim-1]
        if len(params.shape)==1:
            params_sample = params[mask]
        else:
            params_sample = params[mask,:]
        prob_sample = prob[mask]

        if nsamples and nsamples < len(prob_sample):
            rand = np.array(random.sample(range(0, len(prob_sample)), nsamples))
            if len(params_sample.shape)==1:
                params_sample = params_sample[rand]
            else:
                params_sample = params_sample[rand,:]
            prob_sample = prob_sample[rand]

        return(params_sample, prob_sample)
    
    def calculate_param_best_fit(self, params, prob, ndim, name, verbose=True,
                                 sampled=False, return_uncertainty=False):

        # Parameters might have already been sampled
        if not sampled:
            params_sample, prob_sample = self.sample_params(params, prob, ndim)
        else:
            _, prob_median, prob_sig = scs(prob)
            mask = (prob > prob_median - prob_sig)
            params_sample = params[mask]
            prob_sample = prob[mask]

        n = int(self.significant_figures)
        out_fmt = '{0:<18}: {1:>12} + {2:>12} - {3:>12}'

        mask = ~np.isnan(params_sample)
        params_sample = params_sample[mask]

        best = np.percentile(params_sample, 50)
        minval = np.percentile(params_sample, 16)
        maxval = np.percentile(params_sample, 84)

        mcmc = np.round(best, n)
        log_mcmc = np.log10(mcmc)
        if np.isnan(log_mcmc):
            log_mcmc = 0.0
        digits = int(np.ceil(log_mcmc))
        decimal_place = -1 * (digits - n)
        if float(mcmc)==int(mcmc) and decimal_place < 1:
            mcmc=int(mcmc)

        minval = round(minval, decimal_place)
        maxval = round(maxval, decimal_place)
        maxval = maxval-mcmc
        minval = mcmc-minval

        if float(minval)==int(minval) and decimal_place < 1:
            minval=int(minval)
        if float(maxval)==int(maxval) and decimal_place < 1:
            maxval=int(maxval)

        if name=='luminosity':
            logL_unc = 2.17 * self.distance[1]/self.distance[0]
            minval = minval + logL_unc
            maxval = maxval + logL_unc

        if np.log10(mcmc)<-3:
            str_fmt = '%.3e'
            mcmc = str_fmt % mcmc
            maxval = str_fmt % maxval
            minval = str_fmt % minval
        elif decimal_place>0:
            str_fmt = '%7.{0}f'.format(int(decimal_place))
            mcmc = str_fmt % mcmc
            maxval = str_fmt % maxval
            minval = str_fmt % minval

        if verbose: print(out_fmt.format(name, mcmc, maxval, minval))

        if return_uncertainty:
            return(best, float(minval), float(maxval))
        else:
            return(best)