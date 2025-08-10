import numpy as np
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob
import emcee
import dust
import progressbar
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import math

DUST_BB_MASS = 2.4319771e-12
RSG_V_WIND = 50.0 * u.km/u.s

d=const.R_sun.to('cm') * 1.0 * u.km/u.s / (const.M_sun.to('g') / (1.0 * u.year) * 1.0 * u.cm**2/u.g)
DUST_BB_WIND = d.to(u.Unit(1)).value

class mcmc(object):
    def __init__(self, model_type='rsg', dm=30):
        self.bounds = {
            'luminosity': [3.5, 7.0],
            'temperature': [2000.0, 5500.0], #bounds to be set by marcs models
            'tau_V': [0.01, 1.0],
            'dust_temp': [500.0, 1800.0],
            'Av': [0.0, 6.0],
            'Rv': [2.0, 6.0]
        }

        self.model_type = model_type
        self.model_fit_params = {'rsg': ['tau_V', 'luminosity','temperature','dust_temp', 'Av', 'Rv']}
        self.backend = None
        self.dm = dm

    def get_guess(self, model_type='rsg', guess_type='params'):
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

        if model_type == 'rsg':
            guess = np.array([0.02, 4.2, 3200, 1000, 2.3, 4.4])
        else:
            print('ERROR: unrecognized model type. Only "rsg" is currently supported.')
            sys.exit()

        return(guess)
    
    def load_backend(self, model_type, phottable, notau=False): #needs reformating
        # Format objname from input filename
        objname = ''
        if 'name' in phottable.meta.keys():
            objname = phottable.meta['name']
        else:
            objname = self.filename
            if '/' in objname: objname = os.path.split(objname)[1]
            objname = objname.replace('.txt','')
            objname = objname.replace('.dat','')
            objname = objname.replace('.cat','')
            objname = objname.split('_')[0]
            objname = objname.split('-')[0]

        # Create a name from inst_filt, mag, magerr for comparison to backend
        mjd, inst_filt, mag, magerr = self.get_fit_parameters(phottable)
        name = ''
        for i,m,e in zip(inst_filt,mag,magerr):
            name += i+'='+str('%7.4f'%m)+'+/-'+str('%7.4f'%e)
            # Remove spaces
            name = name.replace(' ','')

        newname = ''
        for c in name:
            if c in '1234567890': newname+=c
        newname = str(int(newname)%100207100213100237100267)

        if self.model_type=='rsg' and notau:
            backfile = self.dirs['backends']+objname+'_rsg_notau.h5'
        else:
            backfile = self.dirs['backends']+objname+'_'+self.model_type+'.h5'
        if self.verbose:
            print('Backend file:',backfile)
            print('Backend name:',newname)
        backend = emcee.backends.HDFBackend(backfile, name=newname)

        return backend

    def get_init_pos(self, guess, ndim, nwalkers, sigma=1):
        init_pos = [guess * np.random.lognormal(1.0, sigma, ndim) for i in range(nwalkers)]
        return init_pos
    
    def run_emcee(self, phot, sigma=1.0, nsteps=5000, nwalkers=100, guess_type='params'):
        # mag, magerr, inst_filt = phot['mag'], phot['magerr'], phot['inst_filt']
        
        guess = self.get_guess(model_type=self.model_type, guess_type=guess_type)
        ndim = len(self.model_fit_params[self.model_type])

        #load backend
        backend = self.load_backend()
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
            print('Current number of iterations on backend: ',backend.iteration)
            init_pos = backend.get_last_sample()
        else:
            init_pos = self.get_init_pos(guess, ndim, nwalkers, sigma=sigma)

        # Construct emcee sampler with parameters derived above
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_likelihood, 
                                        args=(phot['inst_filt'], phot['mag'], phot['magerr']), backend=backend)

        # Run MCMC step - slow
        sampler.run_mcmc(init_pos, nsteps, progress=True)

        # Read current model probabilities, samples, and blobs from backend
        reader = self.load_backend(self.model_type, self.phottable)
        sample = np.array(reader.get_chain(flat=True))
        prob = np.array(reader.get_log_prob(flat=True))
        blob = np.array(reader.get_blobs(flat=True))

        params = [] ; blobs = []
        for i,param in enumerate(self.model_fit_params):
            p=self.calculate_param_best_fit(sample[:,i], prob, ndim, param)
            params.append(p)

        for i,param in enumerate(self.model_fit_blobs):
            b=self.calculate_param_best_fit(blob[:,i], prob, ndim, param)
            blobs.append(b)

    def compute_model_mag(self, inst_filt, theta, extinction=None):

        model_mag = self.model_functions[self.model_type](inst_filt, *theta)

        # Catch bad model_mag value
        if model_mag is None:
            return(None, None, None)
        elif all([math.isnan(val) for val in model_mag]): #np.isnan?
            return(None, None, None)

        # Apply dm and extinction according to probability distribution
        model_mag = np.array(model_mag) + self.dm
        if extinction is not None:
            Av, Rv = extinction
        else:
            Av, Rv = self.inject_uniform_into_cdf(self.extinction['Av'],
                self.extinction['Rv'], self.extinction['cdf'])

        for i,val in enumerate(inst_filt):
            if self.extinction_model:
                model_mag[i] += self.extinction['function'][val](Rv, Av)
            elif self.host_ext:
                model_mag[i] += self.host_ext_inst_filt[val]

            model_mag[i] += self.rv[i] * self.mw_ebv

        # Output model magnitudes and extinction values as blobs
        return(model_mag, Av, Rv)

    # Estimate log likelihood for a given age, mass, and data set
    def log_likelihood(self, theta, inst_filt, mag, magerr, extinction=None):

        if self.check_bounds(theta):
            return(-np.inf, None, None)

        if not extinction and self.host_ext:
            extinction=self.host_ext

        model_mag, Av, Rv = self.compute_model_mag(inst_filt, theta,
            extinction=extinction)

        # Catch bad model_mag value
        if model_mag is None:
            return(-np.inf, Av, Rv)

        # Flag missing data values
        mask = np.array(~np.isnan(model_mag))
        mag = mag[mask]
        magerr = magerr[mask]
        model_mag = model_mag[mask]

        # Handle limits
        if self.limits:
            limmask = magerr==0.0
            for m, mm in zip(mag[limmask], model_mag[limmask]):
                if m > mm:
                    return(-np.inf, Av, Rv)

            mag = mag[~limmask]
            magerr = magerr[~limmask]
            model_mag = model_mag[~limmask]

            # If all of the limits have passed check and there are no data
            # left, then we are in limit mode, so just return 1.0

            if len(mag)==0:
                return(-1.0, Av, Rv)

        chi2 = 1.0
        if self.extinction['likelihood']:
            chi2 *= self.extinction['interpolation'](Av, Rv)

        chi2 *= np.sum((mag-model_mag)**2/magerr**2)

        return(-1.0*chi2, Av, Rv)