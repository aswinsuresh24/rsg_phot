import warnings
warnings.simplefilter('ignore')
import numpy as np
import os, glob
import sys
import pickle
import astropy.units as u
import astropy.constants as const
from scipy.stats import gaussian_kde, skewnorm
from scipy.optimize import curve_fit
from scipy import interpolate
import traceback
import pandas as pd
import itertools
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import logging
import multiprocessing_logging
from datetime import datetime
import torch
from sbi import utils as sbi_utils
from sbi.utils import process_prior
from sbi import inference
from sbi.neural_nets import posterior_nn
from sbi.analysis import plot_summary
from torch.distributions import Exponential, LogNormal
from sbi.utils import MultipleIndependent, BoxUniform
from sbi.analysis.plot import sbc_rank_plot, plot_tarp
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
import rsg_phot.sbi_pp as sbi_pp
import signal
import trackio
import json, hashlib
from pathlib import Path

from rsg_phot.mcmc import mcmc
from rsg_phot.mc_parallel import rsg_dataloader
from rsg_phot.utils import NewlineStdout

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''

    parser = argparse.ArgumentParser(description='Fit red supergiant SEDs using SBI++')
    # rsgloader arguments
    parser.add_argument('-g','--gal', type=str, help='Galaxy name', required=True)
    parser.add_argument('-d', '--procdir', type=str, default='.', help='Directory to save processed photometry', required=True)
    parser.add_argument('-p', '--photfile_path', type=str, default=None, help='Root directory to search for dolphot photometry')
    parser.add_argument('-r', '--rsgcat', type=str, default=None, help='Path to pre-processed rsgcat')
    parser.add_argument('--dm', type=float, default=30, help='Distance modulus')
    parser.add_argument('--dmerr', type=float, default=0.5, help='Distance modulus error')
    parser.add_argument('--z', type=float, default=0.0, help='Metallicity')
    parser.add_argument('--trgb', nargs='*', default=('F090W', 30), help='Tip of red giant branch')
    parser.add_argument('--modeltype', type=str, default='MARCS', help='Family of RSG models to fit data to (MARCS / MARCS15 / NewEra)')
    parser.add_argument('--comp', type=str, default='sil', help='Dust composition of RSG model (sil / grf)')
    parser.add_argument('--keep_narrow', default=False, action=argparse.BooleanOptionalAction, help='Fit narrow band photometry?')
    parser.add_argument('--ignore_filts', nargs='*', help='Photometry to avoid fitting')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')

    # SBI model arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device for PyTorch (CPU / GPU)')
    parser.add_argument('--sim_train', default=False, action=argparse.BooleanOptionalAction, help='Generate training set samples?')
    parser.add_argument('--ntrain', type=int, default=int(4e5), help='Number of samples in simulated training set')
    parser.add_argument('--aug_train', default=False, action=argparse.BooleanOptionalAction, help='Augment training set?')
    parser.add_argument('--clip_bright', default=False, action=argparse.BooleanOptionalAction, help='Clip ultra-bright sources from training set?')
    parser.add_argument('--aug_size', type=int, default=int(4e5), help='Number of samples in augmented training set')
    parser.add_argument('--flow_model', type=str, default='nsf', help='Flow model for neural posterior estimation (nsf / maf / mdn)')
    parser.add_argument('--hidden_features', type=int, default=50, help='Number of hidden features')
    parser.add_argument('--ntransforms', type=int, default=5, help='Number of transforms in normalizing flow')
    parser.add_argument('--nbins', type=int, default=10, help='Number of bins for spline flow (only for nsf)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for the SBI neural net')
    parser.add_argument('--stop_epochs', type=int, default=50, help='Patience epochs for SBI training')
    parser.add_argument('--use_combined_loss', default=False, action=argparse.BooleanOptionalAction, help='Use combined loss to train the neural net?')

    return parser


class TruncatedExponential(torch.distributions.Distribution):
    def __init__(self, rate:torch.Tensor, low:torch.Tensor, high:torch.Tensor, 
                 return_numpy:bool=False, validate_args = None, device:str='cpu'):

        self.rate = rate
        self.low, self.high = low, high
        self.device = device

        self.base = Exponential(self.rate)
        self.Z = torch.Tensor(self.base.cdf(torch.tensor(self.high)) - self.base.cdf(torch.tensor(self.low)))

        batch_shape = torch.Size([])
        event_shape = torch.Size([1])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.return_numpy = return_numpy

    @property
    def arg_constraints(self):
        return {
            "rate": torch.distributions.constraints.greater_than(lower_bound=0.0),
            "low": torch.distributions.constraints.greater_than(lower_bound=0),
            "high": torch.distributions.constraints.less_than(upper_bound=12.0)
        }
    
    @property
    def support(self):
        return torch.distributions.constraints.interval(lower_bound=self.low, upper_bound=self.high)
    
    def sample(self, sample_shape=torch.Size([])):
        u = torch.rand(sample_shape, device=self.device)
        u = u * self.Z + float(self.base.cdf(torch.tensor(self.low)))
        x = -torch.log1p(-u) / self.rate
        x = x.unsqueeze(-1)

        return x
    
    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values, device=self.device)
        lp = self.base.log_prob(values) - torch.log(self.Z)
        mask = (values >= self.low) & (values <= self.high)
        log_probs = torch.where(mask, lp, torch.tensor(-float("inf"), device=self.device))
        log_probs = log_probs.squeeze(-1)

        return log_probs.numpy() if self.return_numpy else log_probs
    

class sbifit(object):
    def __init__(self, rsg_dataloader:rsg_dataloader, comp:str='sil', modeltype:str='MARCS', device:str='cpu'):

        self.rsgloader = rsg_dataloader
        self.procdir = os.path.join(self.rsgloader.procdir, 'sbi')
        os.makedirs(self.procdir, exist_ok=True)

        self.rsgcat = self.rsgloader.rsgcat
        self.logger = self.rsgloader.logger
        self.logz = self.rsgloader.z
        self.comp = comp
        self.modeltype = modeltype
        self.device = device

        self.training_set_fname = os.path.join(os.pardir, 'data', 'sbi', f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        self.gen_mc_obj = mcmc(dm=0, dmerr=0, z=self.logz, model_type=self.modeltype, comp=self.comp)

        # in practice, rsgcat has a cut logL ~ 3.5 - avoid wasting 15% of the training set 
        # on observations the model will never have to infer on
        self.gen_mc_obj.bounds['luminosity'] = [3.4, 6.0]
        self.gen_mc_obj.verbose = False

    def sample_exp_prior(self, nsamp:int):
        #uniform prior for temperature                                           
        _temp = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['temperature'][0]]), 
                           high=torch.tensor([self.gen_mc_obj.bounds['temperature'][1]]), 
                           device=self.device).sample((nsamp,)).numpy()
        
        # uniform prior for dust temperature
        _dtemp = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['dust_temp'][0]]), 
                            high=torch.tensor([self.gen_mc_obj.bounds['dust_temp'][1]]), 
                            device=self.device).sample((nsamp,)).numpy()
        
        # truncated exponential prior for tau_V
        _tauv = TruncatedExponential(rate=torch.tensor([0.5]),
                                     low=torch.tensor([self.gen_mc_obj.bounds['tau_V'][0]]),
                                     high=torch.tensor([self.gen_mc_obj.bounds['tau_V'][1]]),
                                     device=self.device).sample((nsamp,)).numpy()

        # uniform prior for luminosity
        _lum = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['luminosity'][0]]), 
                          high=torch.tensor([self.gen_mc_obj.bounds['luminosity'][1]]), 
                          device=self.device).sample((nsamp,)).numpy()
        
        # uniform prior for Rv
        _rv = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['Rv'][0]]), 
                         high=torch.tensor([self.gen_mc_obj.bounds['Rv'][1]]), 
                         device=self.device).sample((nsamp,)).numpy()
        
        # truncated exponential prior for Av
        _av = TruncatedExponential(rate=torch.tensor([0.5]),
                                   low=torch.tensor([self.gen_mc_obj.bounds['Av'][0]]),
                                   high=torch.tensor([self.gen_mc_obj.bounds['Av'][1]]),
                                   device=self.device).sample((nsamp,)).numpy()

        sample_params = np.vstack((_temp.flatten(), _dtemp.flatten(), _tauv.flatten(), _lum.flatten(), _rv.flatten(), _av.flatten())).T

        return sample_params

    def sim_training_set(self, ntrain:int=int(4e5), prior_type = 'mixed', mix_frac=0.3) -> None:
        if not isinstance(ntrain, int):
            ntrain = int(ntrain)

        if prior_type not in ['uniform', 'independent', 'mixed']:
            raise ValueError(f'prior_type must be "independent", "uniform" or "mixed"')

        sim = np.zeros((ntrain, len(self.gen_mc_obj.model_fit_params) + len(self.rsgloader.nrc_filts)))

        if prior_type=='independent':
            sample_params = self.sample_exp_prior(ntrain)

        elif prior_type=='uniform':
            sample_params = self.gen_mc_obj.get_init_pos(ntrain)

        elif prior_type=='mixed':
            n_uniform = int(mix_frac*ntrain)
            n_prior = int(ntrain - n_uniform)

            prior_samp = self.sample_exp_prior(n_prior)
            uniform_samp = self.gen_mc_obj.get_init_pos(n_uniform)
            sample_params = np.vstack((prior_samp, uniform_samp))

        self.logger.info(f'Generating {ntrain} training samples following {prior_type} prior')
        # save model photometry for all samples
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](p_).flatten()[0] for f in self.rsgloader.nrc_filts]) + self.gen_mc_obj.dm
            sim[i, :len(self.gen_mc_obj.model_fit_params)] = p_
            sim[i, len(self.gen_mc_obj.model_fit_params):] = model_mag

        # save training set to csv
        cols = self.gen_mc_obj.model_fit_params + list(self.rsgloader.nrc_filts)
        df = pd.DataFrame(sim, columns=cols)
        sim_outdir = os.path.join(os.pardir, 'data', 'sbi')
        os.makedirs(sim_outdir, exist_ok=True)
        fname = os.path.join(sim_outdir, f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        self.training_set_fname = fname

        df.to_csv(fname, index=False)

    def sim_mag_err(self, train:pd.DataFrame, noise_floor:float=0.01, interp_bins:int=30) -> np.ndarray:
        erc_ = self.rsgloader.cols['errcols'][self.rsgloader.flt_mask]
        mc_ = self.rsgloader.cols['magcols'][self.rsgloader.flt_mask]
        tc_ = self.rsgloader.cols['flts'][self.rsgloader.flt_mask]
        train_err = np.zeros((len(train), len(erc_)))

        for i, (mcol, ecol, tcol) in enumerate(zip(mc_, erc_, tc_)):
            mag, err = self.rsgloader.rsgcat[mcol].values - self.rsgloader.dm, self.rsgloader.rsgcat[ecol].values
            tmag = train[tcol].values
            mask = np.isnan(err) | (err > 1.0) |np.isnan(mag) | (mag > 36.0 - self.rsgloader.dm) 
            mag, err = mag[~mask], err[~mask]

            vmin, vmax = np.percentile(mag, [0.3, 99.7]) 
            mag_bins = np.linspace(vmin, vmax, interp_bins+1)
            #BUG: if any bin in interp_bins is empty, this doesn't work
            bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])
            std_errs = np.array([np.std(err[(mag >= mag_bins[j]) & (mag < mag_bins[j+1])], ddof=1) for j in range(interp_bins)])
            mu_errs = np.array([np.mean(err[(mag >= mag_bins[j]) & (mag < mag_bins[j+1])]) for j in range(interp_bins)])

            mu_interp = interpolate.InterpolatedUnivariateSpline(bin_centers, mu_errs, k=3, ext=3)
            sig_interp = interpolate.InterpolatedUnivariateSpline(bin_centers, std_errs, k=3, ext=3)
            mean_train_errs = mu_interp(tmag)
            sig_train_errs = sig_interp(tmag)

            norm_err = np.random.normal(mean_train_errs, sig_train_errs)
            norm_err = np.sqrt(norm_err**2 + noise_floor**2)
            # norm_err[tmag > max(mag)] = mu_interp(max(mag))
            train_err[:, i] = norm_err

        return train_err
    
    def sim_skew_mag_err(self, train, noise_floor=0.01, interp_bins=30, sigma_f = 0.3):
        erc_ = self.rsgloader.cols['errcols'][self.rsgloader.flt_mask]
        mc_ = self.rsgloader.cols['magcols'][self.rsgloader.flt_mask]
        tc_ = self.rsgloader.cols['flts'][self.rsgloader.flt_mask]
        train_err = np.zeros((len(train), len(erc_)))

        for i, (mcol, ecol, tcol) in enumerate(zip(mc_, erc_, tc_)):
            mag, err = self.rsgloader.rsgcat[mcol].values - self.rsgloader.dm, self.rsgloader.rsgcat[ecol].values
            tmag = train[tcol].values
            mask = np.isnan(err) | (err > 1.0) | (mag > 36.0 - self.rsgloader.dm) | np.isnan(mag)
            mag, err = mag[~mask], err[~mask]

            vmin, vmax = np.percentile(mag, [0.1, 99.9]) 
            mag_bins = np.linspace(vmin, vmax, interp_bins+1)

            sig_edge = None
            for j in range(interp_bins):
                #BUG: if len(ebin_) == 0, this doesn't work
                ebin_ = err[(mag >= mag_bins[j]) & (mag < mag_bins[j+1])]
                tmask_ = (tmag >= mag_bins[j]) & (tmag < mag_bins[j+1])
                if tmask_.sum() == 0:
                    continue
                tbin_ = tmag[tmask_]
                if len(ebin_) < 30:
                    sig_e = np.std(ebin_, ddof=1)
                    mu_e = np.median(ebin_)
                    ae = 0.0
                else:
                    try:
                        ae, mu_e, sig_e = skewnorm.fit(ebin_)
                        if (ae < 0.0) | (mu_e < 0.0) | (sig_e < 0.0):
                            ae, mu_e, sig_e = 0.0, np.median(ebin_), np.std(ebin_, ddof=1)
                    except Exception as e:
                        self.logger.info(traceback.format_exc())
                        sig_e = np.std(ebin_, ddof=1)
                        mu_e = np.median(ebin_)
                        ae = 0.0
                
                if j == interp_bins - 1:
                    sig_edge = sig_e

                resamp_err = skewnorm.rvs(ae, mu_e, sig_e, size=len(tbin_))
                resamp_err = np.sqrt(resamp_err**2 + noise_floor**2)
                train_err[:, i][tmask_] = resamp_err

            bright_mask = tmag < vmin
            bright_err = np.random.normal(0.0, 0.005, size=bright_mask.sum())
            train_err[:, i][bright_mask] = np.sqrt(bright_err**2 + noise_floor**2)

            faint_mask = tmag >= vmax + 0.5
            faint_err = np.random.normal(sigma_f, sigma_f*0.1, size=faint_mask.sum())
            train_err[:, i][faint_mask] = np.sqrt(faint_err**2 + noise_floor**2) 

            transition_mask = (tmag >= vmax) & (tmag < vmax+0.5)
            transition_mags = tmag[transition_mask]
            w = np.clip(np.abs(transition_mags - vmax) / 0.5, a_min=None, a_max=1.0)
            sig_t = (1 - w) * sig_edge + w * sigma_f
            transition_err = np.random.normal(sig_t, 0.1*sig_t)
            train_err[:, i][transition_mask] = np.sqrt(transition_err**2 + noise_floor**2)

            train_err[:, i] = np.minimum(train_err[:, i], 0.6)

        return train_err

    
    def toy_noise_model(self, train:pd.DataFrame, noise_floor:float=0.01, interp_bins:int=30, fit:bool=False):
        """
        toy Gaussian noise model for SBI++
        
        :param train: pd.DataFrame
            training set
        :param noise_floor: float, default=0.01
            noise floor to be added in quadrature with
            observed error
        :param interp_bins: int, default=30
            number of bins to interpolate mean and 1 sigma
            scatter of error bars
        :param fit: bool, default=False
            fit to broken power law instead of interpolating?

        """
        raise NotImplementedError()
    
    def sim_obs_noise(self, ymags, sim_type='chisq'):
        rsgcat = self.rsgloader.rsgcat
        noise_pdf = np.zeros_like(ymags.values)
        if sim_type == 'chisq':
            self.logger.info(f'Adding simulator jitter based on chi sq values')
            off_mags = np.zeros_like(rsgcat[self.rsgloader.cols['magcols'][self.rsgloader.flt_mask]].values)
            for i, idx in enumerate(rsgcat.index):
                try:
                    d = 0.0
                    row = rsgcat.loc[idx]
                    obsmag = row[self.rsgloader.cols['magcols'][self.rsgloader.flt_mask]]
                    t_, td_, l_, tu_, a_ = row['teff_chisq'], row['tdust_chisq'], np.log10(row['lum_chisq']), row['tau_chisq'], row['av_chisq']
                    if l_ > 6.0: 
                        d = l_ - 6.0
                        l_ = 6.0
                    chisq_params = np.meshgrid([t_, td_, tu_, l_, 3.1, a_], indexing='ij', sparse=True)
                    model_mag = np.array([self.rsgloader.gen_mc_obj.model[f](chisq_params).flatten()[0] for f in self.rsgloader.cols['flts'][self.rsgloader.flt_mask]]) + self.rsgloader.gen_mc_obj.dm
                    model_mag = model_mag - 2.5*d
                    off_mags[i, :] = (obsmag - model_mag).values
                except Exception as e:
                    self.logger.info(e)
                    off_mags[i, :] = 90.0

            nsamp = len(ymags)
            for i, offs in enumerate(off_mags.T):
                mask = (np.abs(offs) > 1) | np.isnan(offs) | np.isinf(offs)
                try:
                    ae, mu_e, sig_e = skewnorm.fit(offs[~mask])
                    if (ae < 0.0) | (mu_e < 0.0) | (sig_e < 0.0):
                        ae, mu_e, sig_e = 0.0, np.median(offs[~mask]), np.std(offs[~mask], ddof=1)
                    resamp_noise = skewnorm.rvs(ae, mu_e, sig_e, size=nsamp)
                except:
                    mu, sig = np.mean(offs[~mask]), np.std(offs[~mask], ddof=1)
                    resamp_noise = np.random.normal(mu, sig, size=nsamp)
                noise_pdf[:, i] = resamp_noise

        elif sim_type == 'random':
            self.logger.info(f'Adding simulator jitter using random Gaussian noise')
            for i in range(len(self.rsgloader.cols['magcols'][self.rsgloader.flt_mask])):
                noise_pdf[:, i] = np.random.normal(0.0, 0.1, size=len(ymags))

        ymags = ymags + noise_pdf
        return ymags
    
    def clip_bright_train_samples(self, train):
        train_mags = (self.rsgloader.rsgcat[self.rsgloader.cols['magcols'][self.rsgloader.flt_mask]] - self.rsgloader.dm)
        max_mags = []
        for i in train_mags.columns:
            x_ = train_mags[i]
            x_ = x_[(x_ > -15) & (x_ < 10)]
            max_mags.append(np.percentile(x_, 1))
        max_mags = np.array(max_mags)
        clip_mask = (train[self.rsgloader.cols['flts'][self.rsgloader.flt_mask]] > max_mags).all(axis=1)
        train = train[clip_mask]
        self.logger.info(f'Clipped bright sources in training set, to size {len(train)}')

        return train
    
    def augment_training_set(self, train, size=int(4e5)):
        n = len(train)
        reps = int(np.ceil(size / n))
        self.logger.info(f'Augmenting training set by duplicating {reps} times and sampling {size} simulations')
        aug = pd.concat([train] * reps, ignore_index=True)
        aug = aug.sample(n=size, replace=False).reset_index(drop=True)

        return aug
    
    def model_id_from_config(self, config: dict, n_chars=8):
        config_str = json.dumps(config, sort_keys=True)
        h = hashlib.sha1(config_str.encode()).hexdigest()
        return h[:n_chars]
    
    def model_info(self, savepath):
        with open(savepath.replace('.pt', '.json'), 'rb') as f:
            config = json.load(f)

        with open(savepath.replace('.pt', '.p'), 'rb') as f:
            loss = pickle.load(f)

        self.logger.info('SBI config:' + '\n' + json.dumps(config, indent=2))

        plt.figure()
        plt.plot(loss['validation_loss'], label='Val loss', color='royalblue')
        plt.plot(loss['training_loss'], label='Train loss', color='orange', ls='--')
        plt.legend()
        plt.show();

    def baseline_sbi_model(self, prior_type='independent', augment_train=False, augment_size=int(3e5), 
                           clip_bright=False, flow_model='nsf', hidden_features=15, ntransforms=3, nbins=10,
                           use_combined_loss=False, batch_size=256, valfrac=0.1, stop_epochs=50, noise_floor=0.01,
                           savepath=None):
        assert self.rsgloader.rsgcat is not None

        ndim = int(len(self.gen_mc_obj.model_fit_params))
        if augment_train:
            load_train = os.path.join(self.procdir, f'train_{self.rsgloader.gal}_aug.csv')
        else:
            load_train = os.path.join(self.procdir, f'train_{self.rsgloader.gal}.csv')
        if os.path.exists(load_train):
            self.logger.info(f'Training set exists; Loading x and y train from {load_train}')
            train_set = pd.read_csv(load_train)
            params = train_set[train_set.columns[:ndim]]
            phot = train_set[train_set.columns[ndim:]]
            self.x_train = params.to_numpy(dtype=np.float32)
            self.y_train = phot.to_numpy(dtype=np.float32)
        else:
            self.logger.info(f'Training set does not exist; Will be saved at {load_train}')
            train = pd.read_csv(self.training_set_fname)
            if clip_bright:
                train = self.clip_bright_train_samples(train)
            if augment_train:
                train = self.augment_training_set(train, augment_size)
            train['temperature'] = train['temperature']/1e3
            train['dust_temp'] = train['dust_temp']/1e3
            mags = train[train.columns[ndim:]]
            params = train[train.columns[:ndim]]

            self.x_train = params.to_numpy(dtype=np.float32)
            y_mags = mags[self.rsgloader.cols['flts'][self.rsgloader.flt_mask]]
            y_mags = self.sim_obs_noise(y_mags, sim_type='chisq')
            
            try:
                self.logger.info('Modeling magnitude dependent noise using skewnorm distributions')
                train_err = self.sim_skew_mag_err(train, noise_floor=noise_floor, interp_bins=30)
            except Exception as e:
                self.logger.info(traceback.format_exc())
                self.logger.info('Modeling magnitude dependent noise using splines')
                train_err = self.sim_mag_err(train, noise_floor=noise_floor, interp_bins=30)
            y_err = pd.DataFrame(train_err, columns=self.rsgloader.cols['errcols'][self.rsgloader.flt_mask])
            y_phot = pd.concat([y_mags, y_err], axis=1)
            self.y_train = y_phot.to_numpy(dtype=np.float32)

            train_set = pd.concat([params, y_phot], axis=1)
            train_set.to_csv(load_train, index=False)

        prior_low = sbi_pp.prior_from_train('ll', x_train=self.x_train)
        prior_high = sbi_pp.prior_from_train('ul', x_train=self.x_train)

        if prior_type.lower() == 'uniform':
            lower_bounds = torch.tensor(prior_low).to(self.device)
            upper_bounds = torch.tensor(prior_high).to(self.device)
            self.prior = sbi_utils.BoxUniform(low=lower_bounds, high=upper_bounds, device=self.device)
        elif prior_type.lower() == 'independent':
            self.prior = MultipleIndependent([
                BoxUniform(low=torch.tensor([prior_low[0]]), high=torch.tensor([prior_high[0]]), device=self.device),
                BoxUniform(low=torch.tensor([prior_low[1]]), high=torch.tensor([prior_high[1]]), device=self.device),
                TruncatedExponential(rate=torch.tensor([0.5]), low=torch.Tensor([prior_low[2]]), 
                                     high=torch.Tensor([prior_high[2]]), device=self.device),
                BoxUniform(low=torch.tensor([prior_low[3]]), high=torch.tensor([prior_high[3]]), device=self.device),
                BoxUniform(low=torch.tensor([prior_low[4]]), high=torch.tensor([prior_high[4]]), device=self.device),
                TruncatedExponential(rate=torch.tensor([0.5]), low=torch.Tensor([prior_low[5]]), 
                                     high=torch.Tensor([prior_high[5]]), device=self.device),
            ])
        else:
            raise ValueError(f'Invalid option {prior_type} for prior')

        anpe = inference.NPE(prior=self.prior,
                            density_estimator=posterior_nn(model=flow_model, hidden_features=hidden_features, num_transforms=ntransforms, 
                                                           num_bins=nbins, z_score_theta='independent', z_score_x='independent'),
                            device=self.device,)
        x_tensor = torch.as_tensor(self.x_train.astype(np.float32)).to(self.device)
        y_tensor = torch.as_tensor(self.y_train.astype(np.float32)).to(self.device)
        anpe.append_simulations(x_tensor, y_tensor)

        # define SBI config
        sbi_config = {
            'galaxy': self.rsgloader.gal,
            'prior_type': prior_type,
            'augment_train': str(augment_train),
            'flow_model': flow_model,
            "hidden_features":hidden_features,
            "ntransforms": ntransforms,
            "nbins_nsf": nbins,
            "batch_size": batch_size,
            "use_combined_loss": str(use_combined_loss),
            "patience": stop_epochs,
            "lr": 5e-4,
            "ntrain": len(self.x_train)
        }

        if savepath is None:
            self.savepath = os.path.join(self.procdir, f"npe_{self.model_id_from_config(config=sbi_config, n_chars=8)}.pt")
        else: 
            self.savepath = savepath

        if not os.path.exists(self.savepath):
            self.logger.info('No trained model found. Training NPE...')
            # start experiment tracking 
            trackio.init(project="jwst-rsg-sbi", config=sbi_config)
            sys.stdout = NewlineStdout(sys.stdout)
            p_x_y_estimator = anpe.train(training_batch_size=batch_size, use_combined_loss=use_combined_loss, validation_fraction=valfrac, 
                                         stop_after_epochs=stop_epochs, show_train_summary=True, max_num_epochs=20)
            # save trained NPE
            torch.save(p_x_y_estimator.state_dict(), self.savepath)
            pickle.dump(anpe._summary, open(self.savepath.replace('.pt', '.p'), 'wb'))
            with open(self.savepath.replace('.pt', '.json'), "w") as f:
                json.dump(sbi_config, f, indent=2)

            summary = anpe._summary
            for epoch, (train_loss, val_loss) in enumerate(zip(summary["training_loss"], summary["validation_loss"])):
                trackio.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

            trackio.save(self.savepath)
            trackio.finish()

        self.logger.info(f"Loaded trained NPE from {self.savepath}")
        p_x_y_estimator = anpe._build_neural_net(x_tensor, y_tensor)
        p_x_y_estimator.load_state_dict(torch.load(self.savepath, map_location=torch.device(self.device)))
        anpe._x_shape = sbi_utils.x_shape_from_simulation(y_tensor)
        hatp_x_y = anpe.build_posterior(p_x_y_estimator)
        self.hatp_x_y = hatp_x_y

        return hatp_x_y
        
    def _sampling_timeout_handler(self, signum, frame):
        raise TimeoutError("Sampling exceeded time limit")

    def sample_with_timeout(self, hatp, sample_shape=(2500,), x=None, timeout=10, reset_handler=False, **kwargs):
        cur_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._sampling_timeout_handler)
        signal.alarm(int(timeout))
        try:
            result = hatp.sample(sample_shape, x=x, **kwargs)
        except TimeoutError:
            print(f"Sampling timed out after {timeout} seconds.")
            result = None
        finally:
            signal.alarm(0) 
            if reset_handler: signal.signal(signal.SIGALRM, cur_handler) 
        return result
    
    def simulator(self, theta_in):
        theta_in[:, 0] = theta_in[:, 0]*1e3
        theta_in[:, 1] = theta_in[:, 1]*1e3
        out = np.zeros((len(theta_in), len(self.rsgloader.cols['flts'][self.rsgloader.flt_mask])))

        for i, theta in enumerate(theta_in):
            model_mag = np.array([self.gen_mc_obj.model[f](theta).flatten()[0] for f in self.rsgloader.cols['flts'][self.rsgloader.flt_mask]]) + self.gen_mc_obj.dm
            out[i, :] = model_mag

        outdf = pd.DataFrame(out, columns=self.rsgloader.cols['flts'][self.rsgloader.flt_mask])
        noise = self.sim_skew_mag_err(outdf) 

        out = np.hstack((out, noise))
        out = torch.as_tensor(out.astype(np.float32)).to('cpu')
        return out
    
    def run_sbc_tarp(self, num_sbc_samples=500, num_posterior_samples=2500, num_workers=1):
        # generate ground truth parameters and corresponding simulated observations for SBC.
        thetas = self.prior.sample((num_sbc_samples,))
        self.logger.info(f'Running SBC on {len(thetas)} samples')
        xs = self.simulator(thetas.clone().detach())

        ranks, dap_samples = run_sbc(
            thetas,
            xs,
            self.hatp_x_y,
            num_posterior_samples=num_posterior_samples,
            num_workers=num_workers,
            use_sample_batched=False,  # True can give a speed-up, but might cause memory issues.
        )

        check_stats = check_sbc(
            ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
        )

        self.logger.info("SBC ks pval")
        self.logger.info(check_stats)

        f, ax = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=None,  # by passing None we use a heuristic for the number of bins.
            parameter_labels=self.gen_mc_obj.model_fit_params,
        )
        plt.show()

        f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf", parameter_labels=self.gen_mc_obj.model_fit_params)
        plt.show()

        #TARP
        ecp, alpha = run_tarp(
            thetas,
            xs,
            self.hatp_x_y,
            references=None,  # will be calculated automatically.
            num_posterior_samples=2500,
        )

        atc, ks_pval = check_tarp(ecp, alpha)
        self.logger.info("TARP ATC (should be close to 0)")
        self.logger.info(atc)
        self.logger.info("TARP ks pval (should be larger than 0.05)")
        self.logger.info(ks_pval)

        plot_tarp(ecp, alpha);
        plt.show()

    def test_accuracy(self, nsamp=1000, num_posterior_samples=2500):
        thetas = self.prior.sample((nsamp,))
        self.logger.info(f'Test accuracy using {len(thetas)} samples')
        xs = self.simulator(thetas.clone().detach())

        truths = []
        est = []
        err16, err84 = [], []
        for idx in tqdm(range(nsamp)):
            i = xs[idx]
            self.hatp_x_y.set_default_x(i)
            samp = self.hatp_x_y.sample((num_posterior_samples,), show_progress_bars=False)
            lp = self.hatp_x_y.log_prob(samp)
            samp = samp[lp > torch.quantile(lp, 0.1)]
            samp = samp.numpy()
            med = np.median(samp, axis=0)
            p16, p84 = np.percentile(samp, 16, axis=0), np.percentile(samp, 84, axis=0)
            truths.append(thetas[idx])
            est.append(med)
            err16.append(p16); err84.append(p84)

        truths = np.array(truths)
        est = np.array(est)
        err16 = np.array(err16)
        err84 = np.array(err84)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        labels = self.gen_mc_obj.model_fit_params
        for i in range(6):  
            ax = axes.flatten()[i]
            ax.errorbar(truths[:, i], est[:, i], yerr=[est[:, i]-err16[:, i], err84[:, i]-est[:, i]], fmt='o', 
                        ecolor='gray', alpha=0.5, label='Estimated', markersize=4)
            ax.scatter(truths[:, i], truths[:, i], color='red', label='Truth', s=10)
            ax.plot([truths[:, i].min(), truths[:, i].max()], [truths[:, i].min(), truths[:, i].max()], 
                    ls='--', color='black', alpha=0.7)
            r2, rms = r2_score(truths[:, i], est[:, i]), rmse(truths[:, i], est[:, i])
            ax.annotate(f'R2: {r2:.3f}\nRMSE: {rms:.3f}', xy=(0.7, 0.05), xycoords='axes fraction')
            ax.set_xlabel('Simulated ' + labels[i])
            ax.set_ylabel('Estimated ' + labels[i])
            ax.legend(loc='upper left')

    def run_calibration(self, num_sbc_samples=500, num_posterior_samples=2500, num_workers=1, num_test_samples=1000):
        self.run_sbc_tarp(num_sbc_samples=num_sbc_samples, num_posterior_samples=num_posterior_samples,
                          num_workers=num_workers)
        self.test_accuracy(nsamp=num_test_samples, num_posterior_samples=num_posterior_samples)
    
    def infer(self):
        raise NotImplementedError()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.rsgcat is not None:
        rsgcat_in = pd.read_csv(args.rsgcat)
    else: rsgcat_in = None

    load_args = {
        'gal':args.gal, 'procdir':args.procdir, 'photfile_path':args.photfile_path,
        'dm':args.dm, 'dmerr':args.dmerr, 'z':args.z, 'trgb':tuple([str(args.trgb[0]), float(args.trgb[1])]),
        'modeltype':args.modeltype, 'comp':args.comp,
        'keep_narrow':args.keep_narrow, 'agbcut':False, 'ignore_filts':args.ignore_filts,
        'rsgcat':rsgcat_in
    }

    rsgloader = rsg_dataloader(**load_args)
    sedfit = sbifit(rsg_dataloader=rsgloader, comp=args.comp, modeltype=args.modeltype, device=args.device)

    if args.sim_train:
        sedfit.sim_training_set(ntrain=int(args.ntrain))
        sys.exit()

    hatp_x_y = sedfit.baseline_sbi_model(prior_type='independent', augment_train=args.aug_train, augment_size=int(args.aug_size), 
                                         clip_bright=args.clip_bright, flow_model=args.flow_model, hidden_features=args.hidden_features, 
                                         ntransforms=args.ntransforms, nbins=args.nbins, use_combined_loss=args.use_combined_loss,
                                         batch_size=args.batch_size, valfrac=0.1, stop_epochs=args.stop_epochs)