import warnings
warnings.simplefilter('ignore')
import numpy as np
import os, glob
import sys
import pickle
import astropy.units as u
import astropy.constants as const
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy import interpolate
import traceback
import pandas as pd
import itertools
from tqdm import tqdm
from mcmc import mcmc
from mc_parallel import rsg_dataloader, mcmcfit
from multiprocessing import Pool
import argparse
import logging
import multiprocessing_logging
from datetime import datetime
import torch
from sbi import utils as sbi_utils
from sbi import inference
from sbi.neural_nets import posterior_nn
from sbi.analysis import plot_summary
from torch.distributions import MultivariateNormal, Exponential, LogNormal
from sbi.utils import MultipleIndependent, BoxUniform
import sbi_pp
import signal

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''

    parser = argparse.ArgumentParser(description='Fit red supergiant SEDs using SBI++')
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
    parser.add_argument('--keep_narrow', type=bool, default=False, help='Fit narrow band photometry?')
    parser.add_argument('--ignore_filts', nargs='*', help='Photometry to avoid fitting')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    parser.add_argument('--device', type=str, default='cpu', help='Device for PyTorch (CPU / GPU)')
    parser.add_argument('--sim_train', type=bool, default=False, help='Generate training set samples')
    parser.add_argument('--ntrain', type=float, default=4e5, help='Number of samples in simulated training set')

    return parser

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

        self.training_set_fname = os.path.join('data/sbi', f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        self.gen_mc_obj = mcmc(dm=0, dmerr=0, z=self.logz, model_type=self.modeltype, comp=self.comp)
        self.gen_mc_obj.verbose = False

    def sim_training_set(self, ntrain:int=int(4e5), prior_type = 'independent') -> None:
        if not isinstance(ntrain, int):
            ntrain = int(ntrain)

        sim = np.zeros((ntrain, len(self.gen_mc_obj.model_fit_params) + len(self.rsgloader.nrc_filts)))

        if prior_type=='independent':
            # uniform prior for temperature
            _temp = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['temperature'][0]]), 
                            high=torch.tensor([self.gen_mc_obj.bounds['temperature'][1]]), 
                            device=self.device).sample((ntrain,)).numpy()
            
            # uniform prior for dust temperature
            _dtemp = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['dust_temp'][0]]), 
                                high=torch.tensor([self.gen_mc_obj.bounds['dust_temp'][1]]), 
                                device=self.device).sample((ntrain,)).numpy()
            
            # exponential prior for tau_V
            _tauv = Exponential(torch.tensor([0.5])).sample((ntrain,)).numpy()
            _tauv[_tauv > 12.0] = 12.0

            # uniform prior for luminosity
            _lum = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['luminosity'][0]]), 
                              high=torch.tensor([self.gen_mc_obj.bounds['luminosity'][1]]), 
                              device=self.device).sample((ntrain,)).numpy()
            
            # uniform prior for Rv
            _rv = BoxUniform(low=torch.tensor([self.gen_mc_obj.bounds['Rv'][0]]), 
                             high=torch.tensor([self.gen_mc_obj.bounds['Rv'][1]]), 
                             device=self.device).sample((ntrain,)).numpy()
            
            # lognormal prior for Av
            _av = LogNormal(torch.tensor([-0.6]), torch.tensor([0.75])).sample((ntrain,)).numpy()
            _av[_av > 5.0] = 5.0

            sample_params = np.vstack((_temp.flatten(), _dtemp.flatten(), _tauv.flatten(), _lum.flatten(), _rv.flatten(), _av.flatten())).T

        elif prior_type=='uniform':
            sample_params = self.gen_mc_obj.get_init_pos(ntrain)

        else:
            raise ValueError(f'Invalid option {prior_type} for training set priors')

        # save model photometry for all samples
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](p_).flatten()[0] for f in self.rsgloader.nrc_filts]) + self.gen_mc_obj.dm
            sim[i, :len(self.gen_mc_obj.model_fit_params)] = p_
            sim[i, len(self.gen_mc_obj.model_fit_params):] = model_mag

        # save training set to csv
        cols = self.gen_mc_obj.model_fit_params + list(self.rsgloader.nrc_filts)
        df = pd.DataFrame(sim, columns=cols)
        sim_outdir = 'data/sbi'
        os.makedirs(sim_outdir, exist_ok=True)
        fname = os.path.join(sim_outdir, f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        self.training_set_fname = fname

        df.to_csv(fname, index=False)

    def _exp(self, x, a, x0, scale):
        return a*np.exp((x-x0)*scale)

    def sim_noise(self, train:pd.DataFrame, noise_floor:float=0.01, interp_bins:int=20) -> np.ndarray:
        erc_ = self.rsgloader.cols['errcols'][self.rsgloader.flt_mask]
        mc_ = self.rsgloader.cols['magcols'][self.rsgloader.flt_mask]
        tc_ = self.rsgloader.cols['flts'][self.rsgloader.flt_mask]
        train_err = np.zeros((len(train), len(erc_)))

        for i, (mcol, ecol, tcol) in enumerate(zip(mc_, erc_, tc_)):
            mag, err = self.rsgloader.rsgcat[mcol].values - self.rsgloader.dm, self.rsgloader.rsgcat[ecol].values
            tmag = train[tcol].values
            mask = np.isnan(err) | (err > 0.5)
            mag, err = mag[~mask], err[~mask]

            vmin, vmax = np.percentile(mag, [1, 99])
            mag_bins = np.linspace(vmin, vmax, interp_bins+1)
            bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])
            std_errs = np.array([np.std(err[(mag >= mag_bins[j]) & (mag < mag_bins[j+1])]) for j in range(interp_bins)])
            sig_interp = interpolate.InterpolatedUnivariateSpline(bin_centers, std_errs, k=3, ext=3)

            popt, _ = curve_fit(self._exp, mag, err)
            mean_errs = self._exp(tmag, *popt)
            sig_errs = sig_interp(tmag)

            norm_err = np.random.normal(mean_errs, sig_errs)
            norm_err = np.sqrt(norm_err**2 + noise_floor**2)
            train_err[:, i] = norm_err

        return train_err

    def baseline_sbi_model(self, prior_type='independent', hidden_features=50, ntransforms=5, 
                           nbins=10, batch_size=256, valfrac=0.1, stop_epochs=50, 
                           noise_floor=0.01, plot=False):
        assert self.rsgloader.rsgcat is not None

        ndim = int(len(self.gen_mc_obj.model_fit_params))
        train = pd.read_csv(self.training_set_fname)
        mags = train[train.columns[ndim:]]
        params = train[train.columns[:ndim]]

        x_train = np.array(params, dtype=float)
        y_mags = mags[self.rsgloader.cols['flts'][self.rsgloader.flt_mask]]
        train_err = self.sim_noise(train, noise_floor=noise_floor)
        y_err = pd.DataFrame(train_err, columns=self.rsgloader.cols['errcols'][self.rsgloader.flt_mask])
        y_train = pd.concat([y_mags, y_err], axis=1)
        y_train = np.array(y_train, dtype=float)
        self.x_train, self.y_train = x_train, y_train

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
                Exponential(torch.tensor([0.5])),
                BoxUniform(low=torch.tensor([prior_low[3]]), high=torch.tensor([prior_high[3]]), device=self.device),
                BoxUniform(low=torch.tensor([prior_low[4]]), high=torch.tensor([prior_high[4]]), device=self.device),
                LogNormal(torch.tensor([-0.6]), torch.tensor([0.75]))
            ])
        else:
            raise ValueError(f'Invalid option {prior_type} for prior')

        anpe = inference.NPE(prior=self.prior,
                            density_estimator=posterior_nn('nsf', hidden_features=hidden_features, num_transforms=ntransforms, num_bins=nbins, 
                                                            z_score_theta='independent', z_score_x='independent'),
                            device=self.device,)
        x_tensor = torch.as_tensor(self.x_train.astype(np.float32)).to(self.device)
        y_tensor = torch.as_tensor(self.y_train.astype(np.float32)).to(self.device)
        anpe.append_simulations(x_tensor, y_tensor)
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        savepath = os.path.join(self.procdir, f'npe_{curr_time}.pt')

        if not os.path.exists(savepath):
            self.logger.info('No trained model found. Training NPE...')
            p_x_y_estimator = anpe.train(training_batch_size=batch_size, use_combined_loss=True, validation_fraction=valfrac, 
                                        stop_after_epochs=stop_epochs, show_train_summary=True)
            # save trained NPE
            torch.save(p_x_y_estimator.state_dict(), savepath)
            pickle.dump(anpe._summary, open(os.path.join(self.procdir, f'npe_{curr_time}.p'), 'wb'))
            
            if plot:
                try:
                    _ = plot_summary(anpe, tags=["training_loss", "validation_loss"], figsize=(10, 2),)
                except:
                    print('Failed to plot')

        self.logger.info(f"Loaded trained NPE from {savepath}")
        p_x_y_estimator = anpe._build_neural_net(x_tensor, y_tensor)
        p_x_y_estimator.load_state_dict(torch.load(savepath, map_location=torch.device(self.device)))
        anpe._x_shape = sbi_utils.x_shape_from_simulation(y_tensor)
        hatp_x_y = anpe.build_posterior(p_x_y_estimator)   

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
    
    def infer_sbi(self):
        raise NotImplementedError()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.rsgcat is not None:
        rsgcat_in = pd.read_csv(args.rsgcat)
    else: rsgcat_in = None

    load_args = {
        'gal':args.gal, 'procdir':args.procdir, 'photfile_path':args.photfile_path,
        'dm':args.dm, 'dmerr':args.dmerr, 'z':args.z, 'trgb':tuple(args.trgb),
        'modeltype':args.modeltype, 'comp':args.comp,
        'keep_narrow':args.keep_narrow, 'agbcut':False, 'ignore_filts':args.ignore_filts,
        'rsgcat':rsgcat_in
    }

    rsgloader = rsg_dataloader(**load_args)
    sedfit = sbifit(rsg_dataloader=rsgloader, comp=args.comp, modeltype=args.modeltype, device=args.device)

    if args.sim_train:
        sedfit.sim_training_set(ntrain=int(args.ntrain))
        sys.exit()

    hatp_x_y = sedfit.baseline_sbi_model()