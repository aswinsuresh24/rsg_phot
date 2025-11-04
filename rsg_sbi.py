import warnings
warnings.simplefilter('ignore')
import numpy as np
import os, glob
import sys
import pickle
import astropy.units as u
import astropy.constants as const
from scipy.stats import gaussian_kde
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
import sbi_pp

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''

    parser = argparse.ArgumentParser(description='Fit red supergiant SEDs using SBI++')
    parser.add_argument('-g','--gal', type=str, default='gal', help='Galaxy name', required=True)
    parser.add_argument('-d', '--procdir', type=str, default='.', help='Directory to save processed photometry', required=True)
    parser.add_argument('-p', '--photfile_path', type=str, default='.', help='Root directory to search for dolphot photometry')
    parser.add_argument('--dm', type=float, default=30, help='Distance modulus')
    parser.add_argument('--dmerr', type=float, default=0.5, help='Distance modulus error')
    parser.add_argument('--z', type=float, default=0.0, help='Metallicity')
    parser.add_argument('--modeltype', type=str, default='MARCS', help='Family of RSG models to fit data to (MARCS / MARCS15 / NewEra)')
    parser.add_argument('--trgb', type=tuple, default=('F090W', 30), help='Tip of red giant branch')
    parser.add_argument('--comp', type=str, default='sil', help='Dust composition of RSG model (sil / grf)')
    parser.add_argument('--keep_narrow', type=bool, default=False, help='Fit narrow band photometry?')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    parser.add_argument('-r', '--rsgcat', type=str, default=None, help='Path to pre-processed rsgcat')
    parser.add_argument('--ignore_filts', nargs='*', help='Photometry to avoid fitting')
    parser.add_argument('--device', type=str, default='cpu', help='Device for PyTorch (CPU / GPU)')
    parser.add_argument('--sim_train', type=bool, default=False, help='Generate training set samples')
    parser.add_argument('--ntrain', type=float, default=4e5, help='Number of samples in simulated training set')

    return parser

class sbifit(object):
    def __init__(self, rsg_dataloader: rsg_dataloader, comp='sil', modeltype='MARCS', device='cpu'):

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

    def sim_training_set(self, ntrain:int = int(4e5)):
        if not isinstance(ntrain, int):
            ntrain = int(ntrain)

        self.ntrain = ntrain
        sim = np.zeros((ntrain, len(self.gen_mc_obj.model_fit_params) + len(self.rsgloader.nrc_filts)))

        self.gen_mc_obj.bounds['tau_V'] = [1e-4, 3]
        # sample uniformly for all parameters except tau_V
        sample_params = self.gen_mc_obj.get_init_pos(ntrain)
        # broken uniform dsitributions for tau_V
        f_ = int(0.8*ntrain)
        tau_4 = np.random.uniform(1e-4, 4, f_)
        tau_12 = np.random.uniform(4, 12, ntrain-f_)
        sample_params[:, 2] = np.hstack((tau_4, tau_12))

        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](p_).flatten()[0] for f in self.rsgloader.nrc_filts]) + self.gen_mc_obj.dm
            sim[i, :len(self.gen_mc_obj.model_fit_params)] = p_
            sim[i, len(self.gen_mc_obj.model_fit_params):] = model_mag

        cols = self.gen_mc_obj.model_fit_params + list(self.rsgloader.nrc_filts)
        df = pd.DataFrame(sim, columns=cols)
        sim_outdir = 'data/sbi'
        os.makedirs(sim_outdir, exist_ok=True)
        fname = os.path.join(sim_outdir, f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        self.training_set_fname = fname

        df.to_csv(fname, index=False)
        self.gen_mc_obj.reset_bounds()

    def sim_noise(self, train):
        train_err = np.zeros((len(train), len(self.rsgloader.cols['errcols'])))
        for i, col in enumerate(self.rsgloader.cols['errcols']):
            err = self.rsgloader.rsgcat[col].values
            mask = np.isnan(err) | (err > 0.5)
            err = err[~mask]
            kde = gaussian_kde(err)
            resamp_errs = kde.resample(len(train))
            resamp_errs[resamp_errs < 0.01] = 0.01
            train_err[:, i] = resamp_errs

        return train_err

    def train_sbi(self, plot=False):
        assert self.rsgloader.rsgcat is not None

        train = pd.read_csv(self.training_set_fname)
        mags = train[train.columns[6:]]
        params = train[train.columns[:6]]

        x_train = np.array(params, dtype=float)
        y_mags = mags[self.rsgloader.cols['flts']]
        train_err = self.sim_noise(train)
        y_err = pd.DataFrame(train_err, columns=self.rsgloader.cols['errcols'])
        y_train = pd.concat([y_mags, y_err], axis=1)
        y_train = np.array(y_train, dtype=float)

        prior_low   = sbi_pp.prior_from_train('ll', x_train=x_train)
        prior_high  = sbi_pp.prior_from_train('ul', x_train=x_train)
        lower_bounds = torch.tensor(prior_low).to(self.device)
        upper_bounds = torch.tensor(prior_high).to(self.device)
        prior = sbi_utils.BoxUniform(low=lower_bounds, high=upper_bounds, device=self.device)

        anpe = inference.NPE(prior=prior,
                            density_estimator=posterior_nn('nsf', hidden_features=50, num_transforms=5, num_bins=20, 
                                                            z_score_theta='independent', z_score_x='independent'),
                            device=self.device,)
        x_tensor = torch.as_tensor(x_train.astype(np.float32)).to(self.device)
        y_tensor = torch.as_tensor(y_train.astype(np.float32)).to(self.device)
        anpe.append_simulations(x_tensor, y_tensor)
        p_x_y_estimator = anpe.train(training_batch_size=256, use_combined_loss=True, validation_fraction=0.1, 
                                    stop_after_epochs=50, show_train_summary=True)
        
        # save trained NPE
        savepath = os.path.join(self.procdir, 'npe.pt')
        torch.save(p_x_y_estimator.state_dict(), savepath)
        pickle.dump(anpe._summary, open(os.path.join(self.procdir, 'npe.p'), 'wb'))

        if plot:
            try:
                _ = plot_summary(anpe, tags=["training_loss", "validation_loss"], figsize=(10, 2),)
            except:
                print('Failed to plot')

        return savepath
    
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
        'dm':args.dm, 'dmerr':args.dmerr, 'z':args.z, 'trgb':args.trgb,
        'modeltype':args.modeltype, 'comp':args.comp,
        'keep_narrow':args.keep_narrow, 'agbcut':False, 'ignore_filts':args.ignore_filts,
        'rsgcat':rsgcat_in
    }

    rsgloader = rsg_dataloader(**load_args)
    sedfit = sbifit(rsg_dataloader=rsgloader, comp=args.comp, modeltype=args.modeltype, device=args.device)

    if args.sim_train:
        sedfit.sim_training_set(ntrain=int(args.ntrain))