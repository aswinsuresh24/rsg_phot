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
    parser.add_argument('--procdir', type=str, default='.', help='Directory to save fit results', required=True)
    parser.add_argument('--z', type=float, default=0.0, help='Metallicity')
    parser.add_argument('--comp', type=str, default='sil', help='Dust composition of RSG model (sil / grf)')
    parser.add_argument('--modeltype', type=str, default='MARCS', help='Family of RSG models to fit data to (MARCS / MARCS15 / NewEra)')
    parser.add_argument('--ntrain', type=float, default=3e5, help='Number of samples in simulated training set')

    return parser

class sbifit(object):
    def __init__(self, procdir, comp='sil', modeltype='MARCS', z=0.0, ntrain=3e5, device='cpu'):

        self.procdir = procdir
        os.makedirs(self.procdir, exist_ok=True)

        self.logz = z
        self.comp = comp
        self.modeltype = modeltype
        self.ntrain = int(ntrain)
        self.device = device

        self.gen_mc_obj = mcmc(dm=0, dmerr=0, z=self.logz, model_type=self.modeltype, comp=self.comp)
        self.gen_mc_obj.verbose = False
        self.gen_mc_obj.bounds['tau_V'] = [1e-4, 3]

        self.bounds = {
            'luminosity': [3.0, 6.0],
            'temperature': [2600.0, 5000.0], 
            'tau_V': [1e-4, 12.0],
            'dust_temp': [200.0, 1800.0],
            'Av': [0.0, 5.0],
            'Rv': [2.0, 6.0]
        }

        if self.modeltype=='MARCS15':
            if self.logz!=0.0:
                raise ValueError('15Msun MARCS model is only avaiable at Z=0.0')
            self.bounds['temperature'] = [3300.0, 4500.0]

        self.nrc_filts = np.array(['F070W','F090W','F115W','F140M','F150W', 'F150W2', 'F162M',
                                    'F164N','F182M','F187N','F200W','F210M','F212N','F250M',
                                    'F277W','F300M','F322W2','F323N','F335M','F356W','F360M',
                                    'F405N','F410M','F430M','F444W','F460M','F466N','F470N','F480M'])
        self.wv_all = [float(i.replace('F', '').replace('W2', '').replace('M', '').replace('N', '').replace('W', ''))/100 
                       for i in self.nrc_filts]
        self.model_params = ['temperature', 'dust_temp', 'tau_V', 'luminosity', 'Rv', 'Av']

    def sim_training_set(self, ntrain=3e5):

        self.ntrain = ntrain

        sim = np.zeros((ntrain, len(self.model_params) + len(self.nrc_filts)))
        sample_params = self.gen_mc_obj.get_init_pos(ntrain)

        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](p_).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            sim[i, :len(self.model_params)] = p_
            sim[i, len(self.model_params):] = model_mag

        cols = self.model_params + list(self.nrc_filts)
        df = pd.DataFrame(sim, columns=cols)
        fname = os.path.join(self.procdir, f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        df.to_csv(fname, index=False)

    def sim_noise(self, train, rsgcat, sedfit):
        train_err = np.zeros((len(train), len(sedfit.cols['errcols'])))
        for i, col in enumerate(sedfit.cols['errcols']):
            err = rsgcat[col].values
            mask = np.isnan(err) | (err > 0.5)
            err = err[~mask]
            kde = gaussian_kde(err)
            resamp_errs = kde.resample(len(train))
            resamp_errs[resamp_errs < 0.01] = 0.01
            train_err[:, i] = resamp_errs

        return train_err

    def train_sbi(self, training_set_path, rsgcat, sedfit, plot=False):
        train = pd.read_csv(training_set_path)
        train['temperature'] = np.log10(train['temperature'])
        train['dust_temp'] = np.log10(train['dust_temp'])

        mags = train[train.columns[6:]]
        params = train[train.columns[:6]]

        x_train = np.array(params, dtype=float)
        y_mags = mags[sedfit.cols['flts']]
        train_err = self.sim_noise(train, rsgcat, sedfit)
        y_err = pd.DataFrame(train_err, columns=sedfit.cols['errcols'])
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
        pickle.dump(anpe._summary, open('data/sbi/anpe_3.p', 'wb'))

        if plot:
            try:
                _ = plot_summary(anpe, tags=["training_loss", "validation_loss"], figsize=(10, 2),)
            except:
                print('Failed to plot')

        return savepath

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    sbigen = sbifit(procdir=args.procdir, 
                    comp=args.comp, 
                    modeltype=args.modeltype, 
                    z=args.z,
                    ntrain=args.ntrain) 
    sbigen.sim_training_set()