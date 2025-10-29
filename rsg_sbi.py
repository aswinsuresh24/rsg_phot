import warnings
warnings.simplefilter('ignore')
import numpy as np
import os, glob
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import pandas as pd
import itertools
from tqdm import tqdm
from mcmc import mcmc
from multiprocessing import Pool
import argparse
import logging
import multiprocessing_logging
from datetime import datetime
import torch
from sbi import utils as sbi_utils
from sbi import inference

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

    return parser

class sbifit(object):
    def __init__(self, procdir, comp='sil', modeltype='MARCS', z=0.0):

        self.procdir = procdir
        os.makedirs(self.procdir, exist_ok=True)

        self.logz = z
        self.comp = comp
        self.modeltype = modeltype

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

    def sim_training_set(self, nsamp):
        sim = np.zeros((nsamp, len(self.model_params) + len(self.nrc_filts)))
        sample_params = self.gen_mc_obj.get_init_pos(nsamp)

        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](p_).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            sim[i, :len(self.model_params)] = p_
            sim[i, len(self.model_params):] = model_mag

        cols = self.model_params + list(self.nrc_filts)
        df = pd.DataFrame(sim, columns=cols)
        fname = os.path.join(self.procdir, f"sim_{self.modeltype}_{self.comp}_Z{self.logz}.csv")
        df.to_csv(fname, index=False)