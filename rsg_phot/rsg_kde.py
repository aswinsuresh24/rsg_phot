import numpy as np
import os
import sys
import astropy.units as u
import traceback
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly import express as px
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from numpy.polynomial import polynomial as P
from scipy.stats import median_abs_deviation
from pathlib import Path
import json
from multiprocessing import Pool
from contextlib import contextmanager
import pandas as pd
from tqdm import tqdm

from rsg_phot.mc_parallel import rsg_dataloader, mcmcfit
from rsg_phot.rsg_sbi import sbifit

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

ALL_CONFIGS = {
    'ngc5236': {'rsgcat': Path('../data/dolphot/ngc5236/ngc5236_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5236',
                              'procdir':Path('../data/dolphot/ngc5236'), 'photfile_path':None,
                              'dm':28.46, 'dmerr':0.05, 'z':-0.25, 'trgb':('F090W', 24.52),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False,},
                'model': "957a9f22",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5236/ngc5236_sbi_cat.csv'),
                'rsg_color': 0.4, 'dm': 0.4,
                'z_true': 0.74, 'z_err': 0.01, 
                'sfr': 0.62, 'sfr_err': 0.20, 'A': 306.58,
                'logm': 10.41, 'logm_err': 0.1},
    'ngc5194': {'rsgcat': Path('../data/dolphot/ngc5194/ngc5194_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5194',
                              'procdir':Path('../data/dolphot/ngc5194'), 'photfile_path':None,
                              'dm':29.67, 'dmerr':0.02, 'z':0.0, 'trgb':('F200W', 24.2),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts': ['F090W', 'F410M', 'F430M'],},
                'model': "fb3481bf",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5194/ngc5194_sbi_cat.csv'),
                'rsg_color': 0.4, 'dm': 0.5,
                'z_true': 0.93, 'z_err': 0.21,
                'sfr': 0.65, 'sfr_err': 0.20, 'A': 490.39,
                'logm': 10.73, 'logm_err': 0.1},
    'ngc4258': {'rsgcat': Path('../data/dolphot/ngc4258/ngc4258_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4258',
                              'procdir':Path('../data/dolphot/ngc4258'), 'photfile_path':None,
                              'dm':29.397, 'dmerr':0.03, 'z':-0.25, 'trgb':('F090W', 25.055),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False},
                'model': "ef220e94",
                'f1': 'F115W', 'f2': 'F210M',
                'sbicat_path': Path('../data/dolphot/ngc4258/ngc4258_sbi_cat.csv'),
                'rsg_color': 0.2, 'dm': 0.5,
                'z_true': 0.63, 'z_err': 0.13,
                'sfr': -0.03, 'sfr_err': 0.20, 'A': 1205.99,
                'logm': 10.67, 'logm_err': 0.1},
    'ngc628': {'rsgcat': Path('../data/dolphot/ngc628/ngc628_sil_rsgcat.csv'),
               'load_args': {'gal':'ngc628',
                             'procdir':Path('../data/dolphot/ngc628'), 'photfile_path':None,
                             'dm':30.04, 'dmerr':0.125, 'z':-0.25, 'trgb':('F090W', 26.13),
                             'modeltype':'MARCS', 'comp':'sil',
                             'keep_narrow':False, 'ignore_filts':['F090W', 'F140M', 'F182M', 'F410M', 'F430M', 'F480M'],},
               'model': "89285257",
               'f1': 'F115W', 'f2': 'F200W',
               'sbicat_path': Path('../data/dolphot/ngc628/ngc628_sbi_cat.csv'),
               'rsg_color': 0.4, 'dm': 0.5,
               'z_true': 0.62, 'z_err': 0.01,
               'sfr': 0.23, 'sfr_err': 0.20, 'A': 600.54,
               'logm': 10.24, 'logm_err': 0.1}, 
    'ngc5643': {'rsgcat': Path('../data/dolphot/ngc5643/ngc5643_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5643',
                              'procdir':Path('../data/dolphot/ngc5643'), 'photfile_path':None,
                              'dm':30.57, 'dmerr':0.06, 'z':-0.25, 'trgb':('F090W', 26.20),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts':['F300M'],},
                'model': "be8ad86d",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5643/ngc5643_sbi_cat.csv'),
                'rsg_color': 0.4, 'dm': 0.5,
                'z_true': 0.62, 'z_err': 0.01,
                'sfr': 0.33, 'sfr_err': 0.20, 'A': 295.49,
                'logm': 10.06, 'logm_err': 0.1}, 
    'ngc7320': {'rsgcat': Path('../data/dolphot/ngc7320/ngc7320_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc7320',
                              'procdir':Path('../data/dolphot/ngc7320'), 'photfile_path':None,
                              'dm':30.57, 'dmerr':0.5, 'z':-0.25, 'trgb':('F150W', 27.0),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False,},
                'model': "c74a700a",
                'f1': 'F090W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc7320/ngc7320_sbi_cat.csv'),
                'rsg_color': 0.9, 'dm': 0.5,
                'z_true': 0.49, 'z_err': 0.11,
                'sfr': -0.94, 'sfr_err': 0.2, 'A': 19.52,
                'logm': 9.24, 'logm_err': 0.1},
    'ngc1367': {'rsgcat': Path('../data/dolphot/ngc1367/ngc1367_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc1367',
                              'procdir':Path('../data/dolphot/ngc1367'), 'photfile_path':None,
                              'dm':30.40, 'dmerr':0.07, 'z':0.00, 'trgb':('F090W', 29.13),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts':['F300M'],},
                'model': "27370b04",
                'f1': 'F150W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc1367/ngc1367_sbi_cat.csv'),
                'rsg_color': -0.06, 'dm': 0.5,
                'z_true': 0.78, 'z_err': 0.15,
                'sfr': -0.37, 'sfr_err': 0.20, 'A': 96.06,
                'logm': 9.51, 'logm_err': 0.10}, # minweight 0.1
    'ngc1365': {'rsgcat': Path('../data/dolphot/ngc1365/ngc1365_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc1365',
                              'procdir':Path('../data/dolphot/ngc1365'), 'photfile_path':None,
                              'dm':31.29, 'dmerr':0.065, 'z':-0.25, 'trgb':('F090W', 27.34),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False,},
                'model': "05602ed7",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc1365/ngc1365_sbi_cat.csv'),
                'rsg_color': 0.4, 'dm': 0.5,
                'z_true': 0.73, 'z_err': 0.02,
                'sfr': 1.15, 'sfr_err': 0.20, 'A': 2503.47,
                'logm': 10.75, 'logm_err': 0.1},
    'ngc4536': {'rsgcat': Path('../data/dolphot/ngc4536/ngc4536_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4536',
                              'procdir':Path('../data/dolphot/ngc4536'), 'photfile_path':None,
                              'dm':30.99, 'dmerr':0.06, 'z':-0.25, 'trgb':('F090W', 27.01),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts': ['F115W', 'F444W'],},
                'model': "9badc531",
                'f1': 'F150W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc4536/ngc4536_sbi_cat.csv'),
                'rsg_color': 0.0, 'dm': 0.5,
                'z_true': 0.62, 'z_err': 0.13,
                'sfr': 0.47, 'sfr_err': 0.20, 'A': 908.95,
                'logm': 10.19, 'logm_err': 0.1},
    'ngc5457': {'rsgcat': Path('../data/dolphot/ngc5457/ngc5457_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5457',
                              'procdir':Path('../data/dolphot/ngc5457'), 'photfile_path':None,
                              'dm':29.07, 'dmerr':0.05, 'z':-0.25, 'trgb':('F090W', 25.04),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts':['F322W2'],},
                'model': "c2583613",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5457/ngc5457_sbi_cat.csv'),
                'rsg_color': 0.4, 'dm': 0.5,
                'z_true': 0.55, 'z_err': 0.01,
                'sfr': 0.54, 'sfr_err': 0.20, 'A': 1350.65,
                'logm': 10.39, 'logm_err': 0.1},
    'ngc4449': {'rsgcat': os.path.join(os.pardir, 'data/dolphot/ngc4449/ngc4449_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4449', 'procdir':os.path.join(os.pardir, 'data/dolphot/ngc4449'), 'photfile_path':None,
                              'dm':28.02, 'dmerr':0.32, 'z':-0.5, 'trgb':('F090W', 25.11),
                              'modeltype':'MARCS', 'comp':'sil', 'keep_narrow':False,},
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc4449/ngc4449_sbi_cat.csv'),
                'model': "2ed03571",
                'rsg_color': 0.3, 'dm': 0.3,
                'z_true': 0.34, 'z_err': 0.03,
                'sfr': -0.37, 'sfr_err': 0.20, 'A': 29.55,
                'logm':9.03, 'logm_err':0.10}, # single seed selection iteration (skip trend removal) 
    'ngc4485': {'rsgcat': os.path.join(os.pardir, 'data/dolphot/ngc4485/ngc4485_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4485', 'procdir':os.path.join(os.pardir, 'data/dolphot/ngc4485'), 'photfile_path':None,
                              'dm':29.67, 'dmerr':0.1, 'z':-0.5, 'trgb':('F090W', 25.62),
                              'modeltype':'MARCS', 'comp':'sil', 'keep_narrow':False,},
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc4485/ngc4485_sbi_cat.csv'),
                'model': "41d94035",
                'rsg_color': 0.3, 'dm': 0.5,
                'z_true': 0.25, 'z_err': 0.03,
                'sfr': 0.23, 'sfr_err': 0.20, 'A': 224.15,
                'logm':9.73, 'logm_err':0.10},
    'ngc4548': {'rsgcat': os.path.join(os.pardir, 'data/dolphot/ngc4548/ngc4548_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4548', 'procdir':os.path.join(os.pardir, 'data/dolphot/ngc4548'), 'photfile_path':None,
                              'dm':30.08, 'dmerr':0.05, 'z':0.0, 'trgb':('F090W', 27.00),
                              'modeltype':'MARCS', 'comp':'sil', 'keep_narrow':False},
                'model': "175bb2bb",
                'f1': 'F150W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc4548/ngc4548_sbi_cat.csv'),
                'rsg_color': 0.0, 'dm': 0.5,
                'z_true': 1.45, 'z_err': 0.5,
                'sfr': -0.28, 'sfr_err': 0.20, 'A': 206.32,
                'logm': 10.65, 'logm_err': 0.1}, 
    'ngc4038': {'rsgcat': os.path.join(os.pardir, 'data/dolphot/ngc4038/ngc4038_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4038', 'procdir':os.path.join(os.pardir, 'data/dolphot/ngc4038'), 'photfile_path':None,
                              'dm':31.053, 'dmerr':0.05, 'z':0.0, 'trgb':('F090W', 27.72),
                              'modeltype':'MARCS', 'comp':'sil', 'keep_narrow':False,},
                'model': "2fa77c2e",
                'f1': 'F115W', 'f2': 'F150W',
                'sbicat_path': Path('../data/dolphot/ngc4038/ngc4038_sbi_cat.csv'),
                'rsg_color': 0.0, 'dm': 0.5,
                'z_true': 1.02, 'z_err': 0.05,
                'sfr': 1.03, 'sfr_err': 0.20, 'A': 318.96,
                'logm': 10.54, 'logm_err': 0.1}, 
    'ngc3034': {'rsgcat': os.path.join(os.pardir, 'data/dolphot/ngc3034/ngc3034_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc3034', 'procdir':os.path.join(os.pardir, 'data/dolphot/ngc3034'), 'photfile_path':None,
                              'dm':27.95, 'dmerr':0.21, 'z':0.0, 'trgb':('F090W', 23.95),
                              'modeltype':'MARCS', 'comp':'sil', 'keep_narrow':False,},
                'model': "7fb4df30",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc3034/ngc3034_sbi_cat.csv'),
                'rsg_color': 0.35, 'dm': 0.5,
                'z_true': 1.00, 'z_err': 0.3,
                'sfr': 0.85, 'sfr_err': 0.20, 'A': 115.49,
                'logm': 10.01, 'logm_err': 0.1}, 
}

def load_gal(gal:str):
    config = ALL_CONFIGS[gal]
    rsgcat = pd.read_csv(config['rsgcat'])
    if any(rsgcat['lum_chisq'] > 100):
        rsgcat['lum_chisq'] = np.log10(rsgcat['lum_chisq'])
    config['load_args']['rsgcat'] = rsgcat
    rsgloader = rsg_dataloader(**config['load_args'])

    sedfit = sbifit(rsgloader)
    sedfit_mc = mcmcfit(rsgloader, ncores=1, verbose=True)
        
    config_id = config['model']
    config_path = sedfit.procdir / f'npe_{config_id}.json'
    if not config_path.exists():
        config_path_alt = sedfit.procdir.parent / 'sbi_opt' / f'npe_{config_id}.json'
        if config_path_alt.exists():
            config_path = config_path_alt
            sedfit.procdir = sedfit.procdir.parent / 'sbi_opt'
        else:
            raise FileNotFoundError(f'No config file found for config_id {config_id} in {sedfit.procdir} or {sedfit.procdir.parent / "sbi_opt"}')
    with open(config_path) as f:
        sbi_config = json.load(f)

    hatp_x_y = sedfit.baseline_sbi_model(sbi_config=sbi_config)

    return rsgloader, sedfit, sedfit_mc, hatp_x_y

Z_MAPPING = {-0.5:[], -0.25:[], 0.0:[]}
for key, val in ALL_CONFIGS.items():
    if key in ['ngc4038', 'ngc3034']:
        continue  # skip ngc4038 and ngc3034 as they are outliers in temperature distribution and not processed yet, respectively
    Z_MAPPING[val['load_args']['z']] = Z_MAPPING.get(val['load_args']['z'], []) + [key]
    
def lin(x, m, c): 
    return m*x+c

class AdaptiveKDE:
    """
    3-class KDE classifier with pseudo-adaptive bandwidth
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.kdes = {}
    
    def fit(self, X, y, class_names=['RSG', 'AGB', 'Blue']):
        """Fit weighted KDEs (pseudo-adaptive)"""
        X_scaled = self.scaler.fit_transform(X)
        
        for class_idx, class_name in enumerate(class_names):
            X_class = X_scaled[y == class_idx]
            n = len(X_class)
            
            # Pilot KDE
            pilot_kde = gaussian_kde(X_class.T, bw_method='scott')
            pilot_densities = pilot_kde(X_class.T)
            
            # Abramson weights
            g = np.exp(np.mean(np.log(pilot_densities + 1e-10)))
            weights = np.power(pilot_densities / g, -self.alpha)
            weights = weights / weights.sum() * n
            
            # Weighted KDE
            kde = gaussian_kde(X_class.T, bw_method='scott', weights=weights)
            
            self.kdes[class_idx] = kde
                    
        return self
    
    def predict_proba(self, X):
        """Predict on arbitrary points"""
        X_scaled = self.scaler.transform(X)
        n_samples = len(X)
        n_classes = len(self.kdes)
        
        likelihoods = np.zeros((n_samples, n_classes))
        for class_idx, kde in self.kdes.items():
            likelihoods[:, class_idx] = kde(X_scaled.T)
            likelihoods[:, class_idx] /= np.max(likelihoods[:, class_idx])
        
        # Normalize
        probs = likelihoods / likelihoods.sum(axis=1, keepdims=True)
        return probs


class star_class(object):
    def __init__(self, gal:str, sbicat_path:Path, f1:str, f2:str):
        self.gal = gal
        if not sbicat_path.exists():
            raise FileNotFoundError(f"Catalog {sbicat_path} not found")
        self.sbicat_path = sbicat_path  
        self.df = pd.read_csv(self.sbicat_path)
        disc_ = (self.df['use_res'].isna()) | (self.df['use_res'] == 0) | (self.df['temperature_median'].isna()) | \
                (self.df['luminosity_median'].isna()) | (self.df['tau_V_median'].isna())
        self.df = self.df[~disc_]
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
        if (f'{f1}_mag' not in self.df.columns):
            raise ValueError(f'Filter {f1} not found')
        if (f'{f2}_mag' not in self.df.columns):
            raise ValueError(f'Filter {f2} not found')
        self.f1 = f1
        self.f2 = f2
        self.cmd_mask = (self.df[f'{self.f1}_mag'] > 10) & (self.df[f'{self.f1}_mag'] < 32)  & \
                        (self.df[f'{self.f2}_mag'] > 10) & (self.df[f'{self.f2}_mag'] < 32) #& \
                        # (self.df[f'{self.f2}_mag'] > 21) & (self.df[f'{self.f2}_mag'] < 26)
        self.rsgloader, self.sedfit, self.sedfit_mc, self.hatp_x_y = self.load_gal(self.gal)
        self.logger = self.rsgloader.logger

    def load_gal(self, gal:str):
        config = ALL_CONFIGS[gal]
        rsgcat = pd.read_csv(config['rsgcat'])
        if any(rsgcat['lum_chisq'] > 100):
            rsgcat['lum_chisq'] = np.log10(rsgcat['lum_chisq'])
        config['load_args']['rsgcat'] = rsgcat
        rsgloader = rsg_dataloader(**config['load_args'])

        sedfit = sbifit(rsgloader)
        sedfit_mc = mcmcfit(rsgloader, ncores=1, verbose=True)
            
        config_id = config['model']
        config_path = sedfit.procdir / f'npe_{config_id}.json'
        if not config_path.exists():
            config_path_alt = sedfit.procdir.parent / 'sbi_opt' / f'npe_{config_id}.json'
            if config_path_alt.exists():
                config_path = config_path_alt
                sedfit.procdir = sedfit.procdir.parent / 'sbi_opt'
            else:
                raise FileNotFoundError(f'No config file found for config_id {config_id} in {sedfit.procdir} or {sedfit.procdir.parent / "sbi_opt"}')
        with open(config_path) as f:
            sbi_config = json.load(f)

        hatp_x_y = sedfit.baseline_sbi_model(sbi_config=sbi_config)

        return rsgloader, sedfit, sedfit_mc, hatp_x_y

    def plot_cmd(self):
        rsg_cl = self.df[self.cmd_mask][f'{self.f1}_mag'] - self.df[self.cmd_mask][f'{self.f2}_mag']
        rsg_m = self.df[self.cmd_mask][f'{self.f2}_mag']

        fig, ax = plt.subplots()
        ax.hexbin(rsg_cl, rsg_m, cmap = 'inferno', bins=200, norm=mpl.colors.LogNorm())
        ax.invert_yaxis()
        ax.grid(ls='--', alpha=0.3)

        ax.set_xlabel(f'{self.f1}-{self.f2}')
        ax.set_ylabel(self.f2)
        ax.set_title(f'{self.gal.upper()}')

    def plot_lum_cmd(self):
        rsg_cl = self.df[self.cmd_mask][f'{self.f1}_mag'] - self.df[self.cmd_mask][f'{self.f2}_mag']
        rsg_m = self.df[self.cmd_mask][f'{self.f2}_mag']
        lums = self.df[self.cmd_mask]['luminosity_median']
        lmask = (lums > 4.97) & (lums < 5.03) & (rsg_cl > np.median(rsg_cl))
        f200w_mean, f200w_sig = np.mean(rsg_m[lmask]), np.std(rsg_m[lmask], ddof=1)
        self.logger.info(f"F200W mag for logL ~ 5: mean={f200w_mean:.2f}, std={f200w_sig:.2f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(rsg_cl, rsg_m, c=self.df[self.cmd_mask]['luminosity_median'], 
                    cmap = 'Paired', norm=mpl.colors.LogNorm())
        plt.fill_between(np.linspace(np.min(rsg_cl), np.max(rsg_cl), 100), f200w_mean-f200w_sig, f200w_mean+f200w_sig, 
                         color='orange', alpha=0.3, label='logL ~ 5')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.grid(ls='--', alpha=0.3)
        plt.legend()

        plt.xlabel(f'{self.f1}-{self.f2}')
        plt.ylabel(self.f2)
        plt.title(f'{self.gal.upper()} CMD')
    
    def select_seed_sample(self, rsg_color, dm=0.5, prob_threshold = (0.75, 0.75, 0.5), minweight=0.02, magbins=None, plot=False):
        seed_gen = rsg_seed(self.gal, self.sbicat_path, self.f1, self.f2, rsg_color, dm=dm)
        if magbins is not None:
            seed_gen.mag_bins = magbins
        seed_df, validated, trend_fn, results = seed_gen.run_seed_selection(plot=plot, prob_threshold=prob_threshold, min_weight=minweight, n_max=10)

        for _, pl in zip(range(2), [False, True]):
            seed_gen.colors -= np.vectorize(trend_fn)(seed_gen.mags)
            seed_gen.rsg_color = 0.0
            seed_df, validated, trend_fn, results = seed_gen.run_seed_selection(plot=pl, prob_threshold=prob_threshold, min_weight=minweight, n_max=10)
            
        return seed_df
    
    def plot_seed_cmd(self, seed_df):
        self.plot_cmd()
        plt.scatter(seed_df[f'{self.f1}_mag'] - seed_df[f'{self.f2}_mag'], seed_df[f'{self.f2}_mag'], 
                    c=seed_df['class'].map({'RSG': 'coral', 'AGB': 'none', 'Blue': 'cyan'}), s=2)
        plt.show()

    def plot_seed_3d(self, seed_df):
        fig = px.scatter_3d(seed_df, x='temperature_median', y='luminosity_median', z='tau_V_median', color='class', 
                            hover_data=['temperature_median', 'luminosity_median', 'tau_V_median'], symbol='class', 
                            color_discrete_map={'RSG':'royalblue', 'AGB':'coral', 'Blue':'magenta'},
                            title=f'{self.gal.upper()} Seed Sample')
        #update size of points and opacity
        fig.update_traces(marker=dict(size=5, opacity=0.3))
        fig.update_layout(scene = dict(
                            xaxis_title='Temperature (K)',
                            yaxis_title='Luminosity (Lsun)',
                            zaxis_title='Tau'),
                            legend_title='Class')
        fig.show()

        fig.write_html(f'../plots/{self.gal}_seed_sample_3d.html')

    def kde_class(self, seed_df=None, assign_class=True, alpha=0.5):
        X = seed_df[['temperature_median', 'luminosity_median', 'tau_V_median']].values
        y = seed_df['class'].map({'RSG':0, 'AGB':1, 'Blue':2}).values
        self.kde = AdaptiveKDE(alpha=alpha).fit(X, y)

        X_pred = self.df[['temperature_median', 'luminosity_median', 'tau_V_median']].values
        if assign_class:
            self.df[['p_rsg', 'p_agb', 'p_blue']] = self.kde.predict_proba(X_pred)
            kde_class = np.argmax(self.df[['p_rsg', 'p_agb', 'p_blue']].values, axis=1)
            self.df['class_label'] = np.where(kde_class == 0, 'RSG', np.where(kde_class == 1, 'AGB', 'Blue'))
        else:
            probs = self.kde.predict_proba(X_pred)
            return probs

        return self.df

    def plot_kde_class(self):
        df_ = self.df.copy()
        df_.loc[(df_['p_rsg'] > 0.3) & (df_['p_rsg'] < 0.6), 'class_label'] = 'Uncertain'
        fig = px.scatter_3d(df_, x='temperature_median', y='luminosity_median', z='tau_V_median', color='class_label', 
                            hover_data=['temperature_median', 'luminosity_median', 'tau_V_median'], symbol='class_label', 
                            color_discrete_map={'RSG':'royalblue', 'AGB':'coral', 'Blue':'magenta', 'Uncertain':'gray'},
                            title=f'{self.gal.upper()} KDE Classification')
        #update size of points and opacity
        fig.update_traces(marker=dict(size=3, opacity=0.3))
        fig.update_layout(scene = dict(
                            xaxis_title='Temperature (K)',
                            yaxis_title='Luminosity (Lsun)',
                            zaxis_title='Tau'),
                            legend_title='Class')
        fig.show()

        #save figure to html
        fig.write_html(f'../plots/{self.gal}_kde_classification_3d.html')

    def plot_kde_class_cmd(self, plot_class=None):
        self.plot_cmd()
        class_colors = {'RSG':'royalblue', 'AGB':'coral', 'Blue':'magenta'}
        if plot_class:
            class_colors = {plot_class: class_colors[plot_class]}
        for class_label, color in class_colors.items():
            subset = self.df[self.cmd_mask & (self.df['class_label'] == class_label)]
            plt.scatter(subset[f'{self.f1}_mag'] - subset[f'{self.f2}_mag'], subset[f'{self.f2}_mag'], 
                        color=color, s=1, label=class_label)

    def plot_rsg_cmd(self, pcut=0.7):
        self.logger.info(f"{(self.df['p_rsg'] > pcut).sum()} RSGs above pcut {pcut}")
        self.plot_cmd()
        rsg_subset = self.df[self.cmd_mask & (self.df['p_rsg'] > pcut)]
        plt.scatter(rsg_subset[f'{self.f1}_mag'] - rsg_subset[f'{self.f2}_mag'], rsg_subset[f'{self.f2}_mag'], 
                    c=rsg_subset['p_rsg'], s=5, cmap='inferno')
        plt.colorbar(label='P(RSG)')
        

class rsg_seed(star_class):
    def __init__(self, gal:str, sbicat_path:Path, f1:str, f2:str, rsg_color:float, dm=0.5, deredden=False):
        super().__init__(gal, sbicat_path, f1, f2)

        if deredden:
            e_bv = self.df['Av_median'] / 3.1
            d_ebv = self.df['Av_eup'] / 3.1

            from dust_extinction.parameter_averages import CCM89, G23
            ext = G23(Rv=3.1)
            for flt_ in ['F115W', 'F200W']:
                wv = float(flt_[1:4])/100*u.um
                a_lambda = ext(wv)
                A_lambda = a_lambda * e_bv * 3.1
                dA_lambda = a_lambda * d_ebv * 3.1
                self.df[flt_ + '_mag_dered'] = self.df[flt_ + '_mag'] - A_lambda
                self.df[flt_ + '_mag_dered_err'] = np.sqrt(self.df[flt_ + '_err']**2 + dA_lambda**2)
        # self.df['F115W_mag'] = self.df['F115W_mag_dered']
        # self.df['F200W_mag'] = self.df['F200W_mag_dered']

        self.rsg_color = rsg_color
        self.lcut = self.get_lcut()
        self.logger.info(f"Initial RSG color cut: {self.rsg_color:.2f}, luminosity cut: {self.lcut:.2f}")

        self.colors, self.mags = self.df[self.cmd_mask][f'{self.f1}_mag'] - self.df[self.cmd_mask][f'{self.f2}_mag'], self.df[self.cmd_mask][f'{self.f2}_mag']
        self.colors, self.mags = self.colors.values, self.mags.values
        magmin, magmax = np.percentile(self.mags, [0.1, 99.9])
        self.mag_bins = np.arange(magmin, magmax+dm, dm) 
        self.magbins = np.clip(self.mag_bins, magmin, magmax)

    def get_lcut(self):
        rsg_cl = self.df[self.cmd_mask][f'{self.f1}_mag'] - self.df[self.cmd_mask][f'{self.f2}_mag']
        rsg_m = self.df[self.cmd_mask][f'{self.f2}_mag']
        lums = self.df[self.cmd_mask]['luminosity_median']
        lmask = (lums > 4.97) & (lums < 5.03) & (rsg_cl > np.median(rsg_cl))
        f200w_mean, f200w_sig = np.mean(rsg_m[lmask]), np.std(rsg_m[lmask], ddof=1)
        return f200w_mean + f200w_sig
    
    def fit_gmm_bic(self, colors, n_min=3, n_max=10, n_init=5, random_state=42):
        """
        Fit GMMs with n_min..n_max components to a 1-D colour array.
        Returns the model with the lowest BIC, together with the BIC curve.

        Parameters
        ----------
        colors        : (N,) array of colour values for stars in one magnitude slice
        n_min, n_max  : range of component counts to try
        n_init        : number of random initialisations per fit (guards against
                        local minima)
        random_state  : RNG seed for reproducibility

        Returns
        -------
        best_gmm : fitted GaussianMixture with the lowest BIC
        bics     : list of BIC values for n_min..n_max
        n_range  : list of n values tried
        """
        bics, models = [], []
        n_range = list(range(n_min, n_max + 1))

        for n in n_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                n_init=n_init,
                random_state=random_state,
            )
            gmm.fit(colors.reshape(-1, 1))
            bics.append(gmm.bic(colors.reshape(-1, 1)))
            models.append(gmm)

        best_idx = int(np.argmin(bics))
        return models[best_idx], bics, n_range
    
    def slice_and_fit(self, mag, color, mag_bins, n_max=5, min_stars=30, n_init=5):
        """
        Slice the CMD into magnitude bins and fit a BIC-optimal GMM per slice.

        Parameters
        ----------
        mag, color : (N,) arrays
        mag_bins   : bin edges (e.g. np.arange(mag.min(), mag.max(), 0.3))
        n_max      : maximum number of GMM components to try
        min_stars  : slices with fewer stars are skipped
        n_init     : passed to fit_gmm_bic

        Returns
        -------
        results : dict  {(m_lo, m_hi): slice_result | None}
            Each slice_result contains:
                gmm     - fitted GaussianMixture
                bics    - BIC curve
                n_range - n values tried
                n_best  - best fit number of components
                means   - component means sorted blue-red
                sigmas  - component standard deviations (same order)
                weights - component weights (same order)
                colors  - colour values of stars in this slice
                mag_mid - midpoint of the magnitude bin
        """
        results = {}

        for i in range(len(mag_bins) - 1):
            m_lo, m_hi = mag_bins[i], mag_bins[i + 1]
            mask = (mag >= m_lo) & (mag < m_hi)
            c = color[mask]

            if len(c) < min_stars:
                results[(m_lo, m_hi)] = None
                continue

            best_gmm, bics, n_range = self.fit_gmm_bic(c, n_max=n_max, n_init=n_init)

            order   = np.argsort(best_gmm.means_.flatten())
            means   = best_gmm.means_.flatten()[order]
            sigmas  = np.sqrt(best_gmm.covariances_.flatten())[order]
            weights = best_gmm.weights_[order]

            results[(m_lo, m_hi)] = {
                "gmm":     best_gmm,
                "bics":    bics,
                "n_range": n_range,
                "n_best":  best_gmm.n_components,
                "means":   means,
                "sigmas":  sigmas,
                "weights": weights,
                "colors":  c,
                "mag_mid": np.median(mag[mask]),
            }

        return results
        
    def initialize_rsg_chain(self, results, min_weight=0.05, exp_rsg_color=0.5):
        """
        Identify the RSG candidate component in each slice based on the expected
        color of the RSG branch

        Parameters
        ----------
        results    : output of slice_and_fit
        min_weight : components with weight below this are ignored

        Returns
        -------
        candidates : dict  {(m_lo, m_hi): candidate_dict | None}
            Each candidate_dict contains mean, sigma, weight, gmm_idx (sorted),
            is_reddest flag, and mag_mid.
        """
        candidates = {}

        for key, res in results.items():
            if res is None:
                candidates[key] = None
                continue

            # Filter negligible components
            ok = np.array(res["weights"]) >= min_weight
            means   = np.array(res["means"])[ok]
            sigmas  = np.array(res["sigmas"])[ok]
            weights = np.array(res["weights"])[ok]
            # Keep track of original sorted indices so we can map back to GMM
            orig_idx = np.where(ok)[0]

            if len(means) < 2:
                candidates[key] = None
                continue

            rsg_local_idx    = np.argmin(np.abs(means - exp_rsg_color))  # closest to expected RSG color
            rsg_sorted_idx   = int(orig_idx[rsg_local_idx]) # index in sorted order

            candidates[key] = {
                "mean":       float(means[rsg_local_idx]),
                "sigma":      float(sigmas[rsg_local_idx]),
                "weight":     float(weights[rsg_local_idx]),
                "gmm_idx":    rsg_sorted_idx,
                "is_reddest": rsg_sorted_idx == len(res["means"]) - 1,
                "mag_mid":    res["mag_mid"],
                "key":        key,
            }

        return candidates
    
    def fit_and_validate_chain(self, candidates, deg=2, sigma_clip=2.0):
        """
        Fit a smooth polynomial (colour vs magnitude) through the per-slice RSG
        candidates and sigma-clip outliers.  Outlier slices are typically ones
        where the GMM failed (merged blue+RSG or split AGB into RSG region).

        Parameters
        ----------
        candidates : output of initialize_chain_by_gap
        deg        : polynomial degree (1 = linear, 2 = quadratic)
        sigma_clip : rejection threshold in units of MAD-based sigma

        Returns
        -------
        validated : dict with same keys as candidates; each entry gains
                    on_trend  - bool
                    predicted - colour predicted by the trend at this magnitude
                    residual  - observed minus predicted
        trend_fn  : callable  mag -> predicted_colour
        """
        keys  = [k for k, v in candidates.items() if v is not None]
        mags  = np.array([candidates[k]["mag_mid"] for k in keys])
        means = np.array([candidates[k]["mean"]    for k in keys])

        mask = np.ones(len(mags), dtype=bool)

        for _ in range(10):
            if mask.sum() < deg + 2:
                break
            coeffs    = P.polyfit(mags[mask], means[mask], deg=deg)
            predicted = P.polyval(mags, coeffs)
            residuals = means - predicted
            mad       = median_abs_deviation(residuals[mask])
            mask      = np.abs(residuals) < sigma_clip * mad * 1.4826

        coeffs    = P.polyfit(mags[mask], means[mask], deg=deg)
        predicted = P.polyval(mags, coeffs)

        validated = {}
        for i, k in enumerate(keys):
            c = candidates[k].copy()
            c["on_trend"]  = bool(mask[i])
            c["predicted"] = float(predicted[i])
            c["residual"]  = float(means[i] - predicted[i])
            validated[k]   = c

        # Fill entries that were None in candidates
        for k, v in candidates.items():
            if k not in validated:
                validated[k] = None

        trend_fn = lambda m: float(P.polyval(np.asarray(m, dtype=float), coeffs))

        return validated, trend_fn
    

    def recover_merged_slices(self, validated, results, trend_fn,
                              max_residual=0.12, min_weight=0.02):
        """
        For slices where the RSG candidate deviates from the smooth colour trend
        (likely because the GMM merged the blue/foreground population with RSGs,
        pulling the component mean blueward), attempt to substitute the component
        whose mean is closest to the trend prediction.

        If no suitable component exists, fall back to the trend colour itself
        (flagged as interpolated=True); those stars will receive low posterior
        probabilities and will be down-weighted at the extraction step.

        Parameters
        ----------
        validated     : output of fit_and_validate_chain
        results       : output of slice_and_fit
        trend_fn      : callable from fit_and_validate_chain
        max_residual  : maximum allowed |mean - trend| for a recovery candidate
        min_weight    : minimum component weight to consider

        Returns
        -------
        validated : updated in place and returned
        """
        for key, cand in validated.items():
            if cand is None or cand["on_trend"]:
                continue

            res            = results[key]
            expected_color = trend_fn(res["mag_mid"])
            means          = np.array(res["means"])
            sigmas         = np.array(res["sigmas"])
            weights        = np.array(res["weights"])

            dists = np.abs(means - expected_color)
            best  = int(np.argmin(dists))

            if dists[best] <= max_residual and weights[best] >= min_weight:
                validated[key] = {
                    "mean":        float(means[best]),
                    "sigma":       float(sigmas[best]),
                    "weight":      float(weights[best]),
                    "gmm_idx":     best,
                    "is_reddest":  best == len(means) - 1,
                    "mag_mid":     res["mag_mid"],
                    "key":         key,
                    "on_trend":    True,
                    "predicted":   expected_color,
                    "residual":    float(means[best] - expected_color),
                    "recovered":   True,
                }
            else:
                # No good component — mark as interpolated
                validated[key]["interpolated"]  = True
                validated[key]["mean"]          = expected_color
                validated[key]["on_trend"]      = True   # treat as usable

        return validated
    
    def extract_seeds(self, validated, results, mag, color, lcut, prob_threshold=(0.75, 0.75, 0.5)):
        """
        Assign stars to RSG, AGB, or blue-star populations in each magnitude slice
        using GMM posterior probabilities.
    
        The RSG component is identified by the validated chain.  All components
        redder than the RSG component (higher sorted index) are pooled as AGBs;
        all components bluer (lower sorted index) are pooled as blue/foreground
        stars.  Within each population a star is included when the summed
        posterior probability across its assigned components meets prob_threshold.
    
        Parameters
        ----------
        validated       : output of recover_merged_slices
        results         : output of slice_and_fit
        mag, color      : original full arrays
        prob_threshold  : minimum summed posterior probability to include a star
                        in any population seed
    
        Returns
        -------
        rsg_indices  : (M,) int array   — indices of RSG seed stars
        agb_indices  : (M,) int array   — indices of AGB seed stars
        blue_indices : (M,) int array   — indices of blue/foreground seed stars
        rsg_probs    : (M,) float array — RSG posterior probability per RSG star
        agb_probs    : (M,) float array — summed AGB posterior per AGB star
        blue_probs   : (M,) float array — summed blue posterior per blue star
        """
        rsg_indices,  rsg_probs  = [], []
        agb_indices,  agb_probs  = [], []
        blue_indices, blue_probs = [], []
    
        for key, cand in validated.items():
            if cand is None:
                continue
    
            m_lo, m_hi = key
            slice_mask  = np.where((mag >= m_lo) & (mag < m_hi))[0]
            c           = color[slice_mask]
            m_          = mag[slice_mask]
            rsg_mean, rsg_sig = cand["mean"], cand["sigma"]
    
            if len(c) == 0:
                continue
    
            res   = results[key]
            gmm   = res["gmm"]
            n_comp = gmm.n_components
    
            # sorted order: index 0 = bluest component, index n-1 = reddest
            order            = np.argsort(gmm.means_.flatten())
            rsg_sorted_idx   = cand["gmm_idx"]          # position in sorted order
            rsg_internal_idx = int(order[rsg_sorted_idx])
    
            # Sorted indices for AGB (redder) and blue (bluer) components
            agb_sorted_idxs  = list(range(rsg_sorted_idx + 1, n_comp))
            if rsg_sorted_idx == 0: 
                blue_comp = 1
            else:
                blue_comp = rsg_sorted_idx
            blue_sorted_idxs = list(range(0, blue_comp))
    
            # Map sorted indices - GMM internal indices
            agb_internal_idxs  = [int(order[i]) for i in agb_sorted_idxs]
            blue_internal_idxs = [int(order[i]) for i in blue_sorted_idxs]
    
            posteriors = gmm.predict_proba(c.reshape(-1, 1))  # shape (N_slice, n_comp)
    
            # RSG: single component
            rsg_post = posteriors[:, rsg_internal_idx]
            mask_rsg = rsg_post >= prob_threshold[1]
            rsg_indices.extend(slice_mask[mask_rsg].tolist())
            rsg_probs.extend(rsg_post[mask_rsg].tolist())
    
            # AGB: sum posteriors across all redder components
            if agb_internal_idxs:
                agb_post = posteriors[:, agb_internal_idxs].sum(axis=1)
                mask_agb = agb_post >= prob_threshold[2]
                luminous_mask = m_ > lcut
                # Exclude stars already claimed by RSG to avoid double-counting
                mask_agb &= ~mask_rsg
                rsg_in_agb = np.copy(mask_agb)
                mask_agb &= luminous_mask
                rsg_in_agb &= ~luminous_mask
                agb_indices.extend(slice_mask[mask_agb].tolist())
                agb_probs.extend(agb_post[mask_agb].tolist())

                rsg_indices.extend(slice_mask[rsg_in_agb].tolist())
                rsg_probs.extend(agb_post[rsg_in_agb].tolist())
    
            # Blue: sum posteriors across all bluer components
            if blue_internal_idxs:
                blue_post = posteriors[:, blue_internal_idxs].sum(axis=1)
                mask_blue = blue_post >= prob_threshold[0]
                mask_blue &= ~mask_rsg
                if agb_internal_idxs:
                    mask_blue &= ~mask_agb
                
                #stars 2 sigma bluer than rsgs should be included in the blue sample even if they have low blue_post, since GMM can fail to separate them
                blue_color_cut = rsg_mean - 3 * rsg_sig
                missed_blue_mask = (c < blue_color_cut)
                mask_blue |= missed_blue_mask
                blue_indices.extend(slice_mask[mask_blue].tolist())
                blue_probs.extend(blue_post[mask_blue].tolist())

                # remove these stars from the RSG sample if they were included due to high AGB posterior
                rsg_indices = [idx for idx in rsg_indices if idx not in slice_mask[missed_blue_mask]]
                rsg_probs = [prob for idx, prob in zip(rsg_indices, rsg_probs) if idx not in slice_mask[missed_blue_mask]]
            
    
        return (
            np.array(rsg_indices,  dtype=int), np.array(agb_indices,  dtype=int),
            np.array(blue_indices, dtype=int), np.array(rsg_probs),
            np.array(agb_probs),               np.array(blue_probs),
        )
    
    def plot_rsg_chain_on_cmd(self, validated, mag, color, rsg_indices=None, ax=None):
        """
        Overplot the chained RSG component means and sigmas on the CMD.
        Optionally highlight the extracted seed stars.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        else:
            fig = ax.figure

        ax.scatter(color, mag, s=1, c="grey", alpha=0.3, rasterized=True,
                label="All stars")

        if rsg_indices is not None and len(rsg_indices):
            ax.scatter(color[rsg_indices], mag[rsg_indices],
                    s=4, c="tomato", alpha=0.6, rasterized=True,
                    label="RSG seed")

        for cand in validated.values():
            if cand is None:
                continue
            color_flag = ("orange" if cand.get("recovered")
                        else ("purple" if cand.get("interpolated") else "red"))
            ax.errorbar(cand["mean"], cand["mag_mid"],
                        xerr=cand.get("sigma", 0),
                        fmt="o", color=color_flag,
                        markersize=5, elinewidth=1.5, alpha=0.85)

        ax.invert_yaxis()
        ax.set_xlabel(f"{self.f1}-{self.f2}")
        ax.set_ylabel(self.f2)
        ax.set_title("Chained RSG seed")
        if rsg_indices is not None:
            ax.legend(markerscale=4, fontsize=8)
        return fig, ax

    def plot_chain_on_cmd(self, validated, mag, color,
                        rsg_indices=None, agb_indices=None,
                        blue_indices=None, ax=None):
        """
        Overplot the chained RSG component means on the CMD.
        Optionally highlight RSG, AGB, and blue seed stars with distinct colours.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        else:
            fig = ax.figure
    
        ax.scatter(color, mag, s=1, c="grey", alpha=0.3, rasterized=True,
                label="All stars")
    
        if blue_indices is not None and len(blue_indices):
            ax.scatter(color[blue_indices], mag[blue_indices],
                    s=4, c="steelblue", alpha=0.6, rasterized=True,
                    label="Blue seed")
    
        if rsg_indices is not None and len(rsg_indices):
            ax.scatter(color[rsg_indices], mag[rsg_indices],
                    s=4, c="tomato", alpha=0.6, rasterized=True,
                    label="RSG seed")
    
        if agb_indices is not None and len(agb_indices):
            ax.scatter(color[agb_indices], mag[agb_indices],
                    s=4, c="firebrick", alpha=0.6, rasterized=True,
                    label="AGB seed")
    
        # RSG chain component markers
        for cand in validated.values():
            if cand is None:
                continue
            chain_color = ("orange" if cand.get("recovered")
                        else ("purple" if cand.get("interpolated") else "red"))
            ax.errorbar(cand["mean"], cand["mag_mid"],
                        xerr=cand.get("sigma", 0),
                        fmt="o", color=chain_color,
                        markersize=5, elinewidth=1.5, alpha=0.85, zorder=5)
    
        ax.invert_yaxis()
        ax.set_xlabel(f"{self.f1} - {self.f2}")
        ax.set_ylabel(self.f2)
        ax.set_title("Seed populations")
        ax.legend(markerscale=4, fontsize=8)
        return fig, ax


    def plot_component_tracking(self, results, validated=None, ax=None):
        """
        Show all GMM component means vs magnitude, with the validated RSG chain
        highlighted in red.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        else:
            fig = ax.figure

        for key, res in results.items():
            if res is None:
                continue
            m = res["mag_mid"]
            for mean, sigma, w in zip(res["means"], res["sigmas"], res["weights"]):
                ax.scatter(mean, m, s=200 * w, alpha=0.4,
                        c="steelblue", edgecolors="k", linewidths=0.4)

        if validated is not None:
            for cand in validated.values():
                if cand is None:
                    continue
                ax.scatter(cand["mean"], cand["mag_mid"],
                        s=80, c="red", zorder=5, marker="*")

        ax.invert_yaxis()
        ax.set_xlabel(f"{self.f1}-{self.f2}")
        ax.set_ylabel(self.f2)
        ax.set_title("GMM components")
        return fig, ax


    def plot_bic_curves(self, results, n_cols=4):
        """
        Grid of BIC-vs-n curves, one panel per magnitude slice.
        """
        valid = [(k, v) for k, v in results.items() if v is not None]
        n_panels = len(valid)
        n_rows   = int(np.ceil(n_panels / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(3 * n_cols, 2.5 * n_rows),
                                squeeze=False)
        axes_flat = axes.flatten()

        for ax in axes_flat:
            ax.set_visible(False)

        for i, (key, res) in enumerate(valid):
            ax = axes_flat[i]
            ax.set_visible(True)
            ax.plot(res["n_range"], res["bics"], "o-", ms=4)
            best_n = res["n_best"]
            best_bic = res["bics"][res["n_range"].index(best_n)]
            ax.axvline(best_n, color="red", lw=1, ls="--")
            ax.set_title(f"m={res['mag_mid']:.2f}  n={best_n}", fontsize=8)
            ax.set_xlabel("N", fontsize=7)
            ax.set_ylabel("BIC", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.tight_layout()
        return fig
    
    def run_seed_selection(self,
        n_max=7,
        min_stars=30,
        n_init=7,
        min_weight=0.02,
        trend_deg=2,
        sigma_clip=2.0,
        max_residual=0.12,
        prob_threshold=(0.75, 0.75, 0.5),
        plot=True,
        return_trend = False
    ):
        """
        End-to-end RSG seed selection pipeline.

        Parameters
        ----------
        mag, color      : (N,) arrays — magnitude and colour of all sources
        mag_bins        : bin edges for magnitude slicing
        n_max           : maximum GMM components per slice
        min_stars       : skip slices with fewer stars
        n_init          : GMM random restarts
        min_weight      : ignore components below this weight
        trend_deg       : polynomial degree for colour-trend fit
        sigma_clip      : sigma-clipping threshold for trend validation
        max_residual    : max |colour - trend| allowed in recovery step
        prob_threshold  : minimum RSG posterior probability to keep a star
        plot            : if True, produce diagnostic figures
        save_plots      : if True, save figures to disk
        plot_prefix     : filename prefix when save_plots=True

        Returns
        -------
        rsg_indices : (M,) int array   — indices of seed RSG stars
        rsg_probs   : (M,) float array — RSG membership probability per star
        validated   : per-slice chain dict (for inspection / debugging)
        trend_fn    : callable mag → predicted RSG colour
        results     : raw per-slice GMM results
        """
        self.logger.info("Step 1/4  Fitting GMMs per magnitude slice ...")
        results = self.slice_and_fit(self.mags, self.colors, self.mag_bins,
                                n_max=n_max, min_stars=min_stars, n_init=n_init)
        n_fitted = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"          {n_fitted} slices fitted (of {len(results)} total)")

        self.logger.info("Step 2/4  Identifying RSG candidate per slice ...")
        candidates = self.initialize_rsg_chain(results, min_weight=min_weight, exp_rsg_color=self.rsg_color)

        self.logger.info("Step 3/4  Fitting color trend and sigma-clipping outlier slices …")
        validated, trend_fn = self.fit_and_validate_chain(candidates, deg=trend_deg, sigma_clip=sigma_clip)
        n_on    = sum(1 for v in validated.values()
                    if v is not None and v.get("on_trend"))
        n_off   = sum(1 for v in validated.values()
                    if v is not None and not v.get("on_trend"))
        self.logger.info(f"          {n_on} on-trend  |  {n_off} off-trend (attempting recovery)")

        validated = self.recover_merged_slices(
            validated, results, trend_fn,
            max_residual=max_residual, min_weight=min_weight)
        n_rec   = sum(1 for v in validated.values() if v and v.get("recovered"))
        n_interp = sum(1 for v in validated.values() if v and v.get("interpolated"))
        self.logger.info(f"          {n_rec} recovered  |  {n_interp} interpolated from trend")

        self.logger.info("Step 4/4  Extracting RSG seed stars via GMM posteriors …")
        rsg_indices, agb_indices, blue_indices, rsg_probs, agb_probs, blue_probs = \
            self.extract_seeds(validated, results, self.mags, self.colors, self.lcut, prob_threshold=prob_threshold)
        self.logger.info(f"          RSGs : {len(rsg_indices)}  |  "
            f"AGBs : {len(agb_indices)}  |  "
            f"Blue : {len(blue_indices)}  "
            f"(p ≥ {prob_threshold})")

        rsg_seed = self.df[self.cmd_mask].iloc[rsg_indices]
        agb_seed = self.df[self.cmd_mask].iloc[agb_indices]
        blue_seed = self.df[self.cmd_mask].iloc[blue_indices]
        temperature_mask = (rsg_seed['temperature_median'] < 4500) & (rsg_seed['temperature_median'] > 3200)
        luminosity_mask = (rsg_seed['luminosity_median'] < 4.5) & (rsg_seed['tau_V_median'] > 0.5)

        agb_seed = pd.concat([agb_seed, rsg_seed[luminosity_mask]])
        rsg_seed = rsg_seed[temperature_mask & ~luminosity_mask]
        rsg_seed['class'] = 'RSG'
        agb_seed['class'] = 'AGB'
        blue_seed['class'] = 'Blue'
        seed_df = pd.concat([rsg_seed, agb_seed, blue_seed])

        if plot:
            fig1, _ = self.plot_chain_on_cmd(validated, self.mags, self.colors,
                                rsg_indices, agb_indices, blue_indices)
            fig2, _ = self.plot_component_tracking(results, validated)
            # fig3    = self.plot_bic_curves(results)
            plt.show()


        return seed_df, validated, trend_fn, results

class validate_selection():
    def __init__(self, star_class, sedfit):
        self.cl = star_class
        self.logger = self.cl.logger
        self.sf = sedfit
        self.kdf = self.cl.df

    def generate_rsg_truth_sample(self, gal, met, pcut=0.7, samp_err=False,
                                  nsamp=20000, gdf=None):
        self.logger.info(f"Generating RSG truth sample for {gal} at Z={met} with pcut={pcut} and samp_err={samp_err}")
        cdf = pd.read_csv('../data/catalog/combined_cat.csv')
        if gdf is None:
            gdf = pd.read_csv(f'../data/dolphot/{gal}/{gal}_kdecat.csv')
        # subset = cdf[(cdf['galaxy'].isin(Z_MAPPING[met])) & (cdf['p_rsg'] > pcut)]
        subset = gdf[gdf['p_rsg'] > pcut]
        subset = subset.loc[~((subset['luminosity_median'] > 5) & (subset['temperature_median'] > 4700) & (subset['chimin'] < 0.012))]
        X = subset[['temperature_median', 'luminosity_median', 'tau_V_median']].values
        if samp_err:
            sig = (subset[['temperature_elow', 'luminosity_elow', 'tau_V_elow']].values + \
                    subset[['temperature_eup', 'luminosity_eup', 'tau_V_eup']].values)/2
            X = np.random.normal(loc=X, scale=sig, size=(50, X.shape[0], X.shape[1])).reshape(-1, X.shape[1])
            resamp_idx = np.random.choice(range(X.shape[0]), size=10000, replace=False)
            X = X[resamp_idx]

        bics, models = [], []
        n_range = list(range(1, 10 + 1))

        for n in n_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                n_init=3,
                random_state=42,
            )
            gmm.fit(X)
            bics.append(gmm.bic(X))
            models.append(gmm)

        best_idx = int(np.argmin(bics))
        best_gmm = models[best_idx]
        resamp, _ = best_gmm.sample(nsamp)

        # simulate low luminosity RSGs that may be missing from the sample due to selection effects
        min_L = subset['luminosity_median'].min()
        l_global = cdf[cdf['p_rsg'] > 0.7]['luminosity_median'].values
        global_lowl = cdf[(cdf['p_rsg'] > 0.7) & (cdf['luminosity_median'] < min_L)]['luminosity_median'].values
        pdf, bin_ = np.histogram(global_lowl, bins=30, density=True)
        c_df = np.cumsum(pdf * np.diff(bin_))
        global_interp = interp1d(c_df, (bin_[1:] + bin_[:-1])/2, bounds_error=False, fill_value=(0, 1))
        n_missing = int((l_global < min_L).sum() / len(l_global) * nsamp)
        n_missing = max(n_missing, 5)
        u = np.random.uniform(0, 1, size=n_missing)
        lowl_resamp = global_interp(u)
        lowl_resamp[lowl_resamp < min(l_global)] = min(l_global)
        t_resamp = gaussian_kde(subset[subset['luminosity_median'] < min_L + 0.3]['temperature_median'].values).resample(int(n_missing)).flatten()
        tau_resamp = gaussian_kde(subset[subset['luminosity_median'] < min_L + 0.3]['tau_V_median'].values).resample(int(n_missing)).flatten()
        lowl_samp = np.vstack((t_resamp, lowl_resamp, tau_resamp)).T
        self.logger.info(f'Lowest simulated RSG luminosity: {min(lowl_resamp):.2f}')

        resamp = np.vstack((resamp, lowl_samp))
        nsamp += int(n_missing)

        resamp[:, 0] = np.clip(resamp[:, 0], 2600, 5000)
        resamp[:, 1] = np.clip(resamp[:, 1], 3.45, 5.99)
        resamp[:, 2] = np.clip(resamp[:, 2], 1e-3, 8.0)

        tdust_kde = gaussian_kde(subset['dust_temp_median'].values)
        av_kde = gaussian_kde(subset['Av_median'].values)
        rv_kde = gaussian_kde(subset['Rv_median'].values)
        tdust_resamp = np.clip(tdust_kde.resample(nsamp).flatten(), 200, 1800)
        av_resamp = np.clip(av_kde.resample(nsamp).flatten(), 0.01, 5.0)
        rv_resamp = np.clip(rv_kde.resample(nsamp).flatten(), 2.0, 6.0)
        resamp = np.hstack([resamp, tdust_resamp[:, None], av_resamp[:, None], rv_resamp[:, None]])

        truth_df = pd.DataFrame(resamp, columns=['temperature_median', 'luminosity_median', 
                                                 'tau_V_median', 'dust_temp_median', 'Av_median', 'Rv_median'])
        truth_df = truth_df[['temperature_median', 'dust_temp_median', 'tau_V_median', 
                             'luminosity_median', 'Rv_median', 'Av_median']]
        return truth_df, nsamp

    def generate_agb_blue_truth_sample(self, gal, met, samp_err=False,
                                        pcut=0.7, nsamp=20000, gdf=None):
        self.logger.info(f"Generating AGB/Blue truth sample for {gal} at Z={met} with pcut={pcut} and samp_err={samp_err}")
        if gdf is None:
            gdf = pd.read_csv(f'../data/dolphot/{gal}/{gal}_kdecat.csv')
        # subset = cdf[(cdf['galaxy'].isin(Z_MAPPING[met])) & ((cdf['p_agb'] > pcut) | 
        # (cdf['p_blue'] > pcut))].sample(100000, random_state=42)
        subset = gdf[(gdf['p_agb'] > pcut) | (gdf['p_blue'] > pcut)]
        X = subset[['temperature_median', 'luminosity_median', 'tau_V_median']].values
        if samp_err:
            sig = (subset[['temperature_elow', 'luminosity_elow', 'tau_V_elow']].values + \
                    subset[['temperature_eup', 'luminosity_eup', 'tau_V_eup']].values)/2
            X = np.random.normal(loc=X, scale=sig, size=(50, X.shape[0], X.shape[1])).reshape(-1, X.shape[1])
            resamp_idx = np.random.choice(range(X.shape[0]), size=10000, replace=False)
            X = X[resamp_idx]

        bics, models = [], []
        n_range = list(range(1, 10 + 1))

        for n in n_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                n_init=3,
                random_state=42,
            )
            gmm.fit(X)
            bics.append(gmm.bic(X))
            models.append(gmm)

        best_idx = int(np.argmin(bics))
        best_gmm = models[best_idx]
        resamp, _ = best_gmm.sample(nsamp)
        resamp[:, 0] = np.clip(resamp[:, 0], 2600, 5000)
        resamp[:, 1] = np.clip(resamp[:, 1], 3.0, 6.0)
        resamp[:, 2] = np.clip(resamp[:, 2], 1e-3, 8.0)

        tdust_kde = gaussian_kde(subset['dust_temp_median'].values)
        av_kde = gaussian_kde(subset['Av_median'].values)
        rv_kde = gaussian_kde(subset['Rv_median'].values)
        tdust_resamp = np.clip(tdust_kde.resample(nsamp).flatten(), 210, 1780)
        av_resamp = np.clip(av_kde.resample(nsamp).flatten(), 0.02, 4.9)
        rv_resamp = np.clip(rv_kde.resample(nsamp).flatten(), 2.1, 5.9)
        resamp = np.hstack([resamp, tdust_resamp[:, None], av_resamp[:, None], rv_resamp[:, None]])
        
        truth_df = pd.DataFrame(resamp, columns=['temperature_median', 'luminosity_median', 'tau_V_median', 
                                                 'dust_temp_median', 'Av_median', 'Rv_median'])
        truth_df = truth_df[['temperature_median', 'dust_temp_median', 'tau_V_median', 'luminosity_median', 
                             'Rv_median', 'Av_median']]
        return truth_df
    
    def plot_sim_sample(self, rsg_truth_sample, rsg_pvals, gal):
        plt.figure()
        plt.scatter(self.kdf[self.kdf['p_rsg'] > 0.7]['temperature_median'], self.kdf[self.kdf['p_rsg'] > 0.7]['luminosity_median'], 
                    c='coral', s=0.5, label=f'{gal.upper()} RSGs')
        plt.scatter(rsg_truth_sample['temperature_median'], rsg_truth_sample['luminosity_median'], c=rsg_pvals[:, 0], 
                    cmap='viridis', s=5, label='Simulated RSGs')
        plt.colorbar(label='RSG Probability')
        plt.legend(fontsize=8)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Luminosity (Lsun)')
        plt.title(f'{gal.upper()} - KDE RSG Probability')
        plt.show()

    def plot_purity_completeness_curve(self, rsg_pvals, agb_pvals):
        completeness = []
        for pc_ in np.linspace(0.00, 0.99, 100):
            completeness.append(np.sum(rsg_pvals[:, 0] > pc_) / len(rsg_pvals) * 100)

        purity = []
        for pc_ in np.linspace(0.00, 0.99, 100):
            p_ = (rsg_pvals[:, 0] > pc_).sum() / ((rsg_pvals[:, 0] > pc_).sum() + (agb_pvals[:, 0] > pc_).sum())
            purity.append(p_ * 100)

        fig, ax1 = plt.subplots()
        ax1.plot(np.linspace(0.00, 0.99, 100), completeness, color='blue')
        ax1.set_xlabel('p_rsg Threshold')
        ax1.set_ylabel('Completeness (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
        ax2.plot(np.linspace(0.00, 0.99, 100), purity, color='orange')
        ax2.set_ylabel('Purity (%)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        plt.axvline(0.7, color='sandybrown', linestyle='--', label='Bronze Threshold') 
        plt.axvline(0.8, color='silver', linestyle='--', label='Silver Threshold')
        plt.axvline(0.9, color='gold', linestyle='--', label='Gold Threshold')
        plt.legend(loc='lower center')
        plt.show()

        return purity, completeness

    def plot_sim_on_cmd(self, sim_rsg_truth, rsg_pvals, gal):
        idx1 = np.where(self.cl.rsgloader.cols['flts'][self.cl.rsgloader.flt_mask] == self.cl.f1.upper())[0][0]
        idx2 = np.where(self.cl.rsgloader.cols['flts'][self.cl.rsgloader.flt_mask] == self.cl.f2.upper())[0][0]
        self.cl.plot_cmd()
        m1, m2 = sim_rsg_truth[:, idx1], sim_rsg_truth[:, idx2]
        plt.scatter(m1-m2, m2+self.cl.rsgloader.dm, c=rsg_pvals[:, 0], cmap='inferno_r', s=5, label='Simulated RSGs')

        plt.colorbar(label='RSG Probability')
        plt.legend(fontsize=8)
        plt.title(f'{gal.upper()} - KDE RSG Probability')
        plt.show()
    
    def plot_sbi_hrd(self, rsg_inf_sample, rsg_truth_sample, rsg_pvals):
        plt.figure()
        plt.scatter(rsg_inf_sample[:, 0], rsg_inf_sample[:, 1], c=rsg_pvals[:, 0], cmap='viridis', s=5, label='Inferred RSGs')
        plt.colorbar(label='RSG Probability')
        plt.scatter(rsg_truth_sample['temperature_median'], rsg_truth_sample['luminosity_median'], 
                    fc='none', ec='blue', s=30, lw=0.2, label='True RSGs')
        plt.scatter(self.kdf[self.kdf['p_rsg'] > 0.7]['temperature_median'], self.kdf[self.kdf['p_rsg'] > 0.7]['luminosity_median'], 
                    c='coral', s=0.5, label='KDE RSGs')

    def calc_completeness_purity(self, gal, sim_pcut=0.8, samp_err=False):
        rsg_truth_sample, nrsg_sim = self.generate_rsg_truth_sample(gal, ALL_CONFIGS[gal]['load_args']['z'], 
                                                                    nsamp=500, pcut=sim_pcut, gdf=self.kdf, samp_err=samp_err)
        sim_rsg_truth = self.sf.simulator(rsg_truth_sample.values)
        inf_rsg_params = []
        for idx in tqdm(range(len(sim_rsg_truth))):
            i = sim_rsg_truth[idx]
            self.sf.hatp_x_y.set_default_x(i)
            samp = self.sf.sample_with_timeout(self.sf.hatp_x_y, sample_shape=(2500,), show_progress_bars=False, timeout=1)
            if samp is None:
                inf_rsg_params.append(i)
                continue
            med = np.percentile(samp, 50, axis=0)
            inf_rsg_params.append(med)
        inf_rsg_params = np.array(inf_rsg_params)

        agb_truth_sample = self.generate_agb_blue_truth_sample(gal, ALL_CONFIGS[gal]['load_args']['z'], 
                                                               nsamp=nrsg_sim*10, pcut=0.6, gdf=self.kdf, samp_err=samp_err)
        sim_agb_truth = self.sf.simulator(agb_truth_sample.values)
        inf_agb_params = []
        for idx in tqdm(range(len(sim_agb_truth))):
            i = sim_agb_truth[idx]
            self.sf.hatp_x_y.set_default_x(i)
            samp = self.sf.sample_with_timeout(self.sf.hatp_x_y, sample_shape=(2500,), show_progress_bars=False, timeout=1)
            if samp is None:
                continue
            med = np.percentile(samp, 50, axis=0)
            inf_agb_params.append(med)
        inf_agb_params = np.array(inf_agb_params)

        rsg_inf_sample =  np.stack((inf_rsg_params[:, 0], inf_rsg_params[:, 3], inf_rsg_params[:, 2])).T
        agb_inf_sample =  np.stack((inf_agb_params[:, 0], inf_agb_params[:, 3], inf_agb_params[:, 2])).T
        inf_sample = np.vstack([rsg_inf_sample, agb_inf_sample])
        inf_pvals = self.cl.kde.predict_proba(inf_sample)
        rsg_pvals = inf_pvals[:len(rsg_inf_sample)]
        agb_pvals = inf_pvals[len(rsg_inf_sample):]
        highl_mask = (inf_rsg_params[:, 3] > 5) & (inf_rsg_params[:, 0] < 4500) & (inf_rsg_params[:, 2] > 1)
        rsg_pvals[highl_mask, :] = np.array([[1.0, 0.0, 0.0]])

        self.plot_sim_sample(rsg_truth_sample, rsg_pvals, gal)

        p7, c7 = np.sum(rsg_pvals[:, 0] > 0.7) / len(rsg_pvals) * 100, ((rsg_pvals[:, 0] > 0.7).sum() / ((rsg_pvals[:, 0] > 0.7).sum() + (agb_pvals[:, 0] > 0.7).sum()))*100
        p8, c8 = np.sum(rsg_pvals[:, 0] > 0.8) / len(rsg_pvals) * 100, ((rsg_pvals[:, 0] > 0.8).sum() / ((rsg_pvals[:, 0] > 0.8).sum() + (agb_pvals[:, 0] > 0.8).sum()))*100
        p9, c9 = np.sum(rsg_pvals[:, 0] > 0.9) / len(rsg_pvals) * 100, ((rsg_pvals[:, 0] > 0.9).sum() / ((rsg_pvals[:, 0] > 0.9).sum() + (agb_pvals[:, 0] > 0.9).sum()))*100
        self.logger.info(f"Completeness and purity for {gal.upper()} RSGs:")
        self.logger.info(f"  pcut=0.7  →  Completeness: {p7:.1f}%  |  Purity: {c7:.1f}%")
        self.logger.info(f"  pcut=0.8  →  Completeness: {p8:.1f}%  |  Purity: {c8:.1f}%")
        self.logger.info(f"  pcut=0.9  →  Completeness: {p9:.1f}%  |  Purity: {c9:.1f}%")

        p, c = self.plot_purity_completeness_curve(rsg_pvals, agb_pvals)
        self.plot_sim_on_cmd(sim_rsg_truth, rsg_pvals, gal)
        self.plot_sbi_hrd(rsg_inf_sample, rsg_truth_sample, rsg_pvals)

        rsg_re_sample = np.stack((rsg_truth_sample.values[:, 0], rsg_truth_sample.values[:, 3], rsg_truth_sample.values[:, 2])).T
        agb_re_sample = np.stack((agb_truth_sample.values[:, 0], agb_truth_sample.values[:, 3], agb_truth_sample.values[:, 2])).T
        truth_sample = np.vstack([rsg_re_sample, agb_re_sample])
        truth_pvals = self.cl.kde.predict_proba(truth_sample)
        rsg_truth_pvals = truth_pvals[:len(rsg_truth_sample)]
        agb_truth_pvals = truth_pvals[len(rsg_truth_sample):]
        highl_mask = (rsg_truth_sample.values[:, 3] > 5) & (rsg_truth_sample.values[:, 0] < 4500) & (rsg_truth_sample.values[:, 2] > 1)
        rsg_truth_pvals[highl_mask, :] = np.array([[1.0, 0.0, 0.0]])

        self.plot_sim_sample(rsg_truth_sample, rsg_truth_pvals, gal)

        p7, c7 = np.sum(rsg_truth_pvals[:, 0] > 0.7) / len(rsg_truth_pvals) * 100, ((rsg_truth_pvals[:, 0] > 0.7).sum() / ((rsg_truth_pvals[:, 0] > 0.7).sum() + (agb_truth_pvals[:, 0] > 0.7).sum()))*100
        p8, c8 = np.sum(rsg_truth_pvals[:, 0] > 0.8) / len(rsg_truth_pvals) * 100, ((rsg_truth_pvals[:, 0] > 0.8).sum() / ((rsg_truth_pvals[:, 0] > 0.8).sum() + (agb_truth_pvals[:, 0] > 0.8).sum()))*100
        p9, c9 = np.sum(rsg_truth_pvals[:, 0] > 0.9) / len(rsg_truth_pvals) * 100, ((rsg_truth_pvals[:, 0] > 0.9).sum() / ((rsg_truth_pvals[:, 0] > 0.9).sum() + (agb_truth_pvals[:, 0] > 0.9).sum()))*100
        self.logger.info(f"Completeness and purity for {gal.upper()} RSGs:")
        self.logger.info(f"  pcut=0.7  →  Completeness: {p7:.1f}%  |  Purity: {c7:.1f}%")
        self.logger.info(f"  pcut=0.8  →  Completeness: {p8:.1f}%  |  Purity: {c8:.1f}%")
        self.logger.info(f"  pcut=0.9  →  Completeness: {p9:.1f}%  |  Purity: {c9:.1f}%")

        p, c = self.plot_purity_completeness_curve(rsg_truth_pvals, agb_truth_pvals)
        self.plot_sim_on_cmd(sim_rsg_truth, rsg_truth_pvals, gal)

        return p, c

    def plot_leave_out_comparison(self, test_df, p_new, gal, test_pcut=0.7, pcut=0.7):
        """
        Diagnostics for the leave-out test: how the retrained KDE's held-out
        probabilities compare with the full-catalogue values.
        """
        p_old  = test_df['p_rsg'].values
        is_rsg = test_df['truth'].values == 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        axes[0].scatter(p_old[~is_rsg], p_new[~is_rsg], s=3, c='coral',     alpha=0.4, label='AGB (full cat)')
        axes[0].scatter(p_old[is_rsg],  p_new[is_rsg],  s=3, c='royalblue', alpha=0.4, label='RSG (full cat)')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].axhline(pcut, color='grey', ls=':', lw=1)
        axes[0].axvline(test_pcut, color='grey', ls=':', lw=1)
        axes[0].set_xlabel('p_rsg (full catalogue)')
        axes[0].set_ylabel('p_rsg (leave-out KDE)')
        axes[0].set_title('Held-out probability comparison')
        axes[0].legend(fontsize=8, markerscale=3)

        axes[1].hist(p_new[is_rsg],  bins=50, range=(0, 1), color='royalblue', alpha=0.6, label='RSG (full cat)')
        axes[1].hist(p_new[~is_rsg], bins=50, range=(0, 1), color='coral',     alpha=0.6, label='AGB (full cat)')
        axes[1].axvline(pcut, color='k', ls='--', lw=1)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('p_rsg (leave-out KDE)')
        axes[1].set_ylabel('N')
        axes[1].set_title('Held-out probability distribution')
        axes[1].legend(fontsize=8)

        sc_ = axes[2].scatter(test_df['temperature_median'][is_rsg], test_df['luminosity_median'][is_rsg],
                              c=p_new[is_rsg], cmap='viridis', s=5, vmin=0, vmax=1)
        plt.colorbar(sc_, ax=axes[2], label='p_rsg (leave-out KDE)')
        axes[2].invert_xaxis()
        axes[2].set_xlabel('Temperature (K)')
        axes[2].set_ylabel('log L (Lsun)')
        axes[2].set_title('Held-out RSGs')

        fig.suptitle(f'{gal.upper()} - leave-out validation')
        fig.tight_layout()
        plt.show()
        return fig

    def calc_leave_out_val(self, gal, train_pcut=0.5, test_pcut=0.7, train_frac=0.5,
                           max_agb_ratio=10, n_classes=3, alpha=0.5, random_state=42,
                           plot=True):
        """
        Leave-out self-consistency test of the KDE classifier.

        A fresh AdaptiveKDE is trained on a random `train_frac` subset of the
        marginally-classified stars (p > `train_pcut`) and applied to the
        held-out remainder, scored only on stars the full-catalogue KDE is
        confident about (p > `test_pcut`).  Training on the looser cut makes
        the seed deliberately fuzzier than the evaluation set: with identical
        cuts the two samples are drawn from the same clean core of each class
        and the test returns ~100% by construction.

        The full-catalogue labels act as the reference "truth", so
        completeness/purity here quantify how stable the classification is
        against the choice (and quality) of training sample - they do NOT
        measure absolute accuracy, since a bias shared by both KDEs is
        invisible to this test.  Use `calc_completeness_purity` for that.

        Note that with `n_classes=2` the probabilities are renormalised over
        {RSG, AGB} only and therefore sit systematically higher than the
        3-class `p_rsg` in the catalogue; the returned `p_new` vs `p_old`
        comparison makes that offset explicit.

        Parameters
        ----------
        gal           : galaxy name (labelling only)
        train_pcut    : probability cut defining the TRAINING pool.  Lower than
                        `test_pcut` so the KDE is seeded with a less clean,
                        more inclusive sample.
        test_pcut     : probability cut defining the reference classes stars
                        are scored against.  Must be >= `train_pcut`.
        train_frac    : fraction of each training pool used for training
        max_agb_ratio : cap on the AGB:RSG ratio in the TRAINING set only.  The
                        test set keeps the catalogue's natural class ratio so
                        that purity stays meaningful.
        n_classes     : 2 (RSG/AGB) or 3 (RSG/AGB/Blue)
        alpha         : AdaptiveKDE bandwidth-adaptation exponent
        random_state  : seed for the train/test split
        plot          : produce diagnostic figures

        Returns
        -------
        res : dict with the retrained kde, the held-out frame, new/old
              probabilities, and completeness/purity at p = 0.7/0.8/0.9
        """
        if test_pcut < train_pcut:
            raise ValueError(f'test_pcut ({test_pcut}) must be >= train_pcut ({train_pcut})')

        cols = ['temperature_median', 'luminosity_median', 'tau_V_median']
        class_names = ['RSG', 'AGB', 'Blue'][:n_classes]
        pcols = ['p_rsg', 'p_agb', 'p_blue'][:n_classes]

        pools = [self.kdf[self.kdf[pc] > train_pcut] for pc in pcols]
        if any(len(p) == 0 for p in pools):
            raise ValueError(f'No stars above train_pcut {train_pcut} for one or more of {class_names}')

        n_rsg_train = int(len(pools[0]) * train_frac)
        if n_rsg_train < 10:
            raise ValueError(f'Only {len(pools[0])} RSGs above train_pcut {train_pcut}; too few to split')

        rng = np.random.RandomState(random_state)
        train, test = [], []
        for i, pool in enumerate(pools):
            n_train = int(len(pool) * train_frac)
            if i > 0:
                n_train = min(n_train, max_agb_ratio * n_rsg_train)
            tr = pool.sample(n=n_train, replace=False, random_state=rng)
            # held out, then restricted to the confidently-classified stars
            held = pool.drop(tr.index)
            te   = held[held[pcols[i]] > test_pcut]     # natural class ratio preserved
            tr, te = tr.copy(), te.copy()
            tr['truth'], te['truth'] = i, i
            train.append(tr)
            test.append(te)
            self.logger.info(f'  {class_names[i]:5s}: {len(pool)} above p={train_pcut}  ->  '
                             f'{len(tr)} train  |  {len(held)} held out  |  '
                             f'{len(te)} test (p > {test_pcut})')

        train_df = pd.concat(train)
        test_df  = pd.concat(test)
        if len(test[0]) == 0:
            raise ValueError(f'No held-out RSGs above test_pcut {test_pcut}; '
                             f'lower test_pcut or train_frac')

        kde = AdaptiveKDE(alpha=alpha).fit(train_df[cols].values,
                                           train_df['truth'].values,
                                           class_names=class_names)

        # single call: predict_proba max-normalises each class over the array it is given
        probs = kde.predict_proba(test_df[cols].values)
        p_new = probs[:, 0]
        p_old = test_df['p_rsg'].values
        is_rsg = test_df['truth'].values == 0

        thresholds = np.linspace(0.0, 0.99, 100)
        completeness, purity = [], []
        for t in thresholds:
            tp = np.sum(p_new[is_rsg] > t)
            fp = np.sum(p_new[~is_rsg] > t)
            completeness.append(tp / is_rsg.sum() * 100)
            purity.append(tp / (tp + fp) * 100 if (tp + fp) else np.nan)
        completeness, purity = np.array(completeness), np.array(purity)

        self.logger.info(f'Leave-out validation for {gal.upper()} ({n_classes}-class, '
                         f'train p>{train_pcut} @ {train_frac}, test p>{test_pcut}):')
        summary = {}
        for t in (0.7, 0.8, 0.9):
            tp = np.sum(p_new[is_rsg] > t)
            fp = np.sum(p_new[~is_rsg] > t)
            c_ = tp / is_rsg.sum() * 100
            p_ = tp / (tp + fp) * 100 if (tp + fp) else np.nan
            summary[t] = (c_, p_)
            self.logger.info(f'  pcut={t:.1f}  ->  Completeness: {c_:.1f}%  |  Purity: {p_:.1f}%')

        # agreement between the two KDEs at a common threshold
        flip_rsg = np.mean(p_new[is_rsg]  <= test_pcut) * 100
        flip_agb = np.mean(p_new[~is_rsg] >  test_pcut) * 100
        self.logger.info(f'  Label flips at p={test_pcut}: {flip_rsg:.1f}% of held-out RSGs lost, '
                         f'{flip_agb:.1f}% of held-out non-RSGs gained')
        self.logger.info(f'  Median p_rsg shift (new - old), held-out RSGs: '
                         f'{np.median(p_new[is_rsg] - p_old[is_rsg]):+.3f}')

        if plot:
            fig, ax1 = plt.subplots()
            ax1.plot(thresholds, completeness, color='blue')
            ax1.set_xlabel('p_rsg Threshold')
            ax1.set_ylabel('Completeness (%)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2 = ax1.twinx()
            ax2.plot(thresholds, purity, color='orange')
            ax2.set_ylabel('Purity (%)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            for t, c_ in zip((0.7, 0.8, 0.9), ('sandybrown', 'silver', 'gold')):
                ax1.axvline(t, color=c_, linestyle='--')
            ax1.set_title(f'{gal.upper()} - leave-out completeness / purity')
            plt.show()

            self.plot_leave_out_comparison(test_df, p_new, gal, test_pcut=test_pcut)

        return {'kde': kde, 'train_df': train_df, 'test_df': test_df,
                'p_new': p_new, 'p_old': p_old, 'is_rsg': is_rsg,
                'thresholds': thresholds, 'completeness': completeness,
                'purity': purity, 'summary': summary}