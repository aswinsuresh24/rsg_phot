import scienceplots
import numpy as np
import os, glob
import emcee
import progressbar
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import pickle
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats as scs
from scipy import interpolate
from scipy.integrate import simpson
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly import express as px
from astropy.io.misc.hdf5 import read_table_hdf5
import pandas as pd
import itertools
from tqdm import tqdm
import corner
import time
from scipy.stats import gaussian_kde, norm, skewnorm
from scipy.optimize import curve_fit
import torch
from sbi import utils as sbi_utils
from sbi.neural_nets import posterior_nn
from sbi import inference
from torch.distributions import MultivariateNormal, Exponential, LogNormal
from sbi.utils import MultipleIndependent, BoxUniform
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from plotly import express as px
import signal
import copy
from pathlib import Path
import json
from multiprocessing import Pool, Process
from contextlib import contextmanager

import rsg_phot.dust as dust
from rsg_phot.mcmc import mcmc
from rsg_phot.mc_parallel import rsg_dataloader, mcmcfit
from rsg_phot.rsg_sbi import sbifit
from rsg_phot import sbi_pp

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

with suppress_stdout():
    from synphot import SpectralElement
    from synphot.models import Empirical1D
    import optuna

ALL_CONFIGS = {
    'ngc5236': {'rsgcat': Path('../data/dolphot/ngc5236/ngc5236_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5236',
                              'procdir':Path('../data/dolphot/ngc5236'), 'photfile_path':None,
                              'dm':28.46, 'dmerr':0.05, 'z':-0.25, 'trgb':('F090W', 24.52),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False},
                'model': "957a9f22",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5236/ngc5236_sbi_cat.csv'),
                'slopes': (-10.997118155619603, -8.299065420560746),
                'intercepts': (25.027455331412106, 26.080644859813084),
                'lcut': 20.0},
    'ngc5194': {'rsgcat': Path('../data/dolphot/ngc5194/ngc5194_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5194',
                              'procdir':Path('../data/dolphot/ngc5194'), 'photfile_path':None,
                              'dm':29.67, 'dmerr':0.02, 'z':0.0, 'trgb':('F200W', 24.2),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts': ['F090W', 'F410M', 'F430M']},
                'model': "fb3481bf",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5194/ngc5194_sbi_cat.csv'),
                'slopes': (-11.38235294117647, -8.632587859424925),
                'intercepts':(25.339029411764706, 27.097386581469653),
                'lcut': 21.15},
    'ngc4258': {'rsgcat': Path('../data/dolphot/ngc4258/ngc4258_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4258',
                              'procdir':Path('../data/dolphot/ngc4258'), 'photfile_path':None,
                              'dm':29.397, 'dmerr':0.03, 'z':-0.25, 'trgb':('F090W', 25.055),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False},
                'model': "ef220e94",
                'f1': 'F115W', 'f2': 'F210M',
                'sbicat_path': Path('../data/dolphot/ngc4258/ngc4258_sbi_cat.csv'),
                'slopes': (-9.50366300366301, -9.599206349206352),
                'intercepts':(24.101245421245423, 26.033015873015874),
                'lcut': 21.0},
    'ngc628': {'rsgcat': Path('../data/dolphot/ngc628/ngc628_sil_rsgcat.csv'),
               'load_args': {'gal':'ngc628',
                             'procdir':Path('../data/dolphot/ngc628'), 'photfile_path':None,
                             'dm':30.04, 'dmerr':0.125, 'z':-0.25, 'trgb':('F090W', 29.13),
                             'modeltype':'MARCS', 'comp':'sil',
                             'keep_narrow':False, 'ignore_filts':['F090W', 'F140M', 'F182M', 'F410M', 'F430M', 'F480M']},
               'model': "89285257",
               'f1': 'F115W', 'f2': 'F200W',
               'sbicat_path': Path('../data/dolphot/ngc628/ngc628_sbi_cat.csv'),
               'slopes': (-12.158018867924522, -11.299303944315538),
               'intercepts': (27.09827830188679, 28.472839907192572),
               'lcut': 22.0},
    'ngc5643': {'rsgcat': Path('../data/dolphot/ngc5643/ngc5643_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5643',
                              'procdir':Path('../data/dolphot/ngc5643'), 'photfile_path':None,
                              'dm':30.57, 'dmerr':0.06, 'z':-0.25, 'trgb':('F090W', 26.20),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts':['F300M']},
                'model': "be8ad86d",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5643/ngc5643_sbi_cat.csv'),
                'slopes': (-21.6048780487805, -12.375527426160339),
                'intercepts': (30.181629268292685, 29.860953586497892),
                'lcut': 22.0},
    'ngc7320': {'rsgcat': Path('../data/dolphot/ngc7320/ngc7320_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc7320',
                              'procdir':Path('../data/dolphot/ngc7320'), 'photfile_path':None,
                              'dm':30.57, 'dmerr':0.5, 'z':-0.25, 'trgb':('F150W', 27.0),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False},
                'model': "c74a700a",
                'f1': 'F090W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc7320/ngc7320_sbi_cat.csv'),
                'slopes': (-8.544152744630077, -8.814249363867678),
                'intercepts': (30.82461097852029, 32.85034096692111),
                'lcut': 23.0},
    'ngc1367': {'rsgcat': Path('../data/dolphot/ngc1367/ngc1367_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc1367',
                              'procdir':Path('../data/dolphot/ngc1367'), 'photfile_path':None,
                              'dm':30.40, 'dmerr':0.07, 'z':0.00, 'trgb':('F090W', 29.13),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts':['F300M']},
                'model': "27370b04",
                'f1': 'F150W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc1367/ngc1367_sbi_cat.csv'),
                'slopes': (-16.12578616352201, -17.630630630630638),
                'intercepts': (29.09827830188679, 30.472839907192572),
                'lcut': 22.0},
    'ngc1365': {'rsgcat': Path('../data/dolphot/ngc1365/ngc1365_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc1365',
                              'procdir':Path('../data/dolphot/ngc1365'), 'photfile_path':None,
                              'dm':31.29, 'dmerr':0.065, 'z':-0.25, 'trgb':('F090W', 27.34),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False},
                'model': "05602ed7",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc1365/ngc1365_sbi_cat.csv'),
                'slopes': (-11.168421052631578, -11.948207171314737),
                'intercepts': (27.26592631578947, 29.59770517928287),
                'lcut': 23.0},
    'ngc4536': {'rsgcat': Path('../data/dolphot/ngc4536/ngc4536_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc4536',
                              'procdir':Path('../data/dolphot/ngc4536'), 'photfile_path':None,
                              'dm':30.99, 'dmerr':0.06, 'z':-0.25, 'trgb':('F090W', 27.01),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts': ['F115W', 'F444W']},
                'model': "9badc531",
                'f1': 'F150W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc4536/ngc4536_sbi_cat.csv'),
                'slopes': (-18.852941176470587, -17.630630630630638),
                'intercepts': (23.947, 25.269297297297296),
                'lcut': 22.6
                },
    'ngc5457': {'rsgcat': Path('../data/dolphot/ngc5457/ngc5457_sil_rsgcat.csv'),
                'load_args': {'gal':'ngc5457',
                              'procdir':Path('../data/dolphot/ngc5457'), 'photfile_path':None,
                              'dm':29.07, 'dmerr':0.05, 'z':-0.25, 'trgb':('F090W', 25.04),
                              'modeltype':'MARCS', 'comp':'sil',
                              'keep_narrow':False, 'ignore_filts':['F322W2']},
                'model': "c2583613",
                'f1': 'F115W', 'f2': 'F200W',
                'sbicat_path': Path('../data/dolphot/ngc5457/ngc5457_sbi_cat.csv'),
                'slopes': (-12.594059405940596, -12.495192307692308),
                'intercepts': (25.888089108910894, 27.75989903846154),
                'lcut': 20.6},
    # 'ngc4449': {'rsgcat': os.path.join(os.pardir, 'data/dolphot/ngc4449/ngc4449_ngc4485_combined_rsgcat.csv'),
    #             'load_args': {'gal':'ngc4449', 'procdir':os.path.join(os.pardir, 'data/dolphot/ngc4449'), 'photfile_path':None,
    #                           'dm':0.0, 'dmerr':0.32, 'z':-0.25, 'trgb':('F090W', -2.91),
    #                           'modeltype':'MARCS', 'comp':'sil', 'keep_narrow':False},
    #             'model': "f607c658"}
}

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
            # likelihoods[:, class_idx] /= np.max(likelihoods[:, class_idx])
        
        # Normalize
        probs = likelihoods / likelihoods.sum(axis=1, keepdims=True)
        return probs


class star_class(object):
    def __init__(self, gal:str, sbicat_path:Path, f1:str, f2:str):
        self.gal = gal
        if not sbicat_path.exists():
            raise FileNotFoundError(f"Catalog {sbicat_path} not found")
        self.df = pd.read_csv(sbicat_path)
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
                        (self.df[f'{self.f2}_mag'] > 10) & (self.df[f'{self.f2}_mag'] < 32)
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

    def plot_cmd(self, slopes=None, intercepts=None, lcut=None):
        rsg_cl = self.df[self.cmd_mask][f'{self.f1}_mag'] - self.df[self.cmd_mask][f'{self.f2}_mag']
        rsg_m = self.df[self.cmd_mask][f'{self.f2}_mag']

        fig, ax = plt.subplots()
        ax.hexbin(rsg_cl, rsg_m, cmap = 'viridis', bins=200, norm=mpl.colors.LogNorm());
        ax.invert_yaxis()
        ax.grid(ls='--', alpha=0.3)

        if slopes:
            self.slopes = slopes
            self.intercepts = intercepts
            self.lcut = lcut
            xs = np.linspace(-0.5, 1.5, 100)
            left = lin(xs, slopes[0], intercepts[0])
            right = lin(xs, slopes[1], intercepts[1])
            left_mask = (left > np.min(rsg_m)) & (left < np.max(rsg_m)) 
            right_mask = (right < np.max(rsg_m)) & (right > lcut) 
            ax.plot(xs[left_mask], left[left_mask], color='orange', lw=2, ls='--', label='left edge')
            ax.plot(xs[right_mask], right[right_mask], color='orange', lw=2, ls='--', label='right edge')
            ax.plot([np.max(xs[right_mask]), np.max(rsg_cl)], [lcut, lcut], color='orange', lw=2, ls='--', label='logL > 5')
            ax.legend()

        ax.set_xlabel(f'{self.f1}-{self.f2}')
        ax.set_ylabel(self.f2)
        ax.set_title(f'{self.gal.upper()}');  

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
        plt.title(f'{self.gal.upper()} CMD');

    def select_seed_sample(self, slopes, intercepts, lcut, plot=False, plot_3d=False):
        rsg_cl = self.df[self.cmd_mask][f'{self.f1}_mag'] - self.df[self.cmd_mask][f'{self.f2}_mag']
        rsg_m = self.df[self.cmd_mask][f'{self.f2}_mag']

        m2 = rsg_m > lin(rsg_cl, slopes[0], intercepts[0])
        blue_cut = (rsg_m < lin(rsg_cl, slopes[0], intercepts[0] - 0.5)) #| (~m2 & (rsg_m > 24))
        m3 = rsg_m < lin(rsg_cl, slopes[1], intercepts[1])
        m4 = rsg_m < lcut

        rsg_lit = self.df[self.cmd_mask][(m2&m3) | (~m3 & m4)]
        temperature_mask = (np.log10(rsg_lit['temperature_median']) < np.log10(4500)) & (np.log10(rsg_lit['temperature_median']) > np.log10(3200))
        luminosity_mask = (rsg_lit['luminosity_median'] < 4.5) & (rsg_lit['tau_V_median'] > 0.5)
        rsg_lit = rsg_lit[temperature_mask & ~luminosity_mask]

        m3 = rsg_m < lin(rsg_cl, slopes[1], intercepts[1] + 0.3)
        m4 = rsg_m < lcut + 0.2
        agb_lit = self.df[self.cmd_mask][~m3 & ~m4]
        blue_lit = self.df[self.cmd_mask][blue_cut]

        rsg_lit['class'] = 'RSG'
        agb_lit['class'] = 'AGB'
        blue_lit['class'] = 'Blue'  
        seed_df = pd.concat([rsg_lit, agb_lit, blue_lit], ignore_index=True)
        self.seed_df = seed_df

        if plot:
            self.plot_cmd(slopes, intercepts, lcut)
            plt.scatter(rsg_lit[f'{self.f1}_mag'] - rsg_lit[f'{self.f2}_mag'], rsg_lit[f'{self.f2}_mag'], color='red', s=1, label='RSG (seed)')
            plt.scatter(agb_lit[f'{self.f1}_mag'] - agb_lit[f'{self.f2}_mag'], agb_lit[f'{self.f2}_mag'], color='coral', s=1, label='AGB (seed)')
            plt.scatter(blue_lit[f'{self.f1}_mag'] - blue_lit[f'{self.f2}_mag'], blue_lit[f'{self.f2}_mag'], color='cyan', s=1, label='Blue (seed)')
            plt.legend()

        if plot_3d:
            fig = px.scatter_3d(seed_df, x='temperature_median', y='luminosity_median', z='tau_V_median', color='class', 
                                hover_data=['temperature_median', 'luminosity_median', 'tau_V_median'], symbol='class', 
                                color_discrete_map={'RSG':'royalblue', 'AGB':'coral', 'Blue':'magenta'},
                                title=f'{self.gal.upper()} Literature Sample')
            #update size of points and opacity
            fig.update_traces(marker=dict(size=3, opacity=0.3))
            fig.update_layout(scene = dict(
                                xaxis_title='Temperature (K)',
                                yaxis_title='Luminosity (Lsun)',
                                zaxis_title='Tau'),
                                legend_title='Class')
            fig.show()

            #save figure to html
            fig.write_html(f'../plots/{self.gal}_lit_sample_3d.html')
            
        return seed_df

    def kde_class(self, seed_df=None, assign_class=True):
        X = seed_df[['temperature_median', 'luminosity_median', 'tau_V_median']].values
        y = seed_df['class'].map({'RSG':0, 'AGB':1, 'Blue':2}).values
        self.kde = AdaptiveKDE(alpha=0.5).fit(X, y)

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
        self.plot_cmd(self.slopes, self.intercepts, self.lcut)
        class_colors = {'RSG':'royalblue', 'AGB':'coral', 'Blue':'magenta'}
        if plot_class:
            class_colors = {plot_class: class_colors[plot_class]}
        for class_label, color in class_colors.items():
            subset = self.df[self.cmd_mask & (self.df['class_label'] == class_label)]
            plt.scatter(subset[f'{self.f1}_mag'] - subset[f'{self.f2}_mag'], subset[f'{self.f2}_mag'], 
                        color=color, s=1, label=class_label)

    def plot_rsg_cmd(self, pcut=0.7):
        self.plot_cmd(self.slopes, self.intercepts, self.lcut)
        rsg_subset = self.df[self.cmd_mask & (self.df['p_rsg'] > pcut)]
        plt.scatter(rsg_subset[f'{self.f1}_mag'] - rsg_subset[f'{self.f2}_mag'], rsg_subset[f'{self.f2}_mag'], 
                    c=rsg_subset['p_rsg'], s=5, cmap='inferno', norm=mpl.colors.LogNorm())
        
    def bootstrap_kde_class(self, slopes, slope_errs, intercepts, intercept_errs, lcut, n_bootstrap=100, savepath=None):
        boot_df = self.df.copy()
        line_probs = np.zeros((len(boot_df), 3, n_bootstrap))
        for i in range(n_bootstrap):
            slopes_boot = np.random.normal(slopes, slope_errs)
            intercepts_boot = np.random.normal(intercepts, intercept_errs)
            # clip to within 3 sigma of means
            slopes_boot[0] = np.clip(slopes_boot[0], slopes[0] - 3*slope_errs[0], slopes[0] + 3*slope_errs[0])
            slopes_boot[1] = np.clip(slopes_boot[1], slopes[1] - 3*slope_errs[1], slopes[1] + 3*slope_errs[1])
            intercepts_boot[0] = np.clip(intercepts_boot[0], intercepts[0] - 3*intercept_errs[0], intercepts[0] + 3*intercept_errs[0])
            intercepts_boot[1] = np.clip(intercepts_boot[1], intercepts[1] - 3*intercept_errs[1], intercepts[1] + 3*intercept_errs[1])

            seed_df_boot = self.select_seed_sample(slopes_boot, intercepts_boot, lcut)
            probs = self.kde_class(seed_df_boot, assign_class=False)
            line_probs[:, :, i] = probs

        boot_df['p_rsg_mean'] = np.mean(line_probs[:, 0, :], axis=1)
        boot_df['p_agb_mean'] = np.mean(line_probs[:, 1, :], axis=1)
        boot_df['p_blue_mean'] = np.mean(line_probs[:, 2, :], axis=1)

        boot_df['p_rsg_sig'] = np.std(line_probs[:, 0, :], axis=1, ddof=1)
        boot_df['p_agb_sig'] = np.std(line_probs[:, 1, :], axis=1, ddof=1)
        boot_df['p_blue_sig'] = np.std(line_probs[:, 2, :], axis=1, ddof=1)
        self.df = boot_df
        if savepath:
            boot_df.to_csv(savepath, index=False)
        else:
            return boot_df
    
def run_sc(gal):
    try:
        print(f"Processing {gal} in process {os.getpid()}")
        sbicat_path = ALL_CONFIGS[gal]['sbicat_path']
        f1_, f2_ = ALL_CONFIGS[gal]['f1'], ALL_CONFIGS[gal]['f2']
        sc = star_class(gal, sbicat_path=sbicat_path, f1=f1_, f2=f2_)

        slopes, intercepts, lcut = ALL_CONFIGS[gal]['slopes'], ALL_CONFIGS[gal]['intercepts'], ALL_CONFIGS[gal]['lcut']
        slope_errs = (min(abs(slopes[0]*0.02), 0.25), min(abs(slopes[1]*0.02), 0.25))
        intercept_errs = (min(abs(intercepts[0]*0.01), 0.25), min(abs(intercepts[1]*0.01), 0.25))
        savepath = ALL_CONFIGS[gal]['load_args']['procdir'] / f'{gal}_kdecat.csv'
        sc.bootstrap_kde_class(slopes, slope_errs, intercepts, intercept_errs, lcut, n_bootstrap=2, savepath=savepath)
    except Exception as e:
        print(f"Error processing {gal}: {e}")
        traceback.print_exc()

if __name__ == "__main__":

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)

    # run kde in parallel for all galaxies over 10 cores
    gals = list(ALL_CONFIGS.keys())
    with Pool(processes=10) as pool:
        pool.map(run_sc, gals)