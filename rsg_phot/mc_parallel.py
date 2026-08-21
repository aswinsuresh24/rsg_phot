import warnings
warnings.simplefilter('ignore')
import numpy as np
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import pandas as pd
import itertools
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import logging
import multiprocessing_logging
from datetime import datetime
import matplotlib.pyplot as plt
import corner
from pathlib import Path

from rsg_phot.rsg_cat import save_photfiles
from rsg_phot.mcmc import mcmc, NRC_FILTS, MIRI_FILTS, MODE_FILTS, filt_wavelength
from rsg_phot.utils import logger, TqdmToLogger

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''

    parser = argparse.ArgumentParser(description='Fit red supergiant SEDs using MCMC')
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
    parser.add_argument('--mode', type=str, default='nircam', choices=list(MODE_FILTS.keys()),
                        help='Photometry to fit: nircam (NIRCam only) or miri (NIRCam + MIRI)')
    parser.add_argument('--keep_narrow', default=False, action=argparse.BooleanOptionalAction,  help='Fit narrow band photometry?')
    parser.add_argument('--ignore_filts', nargs='*', help='Photometry to avoid fitting')

    parser.add_argument('--chimin', default=False, action=argparse.BooleanOptionalAction, help='Apply chi-min cuts to create RSG catalog')
    parser.add_argument('--min_det', type=int, default=4, help='Minimum number of detections for RSG catalog')
    parser.add_argument('--mcmc_fit', default=False, action=argparse.BooleanOptionalAction, help='Run MCMC fitting on RSG catalog')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    parser.add_argument('--redo_mcmc', type=bool, default=False, help='Redo MCMC?')

    return parser

RNG_SEED = 42

class rsg_dataloader(object):
    def __init__(self, gal, procdir:Path, photfile_path=None, dm:float=30.0, dmerr:float=0.5, z:float=0.00, 
                 modeltype:str='MARCS', trgb:tuple=('F090W', 30.0), comp:str='sil', keep_narrow:bool=False,
                 ignore_filts=None, rsgcat=None, mode:str='nircam'):

        self.gal = gal
        self.procdir = Path(procdir)
        self.procdir.mkdir(parents=True, exist_ok=True)
        self.photfile_path = photfile_path
        self.backend_dir = self.procdir / 'backends'
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.keep_narrow = keep_narrow
        if ignore_filts is None: ignore_filts = []
        self.ignore_filts = ignore_filts
        self.dm, self.dmerr = dm, dmerr
        self.z = z
        self.modeltype = modeltype
        self.comp = comp
        self.mode = mode.lower()
        if self.mode not in MODE_FILTS:
            raise ValueError(f'Mode {mode} is not valid - use one of {list(MODE_FILTS.keys())}')

        self.nrc_filts = NRC_FILTS
        self.miri_filts = MIRI_FILTS
        # filters actually fit/plotted in this mode (nircam only, or nircam + miri)
        self.filts = MODE_FILTS[self.mode]
        self.wv_all = list(filt_wavelength(self.filts))

        if rsgcat is not None:
            if self.photfile_path is not None:
                try:
                    self.photfile_path = Path(self.photfile_path)
                    self.cat = self.read_cat(self.photfile_path)
                except Exception as e:
                    self.logger.info('Cannot load complete DOLPHOT catalog due to the following exception: ', e)
            else:
                self.logger.info('WARNING: photfile_path not provided; not reading in complete photometry')
                self.cat = None
            self.rsgcat = rsgcat
            self.rsgcat.reset_index(inplace=True, drop=True)
            self.rsgcat.loc[:, 'index'] = self.rsgcat.index
            self.set_cols(self.rsgcat)
        else:
            if self.photfile_path is None:
                raise ValueError('At least one of rsgcat or photfile_path is required as input')
            self.photfile_path = Path(self.photfile_path)
            if not self.photfile_path.exists():
                raise ValueError(f'photfile_path {str(self.photfile_path.resolve(strict=False))} does not exist')
            self.cat = self.read_cat(self.photfile_path)
            self.set_cols(self.cat)
            self.rsgcat = None

        self.gen_mc_obj = mcmc(dm=self.dm, dmerr=self.dmerr, z=self.z, model_type=self.modeltype,
                               comp=self.comp, mode=self.mode)
        self.gen_mc_obj.verbose = False
        self.gen_mc_obj.dirs['backends'] = self.backend_dir
        self.trgb = trgb
        self.trgb = (int(np.where(self.filts==self.trgb[0].upper())[0][0]), float(self.trgb[1]))
        self.chimin_params = {
            'teff_' : np.arange(2600, 5050, 50),
            'tdust_' : np.arange(200, 1800, 100),
            'tau_' : np.array(list(np.linspace(0.01, 2, 21)) + list(np.arange(2.5, 5.5, 0.5))),
            'Av_' : np.array(list(np.linspace(0, 1, 5))+ list(np.linspace(1.5, 3, 4)))
        }
        self.chimin_modeldir = Path("..") / "data" / "chimin_models"
        
    def read_cat(self, photfile_path):
        catpath = self.procdir / 'proc'
        catpath.mkdir(parents=True, exist_ok=True)
        savefiles = list(catpath.glob('*csv'))
        if len(savefiles) == 0:
            save_photfiles(photfile_path, catpath)
            savefiles = list(catpath.glob('*csv'))

        cat = None
        for fl in savefiles:
            if cat is None:
                cat = pd.read_csv(fl)
            else:
                df_ = pd.read_csv(fl)
                cat = pd.concat([cat, df_])
        cat.reset_index(inplace=True)
        cat['index'] = cat.index
        cat = cat.replace(np.nan, 99.999)
        self.logger.info(f'DOLPHOT catalog contains {len(cat)} objects after photometry cuts')

        return cat
    
    def set_cols(self, cat):
        fls = ['mag' in i for i in cat.columns]
        fls = cat.columns[fls]
        flts = [i.replace('_mag','') for i in fls]
        cat_wv = filt_wavelength(flts)
        magcols = [i+'_mag' for i in flts]
        errcols = [i+'_err' for i in flts]

        self.cols = {'flts' : np.array(flts),
                     'magcols' : np.array(magcols),
                     'errcols' : np.array(errcols),
                     'cat_wv': cat_wv}
        
        ign_mask = np.array([i.upper() in self.ignore_filts for i in self.cols['flts']])
        # drop filters the mode's model grid does not cover (e.g. MIRI columns in nircam mode)
        unknown = ~np.isin(np.char.upper(self.cols['flts'].astype(str)), self.filts)
        if unknown.any():
            self.logger.info(f'Ignoring filters not in the {self.mode} model grid: '
                             f'{list(self.cols["flts"][unknown])}')
        ign_mask |= unknown
        if not self.keep_narrow:
            narrow_mask = np.array(['N' in i for i in self.cols['flts']])
            self.flt_mask = ~(narrow_mask | ign_mask)
        else:
            self.flt_mask = ~ign_mask

    def mp_init(init_success: int = 0,
                init_failed: int = 0,
                init_success_cols: list =[]):
        global success
        global failed
        global success_files
        success = init_success
        failed = init_failed
        success_files = init_success_cols

    def base_cuts(self, cat, min_det=4, apply_opt_cut=False):
        def detmask(cat_mags_det, min_det=4):
            if (np.min(self.cols['cat_wv'][self.flt_mask]) < 1.2):
                opt_cut = 1.2
            else:
                opt_cut = 1.6
            opt_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'][self.flt_mask] < opt_cut)]]
            if apply_opt_cut:
                nir_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'][self.flt_mask] > opt_cut) & (self.cols['cat_wv'][self.flt_mask] < 2.6)]]
            else:
                nir_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'][self.flt_mask] < 2.6)]]
            mir_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'][self.flt_mask] > 2.6)]]

            ndetm = cat_mags_det.sum(axis=1) >= min_det
            if apply_opt_cut:
                detm = (opt_dets.sum(axis=1) > 0) & (nir_dets.sum(axis=1) > 0) & (mir_dets.sum(axis=1) > 0) & ndetm
            else:
                detm = (nir_dets.sum(axis=1) > 0) & (mir_dets.sum(axis=1) > 0) & ndetm

            return detm
        
        self.logger.info(f'Applying ndet cuts')
        self.gen_mc_obj.bounds['luminosity'] = [4.0, 4.1]

        base_models = np.zeros((2000, len(self.filts)))
        basedf_cols = [i+'_mag' for i in self.filts]

        sample_params = self.gen_mc_obj.get_init_pos(1000, seed=RNG_SEED)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.filts]) + self.gen_mc_obj.dm
            base_models[i, :] = model_mag

        self.gen_mc_obj.bounds['luminosity'] = [5.8, 6]
        sample_params = self.gen_mc_obj.get_init_pos(1000, seed=RNG_SEED)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.filts]) + self.gen_mc_obj.dm
            base_models[i+1000, :] = model_mag

        min_model = np.max(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        min_model_df = pd.DataFrame([dict(zip(basedf_cols, min_model))])

        max_model = np.min(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        max_model_df = pd.DataFrame([dict(zip(basedf_cols, max_model))])

        cat_mags = cat[self.cols['magcols'][self.flt_mask]].replace(99.999, 0.5)
        mindf = cat_mags-min_model_df[self.cols['magcols'][self.flt_mask]].to_numpy()
        cat_mags = cat_mags.replace(0.5, 99.999)
        maxdf = cat_mags-max_model_df[self.cols['magcols'][self.flt_mask]].to_numpy()

        difm = (mindf < 0).all(axis=1) & (maxdf > 0).all(axis=1)
        ndet = (cat_mags > 10) & (cat_mags < 38) 
        ndetm = detmask(ndet, min_det=min_det)
        self.logger.info(f'{ndetm.sum()} objects after n>4 cut')
        base_mask = difm & ndetm 
        self.logger.info(f'{base_mask.sum()} objects after minmax cut')
        rsgcat = cat[base_mask].replace(np.nan, 99.999)
        self.logger.info(f'RSG catalog contains {len(rsgcat)} objects after ndet cuts')

        return rsgcat
    
    def create_modeldf(self, outpath:Path=None):
        if outpath is None:
            outpath = self.chimin_modeldir / f'{self.comp}_z{self.z:.2f}_{self.mode}_modeldf.csv'
        else:
            outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        # if modeldf for composition and metalllicity exists, read it
        if outpath.exists():
            modeldf = pd.read_csv(outpath)
        # else create a grid at 10 Mpc (needs to be done once)
        else:
            self.logger.info(f'Creating modeldf for comp={self.comp}, Z={self.z:.2f}, mode={self.mode}: {outpath}')
            # modeldf = pd.DataFrame(columns = ['Teff', 'Tdust', 'Tau', 'Av'] + list(self.filts))
            nmodel = 1
            for v in self.chimin_params.values():
                nmodel *= len(v)

            model_ = np.zeros((nmodel, 4 + len(self.filts)))
            self.gen_mc_obj.reset_bounds()
            param_combos = itertools.product(self.chimin_params['teff_'], self.chimin_params['tdust_'],
                                             self.chimin_params['tau_'], self.chimin_params['Av_'])
            for i, (a1, a2, a3, a4) in enumerate(tqdm(param_combos, total=nmodel, mininterval=10)):
                pm_ = [a1, a2, a3, 3.0, 3.1, a4]
                model_mag = np.array([self.gen_mc_obj.model[f](pm_).flatten()[0] for f in self.filts])
                model_[i, :] = ([a1, a2, a3, a4] + list(model_mag))

            modeldf = pd.DataFrame(data=model_, columns=['Teff', 'Tdust', 'Tau', 'Av'] + list(self.filts))
            modeldf.to_csv(outpath, index=False)

        # add distance to galaxy to modeldf
        magcols = ['F' in i for i in modeldf.columns]
        magcols = modeldf.columns[magcols]
        modeldf[magcols] = modeldf[magcols] + self.gen_mc_obj.dm
        return modeldf

    def gen_phot(self, col, noise_floor=0.01):
        phot = {'mag': np.array(col[self.cols['magcols'][self.flt_mask]], dtype=float),
                'magerr': np.array(col[self.cols['errcols'][self.flt_mask]], dtype=float),
                'inst_filt': np.array(self.cols['flts'][self.flt_mask]),
                'index': f'{self.gal}_{int(col['index'])}'}

        limmask = (phot['mag'] > 90.0) | (phot['magerr'] > 90.0) | (np.isnan(phot['mag'])) | (np.isnan(phot['magerr'])) | (phot['magerr'] < 1e-4)
        phot['mag'] = phot['mag'][~limmask]
        phot['magerr'] = np.sqrt(phot['magerr'][~limmask]**2 + noise_floor**2)
        phot['inst_filt'] = phot['inst_filt'][~limmask]

        return phot
    
    def chimin(self, phot, modeldf):
        modelmagval = modeldf[phot['inst_filt']].values
        submagval = modelmagval - phot['mag']
        meanval = np.average(submagval, weights = 1/phot['magerr']**2, axis=1)
        submagval = submagval - meanval[:, None]
        chisqe = np.sum(submagval**2/phot['magerr']**2, axis=1)/len(phot['mag'] - 1)
        chisq = np.sum(submagval**2, axis=1)/len(phot['mag'] - 1)
        minchisq = np.argmin(chisqe)

        bestparams = modeldf.loc[minchisq, ['Teff', 'Tdust', 'Tau', 'Av']].values
        pm_ = [bestparams[0], bestparams[1], bestparams[2], 3.0, 3.1, bestparams[3]]
        bestmodel = np.array([self.gen_mc_obj.model[f](pm_).flatten()[0] for f in phot['inst_filt']]) + self.gen_mc_obj.dm 
        lum = np.mean((phot['mag'] - bestmodel)/-2.5) + 3.0

        return chisq[minchisq], minchisq, lum      

    def chimin_cuts(self, base_rsgcat, modeldf):
        base_rsgcat.loc[:, ['chimin', 'teff_chisq', 'tdust_chisq', 'tau_chisq', 'av_chisq', 'lum_chisq']] = 0.0
        self.logger.info(f'Applying chisq cuts')
        tqdm_out = TqdmToLogger(self.logger, level=logging.INFO)
        for idx in tqdm(base_rsgcat.index, file=tqdm_out, total=len(base_rsgcat), mininterval=20):
            testcol = base_rsgcat.loc[idx]
            phot = self.gen_phot(testcol)

            c_, m_, l_ = self.chimin(phot, modeldf)
            base_rsgcat.loc[idx, ['chimin', 'lum_chisq']] = c_, l_
            base_rsgcat.loc[idx, ['teff_chisq', 'tdust_chisq', 'tau_chisq', 'av_chisq']] = modeldf.loc[m_, ['Teff', 'Tdust', 'Tau', 'Av']].values

        chi_cut = np.percentile(base_rsgcat['chimin'], 90)
        mask = base_rsgcat['chimin'] < chi_cut
        base_rsgcat.loc[:, 'chimin_pass'] = mask

        self.logger.info(f"RSG catalog contains {base_rsgcat['chimin_pass'].sum()} objects after chisq cuts")
        return base_rsgcat
    
    def plot_error_dist(self, cat=None):
        if cat is None:
            cat = self.rsgcat

        errcols = self.cols['errcols']
        err_df = cat[self.cols['errcols']].replace({9.999: np.nan, 99.999: np.nan})

        data = [err_df[c].dropna().values for c in errcols]
        labels = [c.replace('_err','') for c in errcols]

        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)

        for box in bp['boxes']:
            box.set(facecolor='cornflowerblue', edgecolor='black', alpha=0.5)
        for whisker in bp['whiskers']:
            whisker.set(color='black')
        for median in bp['medians']:
            median.set(color='royalblue', linewidth=1.5)

        ax.set_yscale('log')
        ax.set_ylabel('Magnitude error')
        ax.set_xlabel('Filter')
        ax.set_title('Error per filter')
        plt.xticks(rotation=90)
        plt.grid(alpha=0.3, which='both', linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_basecut(self):
        self.gen_mc_obj.bounds['luminosity'] = [3.5, 3.6]

        base_models = np.zeros((2000, len(self.nrc_filts)))
        basedf_cols = [i+'_mag' for i in self.nrc_filts]

        sample_params = self.gen_mc_obj.get_init_pos(1000, seed=RNG_SEED)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm 
            base_models[i, :] = model_mag

        self.gen_mc_obj.bounds['luminosity'] = [5.8, 6]
        sample_params = self.gen_mc_obj.get_init_pos(1000, seed=RNG_SEED)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm 
            base_models[i+1000, :] = model_mag

        min_model = np.max(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        min_model_df = pd.DataFrame([dict(zip(basedf_cols, min_model))])

        max_model = np.min(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        max_model_df = pd.DataFrame([dict(zip(basedf_cols, max_model))])

        random_sed = self.cat[self.cols['magcols']].replace(99.999, np.nan).sample(10000)
        for idx in random_sed.index:
            nanmask = np.isnan(random_sed.loc[idx].values)
            plt.plot(self.cols['cat_wv'][~nanmask], random_sed.loc[idx].values[~nanmask], color='gray', alpha=0.01)
        plt.scatter(self.wv_all[self.trgb[0]], self.trgb[1], marker='*', color='crimson', s=50, ec='black')

        plt.plot(self.wv_all, min_model)
        plt.plot(self.wv_all, max_model)
        plt.gca().invert_yaxis()
        plt.show()

        self.gen_mc_obj.reset_bounds()

        return min_model
    
    def apply_initial_cuts(self, outpath=None, min_det=4):
        base_rsgcat = self.base_cuts(self.cat, min_det=min_det)
        modeldf = self.create_modeldf(outpath=outpath)
        rsgcat = self.chimin_cuts(base_rsgcat, modeldf)
        rsgcat.to_csv(self.procdir / f'{self.gal}_{self.comp}_{self.mode}_rsgcat.csv')
        self.rsgcat = rsgcat

class mcmcfit(object):
    def __init__(self, rsg_dataloader: rsg_dataloader, ncores=1, keep_narrow=False, ignore_filts=None, 
                 modeltype='MARCS', comp='sil', redo_mcmc=False, verbose=False, mode=None):
        if rsg_dataloader.rsgcat is None:
            raise ValueError('Input RSG dataloader should contain rsgcat; Load the catalog into the dataloader using apply_initial_cuts()')
        
        self.rsgloader = rsg_dataloader
        self.rsgloader.keep_narrow = keep_narrow
        if ignore_filts is None: ignore_filts=[]
        self.rsgloader.ignore_filts = ignore_filts

        self.rsgcat = self.rsgloader.rsgcat
        self.logger = self.rsgloader.logger
        self.modeltype = modeltype
        self.comp = comp
        self.ncores = ncores
        self.mcfit_params = np.array(['temperature', 'dust_temp', 'tau_V', 'luminosity', 'Rv', 'Av'])
        self.redo_mcmc = redo_mcmc
        # the dataloader sets the mode unless it is overridden here
        self.mode = self.rsgloader.mode if mode is None else mode.lower()
        if self.mode not in MODE_FILTS:
            raise ValueError(f'Mode {mode} is not valid - use one of {list(MODE_FILTS.keys())}')
        if self.mode!=self.rsgloader.mode:
            self.rsgloader.mode = self.mode
            self.rsgloader.filts = MODE_FILTS[self.mode]
            self.rsgloader.wv_all = list(filt_wavelength(self.rsgloader.filts))
            self.rsgloader.set_cols(self.rsgcat)

        self.mc_obj = mcmc(dm=self.rsgloader.dm, dmerr=self.rsgloader.dmerr, z=self.rsgloader.z,
                           model_type=self.modeltype, comp=self.comp, mode=self.mode)
        self.mc_obj.verbose=verbose
        if self.rsgloader.backend_dir.exists():
            self.mc_obj.dirs['backends'] = self.rsgloader.backend_dir
        else:
            self.mc_obj.dirs['backends'] = Path("..") / 'data' / 'backends'

    def mp_init(self,
                init_success: int = 0,
                init_failed: int = 0,
                init_success_cols: list =[]):
        global success
        global failed
        global success_files
        success = init_success
        failed = init_failed
        success_files = init_success_cols

    def parallel_mc_worker(self, col, nsteps=350, nwalkers=64, burn_in=75):
        try:
            self.logger.info(f"[{datetime.now().strftime('%a, %d %b %Y %H:%M:%S')}] Running MCMC on index {int(col['index'])}")
            phot = self.rsgloader.gen_phot(col)

            self.mc_obj.run_emcee(phot, nsteps=nsteps, nwalkers=nwalkers, burn_in=burn_in, 
                                  return_params=False, calculate_chi=False)

        except Exception as e:
            traceback.format_exc()

    def read_mc_params(self, col, burn_in=75, thin=1, calculate_chi=True):
        phot = self.rsgloader.gen_phot(col)
        fit_params, min_chi_params, chi_posterior, chi_best = self.mc_obj.read_params(phot, burn_in=burn_in, 
                                                                                     thin=thin, calculate_chi=calculate_chi)

        for i, p_ in enumerate(fit_params.keys()):
            self.rsgcat.loc[int(col['index']), [p_+'_mc', p_+'_elow', p_+'_eup', p_+'_best']] = list(fit_params[p_]) + [min_chi_params[i]]

        self.rsgcat.loc[int(col['index']), ['chi_posterior', 'chi_best']] = [chi_posterior, chi_best]

    def plot_corner(self, phot, min_chi_params):
        truths = np.array(min_chi_params)
        reader = self.mc_obj.load_backend(phot)
        sample = np.array(reader.get_chain(flat=True, discard=75, thin=1))

        medians = np.percentile(sample, 50, axis=0)
        p16, p84 = np.percentile(sample, 16, axis=0), np.percentile(sample, 84, axis=0)

        labels = self.mcfit_params
        ndim = len(labels)

        fig = corner.corner(sample, labels=labels, color='C0', hist_kwargs={'density': True},
                            label_kwargs={'fontsize': 10}, show_titles=False, title_fmt='.3f', plot_density=False,
                            plot_datapoints=False, fill_contours=False, truth_color='royalblue', truths=truths)
        
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax_diag = axes[i, i]
            # ax_diag.axvline(medians[i], color='royalblue', lw=1)
            ax_diag.axvline(p16[i], color='royalblue', lw=1, ls='--')
            ax_diag.axvline(p84[i], color='royalblue', lw=1, ls='--')
            ax_diag.set_title(r"${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$".format(medians[i], p84[i]-medians[i], medians[i]-p16[i]), fontsize=9)

            for j in range(i):
                ax = axes[i, j]
                ax.scatter(medians[j], medians[i], marker='x', color='k', s=30)

        fig.suptitle(phot['index'], fontsize=12)
        return fig
    
    def plot_fit(self, phot, fit_params, min_chi_params, chi_post, chi_best, save=False):
        fit_pe = np.array(list(fit_params.values()))
        params = np.meshgrid(*fit_pe[:, 0], indexing='ij', sparse=True)
        model_mag = np.array([self.mc_obj.model[f](params).flatten()[0] for f in phot['inst_filt']]) + self.mc_obj.dm
        model_mag_plot = np.array([self.mc_obj.model[f](params).flatten()[0] for f in self.rsgloader.filts]) + self.mc_obj.dm

        mc_params = np.meshgrid(*min_chi_params, indexing='ij', sparse=True)
        chi_mag = np.array([self.mc_obj.model[f](mc_params).flatten()[0] for f in phot['inst_filt']]) + self.mc_obj.dm
        chi_mag_plot = np.array([self.mc_obj.model[f](mc_params).flatten()[0] for f in self.rsgloader.filts]) + self.mc_obj.dm

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0.1)
        wv = filt_wavelength(phot['inst_filt'])
        ax1.errorbar(x=wv, y=phot['mag'], yerr=phot['magerr'], linestyle='none', marker='o', markerfacecolor='cornflowerblue', 
                    markeredgecolor='black', ecolor='cornflowerblue')
        ax1.plot(self.rsgloader.wv_all, model_mag_plot, marker = 's', markerfacecolor = 'None', markeredgecolor = 'royalblue', 
                markersize = 8, ls='--', lw=2, alpha = 0.6, color='cornflowerblue', 
                label = rf'Posterior $\chi^2={chi_post*(len(phot['inst_filt']) - 1):.2f}$')
        ax1.plot(self.rsgloader.wv_all, chi_mag_plot, marker = 's', markerfacecolor = 'None', markeredgecolor = 'hotpink', 
                markersize = 8, ls='--', lw=2, alpha = 0.6, color='hotpink', 
                label = rf'min $\chi^2={chi_best*(len(phot['inst_filt']) - 1):.2f}$')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.invert_yaxis()
        ax1.legend()
        ax1.set_ylabel('AB mag')

        ax2.errorbar(wv, phot['mag'] - model_mag, yerr=phot['magerr'], linestyle='none', marker='o', markerfacecolor='cornflowerblue',
                    markeredgecolor='black', ecolor='cornflowerblue', label = 'Posterior')
        ax2.errorbar(wv, phot['mag'] - chi_mag, yerr=phot['magerr'], linestyle='none', marker='o', markerfacecolor='hotpink',
                    markeredgecolor='black', ecolor='hotpink', label = r'min $\chi^2$')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlabel(r'$\lambda(\mu m)$')
        ax2.set_ylabel('Residuals')
        ax2.axhline(0.0, linestyle='--', alpha=0.5, color='black');
        if save:
            plt.savefig(f'plots/MCMC_{phot['index']}.png', bbox_inches='tight', dpi=500)
    
    def run_mcmc(self, col, nsteps=350, nwalkers=64, burn_in=75, verbose=True, plot=True):
        phot = self.rsgloader.gen_phot(col)
        fit_params, min_chi_params, chi_posterior, chi_best = self.mc_obj.run_emcee(phot, nsteps=nsteps, 
                                                                                    nwalkers=nwalkers, burn_in=burn_in, 
                                                                                    return_params=True, calculate_chi=True)
        
        if plot:
            self.plot_fit(phot=phot, fit_params=fit_params, min_chi_params=min_chi_params, 
                          chi_post=chi_posterior, chi_best=chi_best)
            _ = self.plot_corner(phot=phot, min_chi_params=min_chi_params)
        
        return fit_params, min_chi_params, chi_posterior, chi_best

    def run_mcmc_parallel(self):
        for p_ in self.mcfit_params:
            self.rsgcat.loc[:, [p_+'_mc', p_+'_elow', p_+'_eup', p_+'_best']] = np.nan
        self.rsgcat.loc[:, ['chi_posterior', 'chi_best']] = np.nan

        # matches the backend filename built in mcmc.load_backend
        mode_suffix = '' if self.mode=='nircam' else '_'+self.mode
        argument_list = []
        for idx_ in self.rsgcat.index:
            backfile = self.rsgloader.backend_dir / str(self.rsgloader.gal.upper()+'_'+str(int(idx_))+'_'+self.rsgloader.modeltype+mode_suffix+'.h5')
            if self.redo_mcmc or not backfile.exists():
                argument_list.append([self.rsgcat.loc[idx_]])
            else:
                continue

        # create argument list for starmap async
        multiprocessing_logging.install_mp_handler(self.logger)
        p = Pool(initializer=self.mp_init, processes=self.ncores)
        result = p.starmap_async(self.parallel_mc_worker, argument_list)

        # run MCMC
        self.logger.info(f"Starting MCMC")
        result.get()

        # read backend file for each object to calculate best fit parameters and write to rsgcat
        for idx_ in self.rsgcat.index:
            self.read_mc_params(self.rsgcat.loc[idx_])

        self.rsgcat.to_csv(self.rsgloader.procdir / f'rsgcat_{self.rsgloader.gal}_MCMC_{self.rsgloader.modeltype}_{self.rsgloader.mode}.csv', index=False)
    
if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.mcmc_fit:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if args.rsgcat is not None:
        rsgcat_in = pd.read_csv(args.rsgcat)
    else: rsgcat_in = None

    load_args = {
        'gal':args.gal, 'procdir':args.procdir, 'photfile_path':args.photfile_path,
        'dm':args.dm, 'dmerr':args.dmerr, 'z':args.z, 'trgb':tuple(args.trgb),
        'modeltype':args.modeltype, 'comp':args.comp, 'keep_narrow':args.keep_narrow, 
        'ignore_filts':args.ignore_filts, 'rsgcat':rsgcat_in, 'mode':args.mode
    }

    rsgloader = rsg_dataloader(**load_args)
    if not args.chimin and not args.mcmc_fit:
        rsgloader.logger.info('No operation specified. Use --chimin to create RSG catalog or --mcmc_fit to run MCMC fitting on RSG catalog.')
        sys.exit()

    if args.chimin:
        rsgloader.apply_initial_cuts(min_det=args.min_det)

    if args.mcmc_fit:
        sedfit = mcmcfit(rsgloader, ncores=args.ncores, verbose=False,
                        modeltype=args.modeltype, comp=args.comp, redo_mcmc=args.redo_mcmc,
                        mode=args.mode)
        sedfit.run_mcmc_parallel()