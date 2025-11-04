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
from rsg_cat import save_photfiles
from mcmc import mcmc
from multiprocessing import Pool
import argparse
import logging
import multiprocessing_logging
from datetime import datetime
import matplotlib.pyplot as plt
import corner

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
    parser.add_argument('--keep_narrow', type=bool, default=False, help='Fit narrow band photometry?')
    parser.add_argument('--ignore_filts', nargs='*', help='Photometry to avoid fitting')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    parser.add_argument('--redo_mcmc', type=bool, default=False, help='Redo MCMC?')

    return parser


class rsg_dataloader(object):
    def __init__(self, gal, procdir, photfile_path=None, dm=30.0, dmerr=0.5, z=0.00, 
                 modeltype='MARCS', trgb=('F090W', 30.0), comp='sil', keep_narrow=False, 
                 agbcut=False, ignore_filts=None, rsgcat=None):
        
        self.gal = gal
        self.procdir = procdir
        os.makedirs(self.procdir, exist_ok=True)
        self.photfile_path = photfile_path
        self.backend_dir = os.path.join(self.procdir, 'backends')
        os.makedirs(self.backend_dir, exist_ok=True)
        self.logger = self.getlogger()

        if rsgcat is not None:
            if self.photfile_path is not None:
                try:
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
            if not os.path.exists(self.photfile_path):
                raise ValueError(f'photfile_path {self.photfile_path} does not exist')
            self.cat = self.read_cat(self.photfile_path)
            self.set_cols(self.cat)
            self.rsgcat = None

        self.dm, self.dmerr = dm, dmerr
        self.z = z
        self.modeltype = modeltype
        self.comp = comp
        self.agbcut = agbcut

        self.nrc_filts = np.array(['F070W','F090W','F115W','F140M','F150W', 'F150W2', 'F162M',
                                    'F164N','F182M','F187N','F200W','F210M','F212N','F250M',
                                    'F277W','F300M','F322W2','F323N','F335M','F356W','F360M',
                                    'F405N','F410M','F430M','F444W','F460M','F466N','F470N','F480M'])
        self.wv_all = [float(i.replace('F', '').replace('W2', '').replace('M', '').replace('N', '').replace('W', ''))/100 
                       for i in self.nrc_filts]
        self.gen_mc_obj = mcmc(dm=self.dm, dmerr=self.dmerr, z=self.z, model_type=self.modeltype, comp=self.comp)
        self.gen_mc_obj.verbose = False
        self.gen_mc_obj.dirs['backends'] = self.backend_dir
        self.keep_narrow = keep_narrow
        self.ignore_filts = ignore_filts
        self.trgb = trgb
        self.trgb = (int(np.where(self.nrc_filts==self.trgb[0].upper())[0][0]), self.trgb[1])
        self.chimin_params = {
            'teff_' : np.arange(2600, 5050, 50),
            'tdust_' : np.arange(200, 1800, 100),
            'tau_' : np.array(list(np.linspace(0.01, 2, 21)) + list(np.arange(2.5, 5.5, 0.5))),
            'Av_' : np.array(list(np.linspace(0, 1, 5))+ list(np.linspace(1.5, 3, 4)))
        }
        self.chimin_modeldir = os.path.join('data', 'chimin_models')
        
    def read_cat(self, photfile_path):
        catpath = os.path.join(self.procdir, 'proc')
        os.makedirs(catpath, exist_ok=True)
        if len(glob.glob(os.path.join(catpath, '*csv'))) == 0:
            save_photfiles(photfile_path, catpath)

        cat = None
        for fl in glob.glob(os.path.join(catpath, '*csv')):
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
        cat_wv = np.array([float(i[1:4])/100 for i in fls])
        flts = [i.replace('_mag','') for i in fls]
        magcols = [i+'_mag' for i in flts]
        errcols = [i+'_err' for i in flts]

        self.cols = {'flts' : np.array(flts),
                     'magcols' : np.array(magcols),
                     'errcols' : np.array(errcols),
                     'cat_wv': cat_wv}
    
    def getlogger(self, logfile=None):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(level=logging.INFO,
                            format='%(name)s [@ %(asctime)s] [l %(lineno)d] - %(levelname)s - %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename= logfile,
                            filemode='w')

        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger = logging.getLogger("rsg_sedfit")
        logger.addHandler(console)
        return logger

    def mp_init(init_success: int = 0,
                init_failed: int = 0,
                init_success_cols: list =[]):
        global success
        global failed
        global success_files
        success = init_success
        failed = init_failed
        success_files = init_success_cols

    def base_cuts(self, cat, min_det=4):
        def detmask(cat_mags_det, min_det=4):
            opt_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'] < 1.2)]]
            nir_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'] < 2.6)]]
            mir_dets = cat_mags_det[cat_mags_det.columns[(self.cols['cat_wv'] > 2.6)]]

            ndetm = cat_mags_det.sum(axis=1) >= min_det
            detm = (nir_dets.sum(axis=1) > 0) & (mir_dets.sum(axis=1) > 0) & ndetm

            return detm
        
        self.logger.info(f'Applying ndet cuts')
        self.gen_mc_obj.bounds['luminosity'] = [4.0, 4.1]

        base_models = np.zeros((2000, len(self.nrc_filts)))
        basedf_cols = [i+'_mag' for i in self.nrc_filts]

        sample_params = self.gen_mc_obj.get_init_pos(1000)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            base_models[i, :] = model_mag

        self.gen_mc_obj.bounds['luminosity'] = [5.8, 6]
        sample_params = self.gen_mc_obj.get_init_pos(1000)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            base_models[i+1000, :] = model_mag

        min_model = np.max(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        min_model_df = pd.DataFrame([dict(zip(basedf_cols, min_model))])

        max_model = np.min(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        max_model_df = pd.DataFrame([dict(zip(basedf_cols, max_model))])

        cat_mags = cat[self.cols['magcols']].replace(99.999, 0.5)
        mindf = cat_mags-min_model_df[self.cols['magcols']].to_numpy()
        cat_mags = cat_mags.replace(0.5, 99.999)
        maxdf = cat_mags-max_model_df[self.cols['magcols']].to_numpy()

        difm = (mindf < 0).all(axis=1) & (maxdf > 0).all(axis=1)
        ndet = (cat_mags > 10) & (cat_mags < 38) 
        ndetm = detmask(ndet, min_det=min_det)
        base_mask = difm & ndetm 
        rsgcat = cat[base_mask].replace(np.nan, 99.999)
        self.logger.info(f'RSG catalog contains {len(rsgcat)} objects after ndet cuts')

        return rsgcat
    
    def color_cuts(self, base_rsgcat, min_width=0.5, rel_err=0.25):
        self.logger.info(f'Applying color cuts')
        cat_mags = base_rsgcat[self.cols['magcols']]
        color_dict = {}
        cmb_iterator = list(itertools.combinations(self.nrc_filts, 2))
        idx_iterator = list(itertools.combinations(range(0, 29), 2))
        
        self.gen_mc_obj.reset_bounds()
        self.gen_mc_obj.bounds['luminosity'] = [3.5, 6.0]
        self.gen_mc_obj.bounds['temperature'] = [2600., 5000.]

        sample_models = np.zeros((10000, len(self.nrc_filts)))
        sample_params = self.gen_mc_obj.get_init_pos(10000)

        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            sample_models[i, :] = model_mag

        for i, j in zip(idx_iterator, cmb_iterator):
            f1, f2 = j[0], j[1]
            i1, i2 = i[0], i[1]
            index = f1+'_'+f2
            color_bounds = sample_models[:, i1] - sample_models[:, i2]
            cl_min = min(color_bounds) - rel_err 
            cl_max = max(color_bounds) + rel_err 
            cld = cl_max - cl_min
            if cld < min_width:
                cl_min = cl_min - (min_width - cld)/2
                cl_max = cl_max + (min_width - cld)/2
            color_dict[index] = (cl_min, cl_max)

        nwm = np.array(['N' in i for i in self.cols['magcols']])
        im = [i.split('_mag')[0].upper() in self.ignore_filts for i in sedfit.cols['magcols']]
        mfls = self.cols['magcols'][~(nwm|im)]

        colorm = np.array([True]*len(cat_mags))
        for i, j in itertools.combinations(mfls, 2):
            idx_ = i.replace('_mag', '')+'_'+j.replace('_mag', '')
            cat_color = cat_mags[i] - cat_mags[j]
            dm = (cat_mags[i] > 90) | (cat_mags[j] > 90)
            colm_ = (cat_color > color_dict[idx_][0]) & (cat_color < color_dict[idx_][1])
            colm_ = colm_ | dm
            colorm = colorm & colm_

        rsgcat = base_rsgcat[colorm]
        self.logger.info(f'RSG catalog contains {len(rsgcat)} objects after color cuts')
        return rsgcat
    
    def create_modeldf(self, outpath=None):
        if outpath is None:
            outpath = os.path.join(self.chimin_modeldir, f'{self.comp}_z{self.z:.2f}_modeldf.csv')

        # if modeldf for composition and metalllicity exists, read it
        if os.path.exists(outpath):
            modeldf = pd.read_csv(outpath)
        # else create a grid at 10 Mpc (needs to be done once)
        else:
            self.logger.info(f'Creating modeldf for comp={self.comp}, Z={self.z:.2f}: {outpath}')
            modeldf = pd.DataFrame(columns = ['Teff', 'Tdust', 'Tau', 'Av'] + list(self.nrc_filts))
            self.gen_mc_obj.reset_bounds()
            for a1 in tqdm(self.chimin_params['teff_']):
                for a2 in self.chimin_params['tdust_']:
                    for a3 in self.chimin_params['tau_']:
                        for a4 in self.chimin_params['Av_']:
                            pm_ = [a1, a2, a3, 1e3, 3.1, a4]
                            model_mag = np.array([self.gen_mc_obj.model[f](pm_).flatten()[0] for f in self.nrc_filts])
                            modeldf.loc[len(modeldf), ['Teff', 'Tdust', 'Tau', 'Av'] + list(self.nrc_filts)] = [a1, a2, a3, a4] + list(model_mag)

            modeldf.to_csv(outpath, index=False)

        # add distance to galaxy to modeldf
        magcols = ['F' in i for i in modeldf.columns]
        magcols = modeldf.columns[magcols]
        modeldf[magcols] = modeldf[magcols] + self.gen_mc_obj.dm
        return modeldf

    def gen_phot(self, col, noise_floor=0.01):
        phot = {'mag': np.array(col[self.cols['magcols']], dtype=float),
                'magerr': np.array(col[self.cols['errcols']], dtype=float),
                'inst_filt': np.array(self.cols['flts']),
                'index': f'{self.gal}_{int(col['index'])}'}

        if not self.keep_narrow:
            narrow_mask = np.array(['N' in i for i in phot['inst_filt']])
            ign_mask = np.array([i.upper() in self.ignore_filts for i in phot['inst_filt']])
            phot['mag'] = phot['mag'][~(narrow_mask|ign_mask)]
            phot['magerr'] = phot['magerr'][~(narrow_mask|ign_mask)]
            phot['inst_filt'] = phot['inst_filt'][~(narrow_mask|ign_mask)]
        else:
            ign_mask = np.array([i.upper() in self.ignore_filts for i in phot['inst_filt']])
            phot['mag'] = phot['mag'][~ign_mask]
            phot['magerr'] = phot['magerr'][~ign_mask]
            phot['inst_filt'] = phot['inst_filt'][~ign_mask]

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
        pm_ = [bestparams[0], bestparams[1], bestparams[2], 1e3, 3.1, bestparams[3]]
        bestmodel = np.array([self.gen_mc_obj.model[f](pm_).flatten()[0] for f in phot['inst_filt']]) + self.gen_mc_obj.dm 
        lum = 10**(np.mean((phot['mag'] - bestmodel)/-2.5)) * 1e3

        return chisq[minchisq], minchisq, lum      

    def chimin_cuts(self, base_rsgcat, modeldf):
        base_rsgcat.loc[:, ['chimin', 'teff_chisq', 'tdust_chisq', 'tau_chisq', 'av_chisq', 'lum_chisq']] = 0.0
        self.logger.info(f'Applying chisq cuts')
        for idx in tqdm(base_rsgcat.index):
            testcol = base_rsgcat.loc[idx]
            phot = self.gen_phot(testcol)

            c_, m_, l_ = self.chimin(phot, modeldf)
            base_rsgcat.loc[idx, ['chimin', 'lum_chisq']] = c_, l_
            base_rsgcat.loc[idx, ['teff_chisq', 'tdust_chisq', 'tau_chisq', 'av_chisq']] = modeldf.loc[m_, ['Teff', 'Tdust', 'Tau', 'Av']].values

        chi_cut = np.percentile(base_rsgcat['chimin'], 75)
        if self.agbcut:
            tm = (base_rsgcat['teff_chisq'] > 3300) & (base_rsgcat['teff_chisq'] < 4700)
            tum = base_rsgcat['tau_chisq'].values > 1
            lm = np.log10(base_rsgcat['lum_chisq'].values) > 4.5
            cm = base_rsgcat['chimin'] < chi_cut
            mask = cm & (tm | tum | lm)
        else:
            mask = base_rsgcat['chimin'] < chi_cut

        rsgcat = base_rsgcat[mask]
        self.logger.info(f'RSG catalog contains {len(rsgcat)} objects after chisq cuts')
        return rsgcat
    
    def plot_error_dist(self, cat=None):
        if cat is None:
            cat = self.rsgcat

        errcols = self.cols['errcols']
        err_df = cat[self.cols['errcols']].replace({9.999: np.nan, 99.999: np.nan})

        data = [err_df[c].dropna().values for c in errcols]
        labels = [c.replace('_err','') for c in errcols]

        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)

        # styling
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
    
    def apply_initial_cuts(self, colcuts=False, outpath=None, min_det=4):
        if not self.agbcut:
            self.logger.info(f'WARNING: AGB cut set to {self.agbcut}')
        base_rsgcat = self.base_cuts(self.cat, min_det=min_det)
        if colcuts:
            base_rsgcat = self.color_cuts(base_rsgcat)

        modeldf = self.create_modeldf(outpath=outpath)
        rsgcat = self.chimin_cuts(base_rsgcat, modeldf)
        rsgcat.to_csv(os.path.join(self.procdir, f'{self.gal}_{self.comp}_rsgcat.csv'))
        self.rsgcat = rsgcat

class mcmcfit(object):
    def __init__(self, rsg_dataloader: rsg_dataloader, ncores=1, keep_narrow=False, ignore_filts=None, 
                 modeltype='MARCS', comp='sil', redo_mcmc=False, verbose=False):
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

        self.mc_obj = mcmc(dm=self.rsgloader.dm, dmerr=self.rsgloader.dmerr, z=self.rsgloader.z, 
                           model_type=self.modeltype, comp=self.comp)
        self.mc_obj.verbose=verbose
        if os.path.exists(self.rsgloader.backend_dir):
            self.mc_obj.dirs['backends'] = self.rsgloader.backend_dir
        else:
            self.mc_obj.dirs['backends'] = 'data/backends'

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

        fig = corner.corner(sample, labels=labels, color='C1', hist_kwargs={'density': True},
                            label_kwargs={'fontsize': 10}, show_titles=False, title_fmt='.3f', 
                            plot_datapoints=False, fill_contours=True, truth_color='k', truths=truths)
        
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax_diag = axes[i, i]
            ax_diag.axvline(medians[i], color='k', lw=1)
            ax_diag.axvline(p16[i], color='k', lw=1, ls='--')
            ax_diag.axvline(p84[i], color='k', lw=1, ls='--')
            ax_diag.set_title(r"${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$".format(medians[i], p84[i]-medians[i], medians[i]-p16[i]), fontsize=9)

            for j in range(i):
                ax = axes[i, j]
                ax.scatter(medians[j], medians[i], marker='x', color='k', s=30)

        fig.suptitle(phot['index'], fontsize=12)
    
    def plot_fit(self, phot, fit_params, min_chi_params, chi_post, chi_best, save=False):
        fit_pe = np.array(list(fit_params.values()))
        params = np.meshgrid(*fit_pe[:, 0], indexing='ij', sparse=True)
        model_mag = np.array([self.mc_obj.model[f](params).flatten()[0] for f in phot['inst_filt']]) + self.mc_obj.dm
        model_mag_plot = np.array([self.mc_obj.model[f](params).flatten()[0] for f in self.rsgloader.nrc_filts]) + self.mc_obj.dm

        mc_params = np.meshgrid(*min_chi_params, indexing='ij', sparse=True)
        chi_mag = np.array([self.mc_obj.model[f](mc_params).flatten()[0] for f in phot['inst_filt']]) + self.mc_obj.dm
        chi_mag_plot = np.array([self.mc_obj.model[f](mc_params).flatten()[0] for f in self.rsgloader.nrc_filts]) + self.mc_obj.dm

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0.1)
        wv = np.array([float(i[1:4])/100 for i in phot['inst_filt']])
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
            self.plot_corner(phot=phot, min_chi_params=min_chi_params)
        
        return fit_params, min_chi_params, chi_posterior, chi_best

    def run_mcmc_parallel(self):
        for p_ in self.mcfit_params:
            self.rsgcat.loc[:, [p_+'_mc', p_+'_elow', p_+'_eup', p_+'_best']] = np.nan
        self.rsgcat.loc[:, ['chi_posterior', 'chi_best']] = np.nan

        argument_list = []
        for idx_ in self.rsgcat.index:
            if self.redo_mcmc or not os.path.exists(os.path.join(self.rsgloader.backend_dir, self.rsgloader.gal.upper()+'_'+str(int(idx_))+'_'+self.rsgloader.modeltype+'.h5')): 
                argument_list.append([self.rsgcat.loc[idx_]])
            else:
                continue

        # create argument list for starmap async
        multiprocessing_logging.install_mp_handler(self.rsgloader.logger)
        p = Pool(initializer=self.mp_init, processes=self.ncores)
        result = p.starmap_async(self.parallel_mc_worker, argument_list)

        # run MCMC
        self.logger.info(f"Starting MCMC")
        result.get()

        # read backend file for each object to calculate best fit parameters and write to rsgcat
        for idx_ in self.rsgcat.index:
            self.read_mc_params(self.rsgcat.loc[idx_])

        self.rsgcat.to_csv(os.path.join(self.rsgloader.procdir, f'rsgcat_{self.rsgloader.gal}_MCMC_{self.rsgloader.modeltype}.csv'), index=False)
    
if __name__=='__main__':
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
    sedfit = mcmcfit(rsgloader, ncores=args.ncores, verbose=False, 
                     modeltype=args.modeltype, comp=args.comp, redo_mcmc=args.redo)
    sedfit.run_mcmc_parallel()