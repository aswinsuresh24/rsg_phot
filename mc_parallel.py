import numpy as np
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
import pickle
from astropy.io import fits, ascii
from scipy import interpolate
from scipy.integrate import simpson
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io.misc.hdf5 import read_table_hdf5
import pandas as pd
import itertools
from tqdm import tqdm
from rsg_cat import save_photfiles
from mcmc import mcmc
from multiprocessing import pool
import argparse
import logging
import warnings
warnings.simplefilter('ignore')

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Fit red supergiant SEDs')
    parser.add_argument('--gal', type=str, default='gal', help='Galaxy name')
    parser.add_argument('--photfile_path', type=str, default='.', help='Root directory to search for photometry')
    parser.add_argument('--dm', type=float, default=30, help='Distance modulus')
    parser.add_argument('--dmerr', type=float, default=0.5, help='Distance modulus error')
    parser.add_argument('--z', type=float, default=0.0, help='Metallicity')
    parser.add_argument('--trgb', type=tuple, default=('F090W', 30), help='Tip of red giant branch')
    parser.add_argument('--keep_narrow', type=bool, default=False, help='Fit narrow band photometry?')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    parser.add_argument('--ignore_filts', type=list, default=[], help='Photometry to avoid fitting')

    return parser


class parallel_sed_fit(object):
    def __init__(self, gal, photfile_path, dm=30, dmerr=0.5, z=0.0, trgb=('F090W', 30),
                 keep_narrow=False, ncores=10, ignore_filts=[]):
        self.gal = gal
        self.photfile_path = photfile_path
        self.cat = self.read_cat(self.photfile_path)

        self.nrc_filts = np.array(['F070W','F090W','F115W','F140M','F150W', 'F150W2', 'F162M',
                                    'F164N','F182M','F187N','F200W','F210M','F212N','F250M',
                                    'F277W','F300M','F322W2','F323N','F335M','F356W','F360M',
                                    'F405N','F410M','F430M','F444W','F460M','F466N','F470N','F480M'])
        self.wv_all = [float(i.replace('F', '').replace('W2', '').replace('M', '').replace('N', '').replace('W', ''))/100 
                       for i in self.nrc_filts]
        self.gen_mc_obj = mcmc(dm=dm, dmerr=dmerr, z=z)
        self.gen_mc_obj.verbose = True
        self.keep_narrow = keep_narrow
        self.trgb = trgb
        self.trgb = (int(np.where(self.nrc_filts==self.trgb[0].upper())[0][0]), self.trgb[1])
        self.chimin_params = {
            'teff_' : np.arange(2600, 5050, 50),
            'tdust_' : np.arange(200, 1800, 100),
            'tau_' : np.array(list(np.linspace(0.01, 2, 21)) + list(np.arange(2.5, 5.5, 0.5))),
            'Av_' : np.array(list(np.linspace(0, 1, 5))+ list(np.linspace(1.5, 3, 4)))
        }
        
    def read_cat(self, photfile_path):
        catpath = os.path.join(photfile_path, 'proc')
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

        return cat

    def base_cuts(self, cat, fls, min_det=4):
        def detmask(cat_mags_det, min_det=4):
            cat_wv = np.array([float(i[1:4])/100 for i in cat_mags_det.columns])

            opt_dets = cat_mags_det[cat_mags_det.columns[(cat_wv < 1.2)]]
            nir_dets = cat_mags_det[cat_mags_det.columns[(cat_wv > 1.2) & (cat_wv < 2.6)]]
            mir_dets = cat_mags_det[cat_mags_det.columns[(cat_wv > 2.6)]]

            ndetm = cat_mags_det.sum(axis=1) > min_det
            detm = (opt_dets.sum(axis=1) > 0) & (nir_dets.sum(axis=1) > 1) & (mir_dets.sum(axis=1) > 1) & ndetm

            return detm
        
        self.gen_mc_obj.bounds['luminosity'] = [10**3.5, 10**3.6]

        base_models = np.zeros((2000, len(self.nrc_filts)))
        basedf_cols = [i+'_mag' for i in self.nrc_filts]

        sample_params = self.gen_mc_obj.get_init_pos(6, 1000)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            base_models[i, :] = model_mag

        self.gen_mc_obj.bounds['luminosity'] = [10**5.8, 10**6]
        sample_params = self.gen_mc_obj.get_init_pos(6, 1000)
        for i, p_ in enumerate(sample_params):
            model_mag = np.array([self.gen_mc_obj.model[f](sample_params[i]).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
            base_models[i+1000, :] = model_mag

        min_model = np.max(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        min_model_df = pd.DataFrame([dict(zip(basedf_cols, min_model))])

        max_model = np.min(base_models[base_models[:, self.trgb[0]] < self.trgb[1]], axis=0)
        max_model_df = pd.DataFrame([dict(zip(basedf_cols, max_model))])

        cat_mags = cat[fls].replace(99.999, 0.5)
        mindf = cat_mags-min_model_df[fls].to_numpy()
        cat_mags = cat_mags.replace(0.5, 99.999)
        maxdf = cat_mags-max_model_df[fls].to_numpy()

        difm = (mindf < 0).all(axis=1) & (maxdf > 0).all(axis=1)
        ndet = (cat_mags > 10) & (cat_mags < 38) 
        ndetm = detmask(ndet, min_det=min_det)
        base_mask = difm & ndetm 
        rsgcat = cat[base_mask].replace(np.nan, 99.999)

        return rsgcat
    
    def color_cuts(self, base_rsgcat, fls, min_width=0.5, rel_err=0.25):
        cat_mags = base_rsgcat[fls]
        color_dict = {}
        cmb_iterator = list(itertools.combinations(self.nrc_filts, 2))
        idx_iterator = list(itertools.combinations(range(0, 29), 2))
        
        self.gen_mc_obj.reset_bounds()
        self.gen_mc_obj.bounds['luminosity'] = [10**3.5, 10**6.0]
        self.gen_mc_obj.bounds['temperature'] = [3500., 5000.]

        sample_models = np.zeros((10000, len(self.nrc_filts)))
        sample_params = self.gen_mc_obj.get_init_pos(6, 10000)

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

        nwm = np.array(['N' in i for i in fls]) | np.array(['300M' in i for i in fls])
        mfls = fls[~nwm]

        colorm = np.array([True]*len(cat_mags))
        for i, j in itertools.combinations(mfls, 2):
            idx_ = i.replace('_mag', '')+'_'+j.replace('_mag', '')
            cat_color = cat_mags[i] - cat_mags[j]
            dm = (cat_mags[i] > 90) | (cat_mags[j] > 90)
            colm_ = (cat_color > color_dict[idx_][0]) & (cat_color < color_dict[idx_][1])
            colm_ = colm_ | dm
            colorm = colorm & colm_

        rsgcat = base_rsgcat[colorm]
        return rsgcat
    
    def create_modeldf(self):
        outpath = os.path.join(self.photfile_path, f'{self.gal}_modeldf.csv')
        if os.path.exists(outpath):
            modeldf = pd.read_csv(outpath)
        else:
            modeldf = pd.DataFrame(columns = ['Teff', 'Tdust', 'Tau', 'Av'] + list(self.nrc_filts))
            self.gen_mc_obj.reset_bounds()
            for a1 in tqdm(self.chimin_params['teff_']):
                for a2 in self.chimin_params['tdust_']:
                    for a3 in self.chimin_params['tau_']:
                        for a4 in self.chimin_params['Av_']:
                            pm_ = [a1, a2, a3, 1e3, 3.1, a4]
                            model_mag = np.array([self.gen_mc_obj.model[f](pm_).flatten()[0] for f in self.nrc_filts]) + self.gen_mc_obj.dm
                            modeldf.loc[len(modeldf), ['Teff', 'Tdust', 'Tau', 'Av'] + list(self.nrc_filts)] = [a1, a2, a3, a4] + list(model_mag)

            modeldf.to_csv(outpath, index=False)
        return modeldf
    
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
        base_rsgcat.loc[:, ['chimin', 'teff', 'tdust', 'tau', 'av', 'lum']] = 0.0
        for idx in tqdm(base_rsgcat.index):
            testcol = base_rsgcat.loc[idx]

            if self.cols==None:
                raise ValueError('mag and error column information missing, set self.cols')

            phot = {'mag': testcol[self.cols['magcols']].values,
                    'magerr': testcol[self.cols['errcols']].values,
                    'inst_filt': np.array(self.cols['flts']),
                    'index': f'{self.gal}_{int(testcol['index'])}'}

            if not self.keep_narrow:
                    fl_mask = np.array(['N' in i for i in phot['inst_filt']]) | np.array(['300M' in i for i in phot['inst_filt']])
                    phot['mag'] = phot['mag'][~fl_mask]
                    
                    phot['magerr'] = phot['magerr'][~fl_mask]
                    phot['inst_filt'] = phot['inst_filt'][~fl_mask]

            limmask = (phot['mag'] > 90.0) | (phot['magerr'] > 90.0) | (np.isnan(phot['mag'])) | (np.isnan(phot['magerr'])) | (phot['magerr'] < 1e-4)
            phot['mag'] = phot['mag'][~limmask]
            phot['magerr'] = phot['magerr'][~limmask]
            phot['inst_filt'] = phot['inst_filt'][~limmask]

            c_, m_, l_ = self.chimin(phot, modeldf)
            base_rsgcat.loc[idx, ['chimin', 'lum']] = c_, l_
            base_rsgcat.loc[idx, ['teff', 'tdust', 'tau', 'av']] = modeldf.loc[m_, ['Teff', 'Tdust', 'Tau', 'Av']].values

        chi_cut = 2*np.nanmedian(base_rsgcat['chimin'])
        agb_cut = (base_rsgcat['teff'] <= 3300) | (base_rsgcat['teff'] >= 4700) | (base_rsgcat['chimin'] > chi_cut)

        rsgcat = base_rsgcat[~agb_cut]
        return rsgcat
    
    def apply_initial_cuts(self):
        fls = ['mag' in i for i in self.cat.columns]
        fls = self.cat.columns[fls]
        cat_mags = self.cat[fls]
        cat_wv = np.array([float(i[1:4])/100 for i in cat_mags.columns])
        cat_mags = cat_mags.replace(np.nan, 99.999)
        flts = [i.replace('_mag','') for i in fls]
        magcols = [i+'_mag' for i in flts]
        errcols = [i+'_err' for i in flts]

        self.cols = {'flts' : flts,
                     'magcols' : magcols,
                     'errcols' : errcols,
                     'cat_wv': cat_wv}

        base_rsgcat = self.base_cuts(self.cat, fls)
        base_rsgcat = self.color_cuts(base_rsgcat, fls)

        modeldf = self.create_modeldf()
        rsgcat = self.chimin_cuts(base_rsgcat, modeldf)
        self.rsgcat = rsgcat
        return rsgcat

    def parallel_mc_worker(self, phot):
        raise(NotImplementedError)
    
    def run_sed_fit(self, rsgcat):
        raise(NotImplementedError)
    
if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()

    sedfit = parallel_sed_fit(gal=args.gal, 
                              photfile_path=args.photfile_path, 
                              dm=args.dm, 
                              dmerr=args.dmerr, 
                              z=args.z, 
                              trgb=args.trgb,
                              keep_narrow=args.keep_narrow, 
                              ncores=args.ncores,
                              ignore_filts=args.ignore_filts)
    rsgcat = sedfit.apply_initial_cuts()