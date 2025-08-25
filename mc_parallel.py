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

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Fit RSG NIRCam SEDs')
    parser.add_argument('--gal', type=str, default='gal', help='Galaxy name')
    parser.add_argument('--photfile_path', type=str, default='.', help='Root directory to search for photometry')
    parser.add_argument('--dm', type=float, default=30, help='Distance modulus')
    parser.add_argument('--dmerr', type=float, default=0.5, help='Distance modulus error')
    parser.add_argument('--z', type=float, default=0.0, help='Metallicity')
    parser.add_argument('--keep_narrow', type=bool, default=False, help='Fit narrow band photometry?')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')

    return parser


class parallel_sed_fit(object):
    def __init__(self, gal, photfile_path, dm=30, dmerr=0.5, z=0.0,
                 keep_narrow=False, ncores=10):
        self.gal = gal
        self.photfile_path = photfile_path
        self.rsgcat = self.read_cat(self.photfile_path)

        self.nrc_filts = ['F070W','F090W','F115W','F140M','F150W', 'F150W2', 'F162M',
                        'F164N','F182M','F187N','F200W','F210M','F212N','F250M',
                        'F277W','F300M','F322W2','F323N','F335M','F356W','F360M',
                        'F405N','F410M','F430M','F444W','F460M','F466N','F470N','F480M']
        self.wv_all = [float(i.replace('F', '').replace('W2', '').replace('M', '').replace('N', '').replace('W', ''))/100 
                       for i in self.nrc_filts]
        self.gen_mc_obj = mcmc(dm=dm, dmerr=dmerr, z=z)
        self.gen_mc_obj.verbose = True
        self.keep_narrow = keep_narrow
        self.dirs = {}
        
    def read_cat(self, photfile_path):
        catpath = os.path.join(photfile_path, 'proc')
        os.makedirs(catpath, exist_ok=True)
        if len(glob.glob(os.path.join(catpath, '*csv'))) == 0:
            save_photfiles(photfile_path, catpath)

        for fl in glob.glob(os.path.join(catpath, '*csv')):
            if cat is None:
                cat = pd.read_csv(fl)
            else:
                df_ = pd.read_csv(fl)
                cat = pd.concat([cat, df_])
        cat.reset_index(inplace=True)
        cat['index'] = cat.index

        return cat

    def base_cuts(self):
        raise(NotImplementedError)
    
    def color_cuts(self, base_rsgcat):
        raise(NotImplementedError)
    
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

    def chimin_cuts(self, base_rsgcat, cols, modeldf):
        base_rsgcat[['chimin', 'teff', 'tdust', 'tau', 'av', 'lum']] = np.nan
        for idx in tqdm(base_rsgcat.index):
            testcol = base_rsgcat.loc[idx]

            phot = {'mag': testcol[cols['magcols']].values,
                    'magerr': testcol[cols['errcols']].values,
                    'inst_filt': np.array(cols['flts']),
                    'index': f'{self.gal}_{int(testcol['index'])}'}

            if not self.keep_narrow:
                    fl_mask = np.array(['N' in i for i in phot['inst_filt']]) | np.array(['300M' in i for i in phot['inst_filt']])
                    phot['mag'] = phot['mag'][~fl_mask]
                    
                    phot['magerr'] = phot['magerr'][~fl_mask]
                    phot['inst_filt'] = phot['inst_filt'][~fl_mask]

            limmask = (phot['mag'] > 90.0) | (phot['magerr'] > 90.0) | (np.isnan(phot['mag'])) | (np.isnan(phot['magerr']))
            phot['mag'] = phot['mag'][~limmask]
            phot['magerr'] = phot['magerr'][~limmask]
            phot['inst_filt'] = phot['inst_filt'][~limmask]
            c_, m_, l_ = self.chimin(phot, modeldf)
            base_rsgcat.loc[idx, ['chimin', 'lum']] = c_, l_
            base_rsgcat.loc[idx, ['teff', 'tdust', 'tau', 'av']] = modeldf.loc[m_, ['Teff', 'Tdust', 'Tau', 'Av']].values

            chi_cut = np.median(base_rsgcat['chimin']) + 1*np.std(base_rsgcat['chimin'], ddof=1)
            agb_cut = (base_rsgcat['teff'] <= 3300) | (base_rsgcat['teff'] >= 4700) | (base_rsgcat['chimin'] > chi_cut)

            rsgcat = base_rsgcat[~agb_cut]
        return rsgcat
    
    def all_cuts(self, rsgcat):
        fls = ['mag' in i for i in rsgcat.columns]
        fls = rsgcat.columns[fls]
        cat_mags = rsgcat[fls]
        cat_wv = np.array([float(i[1:4])/100 for i in cat_mags.columns])
        cat_mags = cat_mags.replace(99.999, 0.5)
        cat_mags = cat_mags.replace(np.nan, 0.5)
        flts = [i.replace('_mag','') for i in fls]
        magcols = [i+'_mag' for i in flts]
        errcols = [i+'_err' for i in flts]

        cols = {'flts' : flts,
                'magcols' : magcols,
                'errcols' : errcols,
                'cat_wv': cat_wv}

        base_rsgcat = self.base_cuts()
        base_rsgcat = self.color_cuts(base_rsgcat)

        rsgcat = self.chimin_cuts(base_rsgcat, cols, None) #create modeldf
    
    def parallel_mc_worker(self, phot):
        raise(NotImplementedError)
    
    def run_sed_fit(self, rsgcat):
        raise(NotImplementedError)
    
if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()

    sedfit = parallel_sed_fit(args.gal, args.photfile_path, args.dm, args.dmerr, args.z, args.keep_narrow, args.ncores)
