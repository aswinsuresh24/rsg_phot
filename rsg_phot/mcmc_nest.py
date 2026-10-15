import numpy as np
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob
import emcee
import rsg_phot.dust
import progressbar
import sys
import astropy.units as u
import astropy.constants as const
import traceback
import math
import pickle
import random
from astropy.stats import sigma_clipped_stats as scs
import time
from scipy.stats import truncexpon
from pathlib import Path

NRC_FILTS = np.array(['F070W','F090W','F115W','F140M','F150W','F150W2','F162M',
                      'F164N','F182M','F187N','F200W','F210M','F212N','F250M',
                      'F277W','F300M','F322W2','F323N','F335M','F356W','F360M',
                      'F405N','F410M','F430M','F444W','F460M','F466N','F470N','F480M'])
MIRI_FILTS = np.array(['F0560W','F0770W','F1000W','F1130W','F1280W','F1500W',
                       'F1800W','F2100W','F2550W'])

class mcmc_nest(object):
    def __init__(self):
        pass