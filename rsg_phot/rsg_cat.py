import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob, os
import itertools
from astropy import wcs
from astropy.io import fits
from dustmaps.sfd import SFDQuery
from dust_extinction.parameter_averages import CCM89, G23
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_filters(columns):
    """
    get unique filters and lines from dolphot column files

    Parameters
    ----------
    columns : str
        path to dolphot column file

    Returns
    -------
    lines : list
        list of lines from dolphot column file
    filters : list
        list of unique filters
    """
    with open(columns, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines)

    filters = []
    for line in lines:
        if 'Normalized count rate, NIRCAM' in line:
            filters.append(line.split('NIRCAM_')[-1].split('\n')[0])

    return lines, filters

def map_columns(columns):
    """
    generate dictionaries mapping column names to column indices from dolphot column files

    Parameters
    ----------
    columns : str
        path to dolphot column file

    Returns
    -------
    col_dict : dict
        dictionary mapping column names to column indices for combined photometry
    filt_dict : dict
        dictionary mapping filter names to counts, error and flag column indices for
        individual images
    """
    #combined photometry columns
    column_strings = ['Object X position', 'Object Y position', 'Signal-to-noise', 
                      'Object sharpness', 'Crowding', 'Object type']
    column_key = ['X', 'Y', 'SNR', 'Sharpness', 'Crowding', 'Type']
    
    #get unique filters and lines from column file
    lines, filters = get_filters(columns)

    #define keys and strings for instrumental magnitudes and uncertainties
    mag_strings = [f'Instrumental ABMAG magnitude, NIRCAM_{filt}' for filt in filters]
    magerr_strings = [f'Magnitude uncertainty, NIRCAM_{filt}' for filt in filters]
    mag_key, magerr_key = [i + '_mag' for i in filters], [i + '_err' for i in filters]

    keys = column_key+mag_key+magerr_key
    strings = column_strings+mag_strings+magerr_strings
    
    col_dict = {key: [] for key in keys}
    
    #get column indices for combined photometry
    for key, string in zip(keys, strings):
        col = lines[np.char.find(lines, string) > 0]
        if len(col) > 0:
            col_dict[key] = int(col[0].split('.')[0]) - 1
        else:
            col_dict[key] = None

    #get column indices for individual images
    filt_dict = {key: [] for key in filters}
    for filter_ in filters:
        flt_keys = ['Counts', 'Err', 'Flag', 'LC', 'LC_err', 'Img']
        flt_dict = {key: [] for key in flt_keys}
        idx_cts, idx_err, idx_flag = [], [], []
        idx_lc, idx_img, idx_lcerr = [], [], []
        for line in lines:
            #get column indices for counts, errors and flags
            #count uncertainty is used to get the index since 'Normalized count rate' is not unique
            if (filter_ in line) & ('Normalized count rate uncertainty' in line):
                idx = int(line.split(' ')[0].split('.')[0]) - 1
                idx_cts.append(str(idx - 1))
                idx_err.append(str(idx))
                idx_flag.append(str(idx + 9))
            if (filter_ in line) & ('Instrumental VEGAMAG magnitude' in line):
                idx = int(line.split(' ')[0].split('.')[0]) - 1
                idx_lc.append(str(idx))
                idx_lcerr.append(str(idx + 2))
                mask = ['jhat' in i for i in line.split(' ')]
                imgname = np.array(line.split(' '))[mask][0]
                idx_img.append(imgname)
        #first index corresponds to combined photometry
        flt_dict['Counts'] = idx_cts[1:]
        flt_dict['Err'] = idx_err[1:]
        flt_dict['Flag'] = idx_flag[1:]
        flt_dict['LC'] = idx_lc
        flt_dict['LC_err'] = idx_lcerr
        flt_dict['Img'] = idx_img
        filt_dict[filter_] = flt_dict
    
    return col_dict, filters, filt_dict

def mw_extinction(df, filters, verbose=False):
    """
    apply milky way extinction correction to photometry
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing photometry (must contain RA, Dec columns)
    filters : list
        list of unique filters
    verbose : bool
        print median A_lambda for each filter   

    Returns
    -------
    df : pandas.DataFrame
        dataframe with Milky Way extinction corrected magnitudes
    """
    ra, dec = df['RA'], df['Dec']
    coord = SkyCoord(ra=ra, dec=dec, unit = (u.hourangle, u.deg), frame='icrs')

    #query SFD dust map for E(B-V)
    sfd = SFDQuery()
    ebv = sfd(coord)
    R_V = 3.1

    #use extinction law from Gordon+23
    ext = G23(Rv = R_V)
    for flt_ in filters:
        wv = float(flt_[1:4])/100*u.um
        a_lambda = ext(wv)
        A_lambda = a_lambda * ebv * R_V
        if verbose:
            print(f'Median A_lambda in {flt_} is {np.median(A_lambda)}')
        mask_missing = (df[flt_ + '_mag'] > 90.0) | (df[flt_ + '_mag'].isna().values)
        df.loc[~mask_missing, flt_+'_mag'] = df.loc[~mask_missing, flt_+'_mag'] + A_lambda[~mask_missing]

    return df

def save_photfiles(photfile_path, outdir, chunksize=100000, lc=False):
    """
    save photometry files with cuts applied to smaller csv files

    Parameters
    ----------
    photfile_path : str
        path to directory containing dolphot photometry files
    outdir : str
        path to directory to save csv files
    obj : str
        object name
    chunksize : int
        number of rows to read from photometry file at a time
    lc : bool
        keep columns corresponding to individual image photometry

    Returns
    -------
    None
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
  
    photfiles = sorted(glob.glob(os.path.join(photfile_path, '*', '*phot')))
    refimgs = sorted(glob.glob(os.path.join(photfile_path, '*', '*i2d.fits')))
    for i, (photfile, refimg) in enumerate(zip(photfiles, refimgs)):
        column_file = photfile + '.columns'
        #map columns to indices
        col_idx, filters, _ = map_columns(column_file)
        gid = '_'.join(os.path.basename(photfile).split('.phot')[0].split('_')[1:])
        refwcs = wcs.WCS(fits.open(refimg)[1].header)

        #read photometry file in chunks
        photdf = pd.read_csv(photfile, sep = r'\s+', memory_map = True, 
                             header = None, iterator = True, chunksize = chunksize)
        
        for j, _df in tqdm(enumerate(photdf)):
            #apply cuts
            cuts = (_df[col_idx['SNR']] >= 5) & \
                    ((_df[col_idx['Sharpness']])**2 <= 0.04) & \
                    (_df[col_idx['Crowding']] <= 1.5) & \
                    (_df[col_idx['Type']] <= 2)
            _df = _df[cuts].copy()

            _ra, _dec = refwcs.all_pix2world(_df[col_idx['X']], _df[col_idx['Y']], 0)
            _df.loc[:, 'RA'] = _ra
            _df.loc[:, 'Dec'] = _dec
            _df.loc[:, 'gid'] = gid
            _df.loc[:, 'id'] = np.array(_df.index)

            if lc:
                pass
                #BUG: add mw extinction correction to lc mode
            else:
                _df = _df[list(col_idx.values()) + ['RA', 'Dec', 'gid', 'id']]
                _df.columns = list(col_idx.keys()) + ['RA', 'Dec', 'gid', 'id']
                _df = mw_extinction(_df, filters, verbose=False)

            _df.to_csv(f"{outdir}/{os.path.basename(photfile).split('.')[0]}_{j}.csv", 
                       mode = 'a', header = True, index = False)