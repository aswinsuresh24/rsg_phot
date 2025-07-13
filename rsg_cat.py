import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob, os
import itertools

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
        flt_keys = ['Counts', 'Err', 'Flag']
        flt_dict = {key: [] for key in flt_keys}
        idx_cts, idx_err, idx_flag = [], [], []
        for line in lines:
            #get column indices for counts, errors and flags
            #count uncertainty is used to get the index since 'Normalized count rate' is not unique
            if (filter_ in line) & ('Normalized count rate uncertainty' in line):
                idx = int(line.split(' ')[0].split('.')[0]) - 1
                idx_cts.append(str(idx - 1))
                idx_err.append(str(idx))
                idx_flag.append(str(idx + 9))
        #first index corresponds to combined photometry
        flt_dict['Counts'] = idx_cts[1:]
        flt_dict['Err'] = idx_err[1:]
        flt_dict['Flag'] = idx_flag[1:]
        filt_dict[filter_] = flt_dict
    
    return col_dict, filters, filt_dict

def save_photfiles(photfile_path, outdir, chunksize = 100000):
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

    Returns
    -------
    None
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
  
    photfiles = sorted(glob.glob(os.path.join(photfile_path, '*phot')))
    for i, photfile in enumerate(photfiles):
        column_file = photfile + '.columns'
        #map columns to indices
        col_idx, filters, _ = map_columns(column_file)

        #read photometry file in chunks
        photdf = pd.read_csv(photfile, sep = '\s+', memory_map = True, 
                             header = None, iterator = True, chunksize = chunksize)
        
        for j, _df in tqdm(enumerate(photdf)):
            #apply cuts
            cuts = (_df[col_idx['SNR']] >= 5) & \
                    ((_df[col_idx['Sharpness']])**2 <= 0.04) & \
                    (_df[col_idx['Crowding']] <= 1.5) & \
                    (_df[col_idx['Type']] <= 2)
            _df = _df[cuts]
            _df = _df[list(col_idx.values())]
            _df.columns = list(col_idx.keys())
            _df.to_csv(f"{outdir}/{os.path.basename(photfile).split('.')[0]}_{j}.csv", 
                       mode = 'a', header = True, index = False)
            
