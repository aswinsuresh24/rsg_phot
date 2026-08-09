'''
TEMPORARY script.

Rebuilds the DUSTY .hdf5 tables that were written with only 27 tau values because
`dusty_gen.dusty_n_taugrid` stayed at its hardcoded default of 27 while taugrid.dat
actually held 42 taus.

This repeats *only* the post-processing part of dust.py (lines 479-512): read the
DUSTY .out results table and the .s### spectra, build the QTable, scale the spectra
to the grid luminosity, and write the .hdf5.  DUSTY is not re-run and input spectra
are not regenerated - the existing .s### files already contain all 42 taus.

Usage
-----
python fix_missing_tau.py --griddir data/dusty_sil_grid --comp sil
python fix_missing_tau.py --griddir data/dusty_sil_grid --comp sil --dry-run
'''

import os
import glob
import argparse

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.io import ascii
from scipy.integrate import simpson

N_TAUGRID = 42


def create_parser():
    parser = argparse.ArgumentParser(description='Rewrite DUSTY hdf5 tables with all tau values')
    parser.add_argument('--griddir', type=str, default='data/dusty_sil_grid',
                        help='Grid root directory (the one containing dusty_<comp>_out)')
    parser.add_argument('--comp', type=str, default='sil', help='Dust composition (sil or grf)')
    parser.add_argument('--lum', type=float, default=4.0,
                        help='log(L/Lsun) the spectra are scaled to (4.0 in gen_grid)')
    parser.add_argument('--dist', type=float, default=10.0, help='Distance in Mpc')
    parser.add_argument('--n_tau', type=int, default=N_TAUGRID, help='Number of tau values in taugrid.dat')
    parser.add_argument('--skiprows', type=int, default=None,
                        help='Header rows to skip in the .out file (default: auto-detect)')
    parser.add_argument('--dry-run', default=False, action=argparse.BooleanOptionalAction,
                        help='Report what would be rewritten without touching any file')
    return parser


def find_table_start(outfile):
    '''
    Locate the first data row of the RESULTS table in a DUSTY .out file.

    dust.py assumes skiprows=42, but the header length varies between runs, so the
    offset is found from the '###   tau0 ...' header instead.
    '''
    with open(outfile, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('###') and 'tau0' in line:
            # '###  tau0 ...', '###  1  2 ...', '=====' , then the data rows
            return i + 3
    raise ValueError(f'Could not find the results table in {outfile}')


def scale_flux(wv, flux, scale, flux_scale):
    # Renormalize the RSG spectrum so it's in units of erg/s/cm2/angstrom for
    # synphot to interpret
    normalize = simpson(flux, x=wv.to(u.um).value)
    flux = scale.value * flux_scale * flux / normalize / u.um
    flux = flux.to(u.erg / u.s / u.cm**2 / u.Angstrom)

    return flux


def rebuild_table(outdir, basename, lum, flux_scale, n_tau=N_TAUGRID, skiprows=None, dry_run=False):
    outfile = os.path.join(outdir, basename + '.out')

    # read outfile and spectra files
    if skiprows is None:
        skiprows = find_table_start(outfile)
    outfile_rows = np.loadtxt(outfile, skiprows=skiprows, max_rows=n_tau)
    if outfile_rows.shape[0] != n_tau:
        raise ValueError(f'{outfile}: read {outfile_rows.shape[0]} tau rows, expected {n_tau}')

    idx, taus = outfile_rows[:, 0], outfile_rows[:, 1]
    spec_files = [f"{os.path.join(outdir, basename)}.s{int(i):03}" for i in idx]

    missing = [fl_ for fl_ in spec_files if not os.path.exists(fl_)]
    if missing:
        raise FileNotFoundError(f'{outdir}: {len(missing)} spectra missing, e.g. {missing[0]}')

    dusty_tb_file = os.path.join(outdir, basename) + '.hdf5'
    if dry_run:
        return None, dusty_tb_file, len(spec_files)

    # create output hdf5 table for spectrum at each tau
    dusty_tb = QTable()

    for i, fl_ in enumerate(spec_files):
        t_ = ascii.read(fl_, names=['lambda', 'fTot', 'xAtt', 'xDs', 'xDe', 'fInp', 'tauT', 'albedo'],
                        format='basic', data_start=0)
        if i == 0:
            dusty_tb['lambda'] = t_['lambda'] * u.um
            dusty_tb[f'flam_{taus[i]}'] = t_['fTot'] / t_['lambda']
        else:
            dusty_tb[f'flam_{taus[i]}'] = t_['fTot'] / t_['lambda']

    # scale spectra from dusty output to the correct luminosity
    scale = 10**lum * u.Lsun
    wv = dusty_tb['lambda']
    for col_ in dusty_tb.colnames[1:]:
        dusty_tb[col_] = scale_flux(wv, dusty_tb[col_], scale, flux_scale)

    dusty_tb.write(dusty_tb_file, path='data', serialize_meta=True, overwrite=True)

    return dusty_tb, dusty_tb_file, len(spec_files)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    dist = args.dist * u.Mpc
    flux_scale = ((1 * u.L_sun) / (4 * np.pi * dist**2)).to(u.erg / u.s / u.cm**2)

    outroot = os.path.join(args.griddir, f'dusty_{args.comp}_out')
    if not os.path.isdir(outroot):
        raise ValueError(f'{outroot} not found')

    model_dirs = sorted(d for d in glob.glob(os.path.join(outroot, 'rsg_*')) if os.path.isdir(d))
    print(f'Found {len(model_dirs)} model directories in {outroot}')

    n_fixed, failures = 0, []
    for outdir in model_dirs:
        basename = os.path.basename(outdir)
        try:
            _, tb_file, n_spec = rebuild_table(outdir, basename, args.lum, flux_scale,
                                               n_tau=args.n_tau, skiprows=args.skiprows,
                                               dry_run=args.dry_run)
        except Exception as e:
            print(f'  SKIP {basename}: {e}')
            failures.append(basename)
            continue
        action = 'would write' if args.dry_run else 'wrote'
        print(f'  {action} {os.path.basename(tb_file)} with {n_spec} tau columns')
        n_fixed += 1

    print(f'\n{n_fixed}/{len(model_dirs)} tables {"checked" if args.dry_run else "rewritten"}'
          f' with {args.n_tau} taus')
    if failures:
        print(f'{len(failures)} failed: {", ".join(failures)}')
