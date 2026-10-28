import pandeia.engine
import stsynphot
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob, sys, re, json, time
import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.table import QTable
from astropy.io import ascii
from scipy import interpolate
from scipy.integrate import simpson
import subprocess
import tempfile
import argparse
from pathlib import Path

def create_parser():
    '''
    Create an argument parser

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser
    '''
    parser = argparse.ArgumentParser(description='Create DUSTY grids. By default, build the single-shell power-law grid '
                                     '(dusty_likelihood.build_grid) for SN 2026sqf; --gen runs dusty_gen.gen_grid instead. '
                                     'Paths are relative to the repo root.')
    parser.add_argument('--gen', default=False, action='store_true', help='Run dusty_gen.gen_grid instead of the shell grid')
    parser.add_argument('--logz', type=float, default=None, help='log metallicity (0.0, -0.25 and -0.5 for MARCS; 0.0 and -0.5 for NewEra; '
                        'COMARCS -0.5). Default -0.5 for the shell grid, 0.0 for --gen')
    parser.add_argument('--comp', type=str, default='sil', help='Dust composition for --gen (sil or grf)')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for spectra (default data/dusty_grid for the '
                        'shell grid, data/dusty_sil_grid for --gen)')
    parser.add_argument('--modeldir', type=str, default=None, help='Model directory for DUSTY input spectra (default '
                        'data/marcs/comarcs for the shell grid, data/marcs/m1.00_g0.00 for --gen)')
    parser.add_argument('--rewrite_lambda_grid', default=False, action=argparse.BooleanOptionalAction, help='Rewrite the wavelength grid in DUSTY and recompile')

    g = parser.add_argument_group('shell grid', 'rho ~ r^-s from R_in to Y R_in; defaults are the coarse axes in SHELL_GRID_AXES')
    g.add_argument('--out', type=str, default='data/dusty_grid/shell_grid_amC_SiC20_ism.npz',
                   help='Grid file; cells are sharded in <out>.shards/, so rerunning resumes an interrupted build')
    g.add_argument('--phot', type=str, default='data/dolphot/sn2026sqf/sn2026sqf_photometry_new_ujy.txt',
                   help='Photometry table whose filter and instrument columns set the grid bands')
    g.add_argument('--dist', type=float, default=19.0, help='Distance (Mpc)')
    g.add_argument('--c_o', type=float, default=1.4, help='COMARCS C/O')
    g.add_argument('--mass', type=float, default=2.0, help='COMARCS mass (Msun)')
    g.add_argument('--sic', type=float, default=0.2, help='SiC-Pg abundance; amC (--amc_nk) takes the rest')
    g.add_argument('--amc_nk', type=str, default='amC-zb2.nk', help='amC optical constants (.nk in $DUSTY_PATH/Lib_nk)')
    g.add_argument('--size', type=float, nargs=3, default=[3.5, 0.005, 0.25], metavar=('Q', 'AMIN', 'AMAX'),
                   help='MRN size distribution: power q and amin, amax (um)')
    for k, name in [('temp', 'teff'), ('dust_temp', 'tin'), ('tau', 'tau'), ('s', 's'), ('Y', 'Y')]:
        g.add_argument(f'--{name}', type=float, nargs='+', default=None, help=f'{k} axis (default {SHELL_GRID_AXES[k].tolist()})')
    g.add_argument('--nproc', type=int, default=10, help='Parallel DUSTY processes')
    g.add_argument('--retry_failed', default=False, action='store_true', help='Rerun cells whose DUSTY run failed as a whole')
    g.add_argument('--save_sed', default=False, action='store_true',
                   help='Also store the full DUSTY spectrum of every model (fnu in uJy at --dist for 1e4 Lsun, as sed and '
                        'sed_lambda), e.g. as input spectra for an outer shell; ~120 MB for the default grid')
    g.add_argument('--workdir', type=str, default='data/dusty_grid/work_shell',
                   help='DUSTY working directories (keep the path short: DUSTY has fixed-length file names)')
    return parser

# Coarse single-shell grid (amC + SiC, MRN q 3.5, 0.005-0.25 um, COMARCS) for showing that one spherical shell cannot fit
# SN 2026sqf. PCHIP accuracy from hold-out cuts at Teff 4000 (2026sqf_tmp/shell_grid_holdout.py): temp 200 K steps <= 0.04
# mag (default_grid_axes); dust_temp x1.19 steps <= 0.017 mag, so x1.29 here is expected at ~0.05 mag (<= 0.3 sigma);
# tau 0.15 dex <= 0.011 mag; s at 1.0 steps <= 0.07 mag NIRCam, 0.15 mag mid-IR, so ~4x less at 0.5; Y with 1.5, 3, 10,
# 100, 1e4 reached 0.2 mag mid-IR (0.32 at F2550W) in the 10-100 and 100-1e4 gaps, filled here by 30 and 1000. Check the
# grid with validate_grid on posterior draws
SHELL_GRID_AXES = {'temp': np.arange(2600., 4400. + 1e-6, 200.),
                   'dust_temp': np.round(np.geomspace(150., 1500., 10)),
                   'tau': np.round(np.logspace(-1, 2, 21), 5),
                   's': np.array([1., 1.5, 2., 2.5]),
                   'Y': np.array([1.5, 3., 10., 30., 100., 1000., 1e4])}
class dusty_gen(object):
    def __init__(self, 
                 dist=10*u.Mpc, 
                 logz=0.00,
                 logg=0.00,
                 interp_method='pchip',
                 rewrite_lambda_grid=False,
                 modeldir='/Users/aswin/rsg_phot/data/marcs/m1.00_g0.00',
                 outdir='/Users/aswin/rsg_phot/data/dusty_grid',
                 mass=1.0,
                 c_o=None,
                 xi=2.5,
                 comarcs_version=None):

        self.dist = dist

        # This is the total integrated flux of a source with Lum = 1 Lsun in units of
        # erg/s/cm2.  Used to renormalize the RSG spectrum.  This is equivalent to
        # 3.839e33/(4 * pi * (10 * 3.08568025e18)^2) for 1 Lsol at 10 pc.
        self.FLUX_SCALE = ((1 * u.L_sun) / (4 * np.pi * self.dist**2)).to(u.erg/u.s/u.cm**2)

        # Dust-to-gas mass ratio assumption
        self.DUST_TO_GAS = 0.01

        # Read model atmospheres (COMARCS is checked first: its path contains 'marcs')
        self.rsg_modeldir = modeldir
        self.rsg_logz = logz
        sgn = '+' if self.rsg_logz > -1e-5 else '-'
        if 'comarcs' in os.path.basename(os.path.normpath(self.rsg_modeldir)).lower():
            self.modeltype='COMARCS'
        elif 'marcs' in self.rsg_modeldir:
            self.modeltype='MARCS'
        elif 'newera' in self.rsg_modeldir:
            self.modeltype='NewEra'
        else:
            raise ValueError('Model directory must have COMARCS, MARCS or NewEra models')

        if self.modeltype == 'COMARCS':
            self.load_comarcs(logg=logg, logz=logz, mass=mass, c_o=c_o, xi=xi, version=comarcs_version)
        else:
            model_spec = np.loadtxt(os.path.join(self.rsg_modeldir,
                                                 '{modeltype}_models_g{logg:.2f}_Z{sgn}{met:.2f}.dat'.format(modeltype=self.modeltype,
                                                                                                             sgn=sgn,
                                                                                                             logg=logg,
                                                                                                             met=np.abs(self.rsg_logz))),
                                                 dtype=float)
            self.rsg_temp = model_spec[0] * u.K
            self.rsg_data = model_spec[1:]
            self.rsg_wavelength = np.logspace(np.log10(1300.), np.log10(200000.), num=5036) * u.Angstrom
            # blackbody tail beyond the model coverage, scaled over rsg_match (micron)
            self.rsg_join, self.rsg_match = 20.0, (10.0, 20.0)
            self.rsg_tag = self.modeltype.lower()
        self.rsg_minflux = np.min(self.rsg_data)
        self.rsg_interp = self.setup_temp_interp(interp_method)

        # DUSTY setup 
        self.dusty_basedir = os.environ['DUSTY_PATH'] #full path
        self.dusty_datadir = outdir # full path
        self.dusty_lambda_grid = self.make_lambda_grid(npts=500)
        self.dusty_n_taugrid = 27
        self.validate_dusty_dir(rewrite_lambda_grid)
        self.dust_comp = 'sil'
        self.shell_thickness = 2

    def rebin(self, a, newshape):
        newarray = np.zeros(newshape)
        curr = 0
        for i in np.arange(newshape):
            s = slice(curr, curr+int(a.shape[0]/newshape), 1)
            newarray[i] = np.mean(a[s])
            curr += int(a.shape[0]/newshape)
        return newarray
    
    COMARCS_PATTERN = re.compile(r'^(?P<pre>[a-z]+(?P<code>\d+)v\d+)_t(?P<t>\d+)_g(?P<g>[+-]\d+)_m(?P<m>\d+)'
                                 r'_o(?P<o>\d+)_c(?P<c>\d+)_fe(?P<fe>\d+)(?P<extra>(_[a-z]+[+-]\d+)*)'
                                 r'_xi(?P<xi>\d+)_(?P<v>[a-z0-9]+?)(?P<suf>_.*)?\.isp$')
    # log eps(Fe) of the solar-scaled COMARCS sets, so [Fe/H] = log eps(Fe) - 7.563
    COMARCS_FE_SUN = 7.563

    def parse_comarcs_name(self, fname):
        # COMARCS file names encode Teff, log g, mass, log eps of O, C and Fe,
        # any abundance offsets, the microturbulence and the code version, 
        # e.g. mxcom10v06_t2500_g+000_m0100_o88029_c89491_fe75630_xi25_c10b.isp
        m = self.COMARCS_PATTERN.match(fname)
        if m is None:
            return None
        val = lambda s: int(s) / 10**(len(s) - 1)
        return {'file': fname, 'code': int(m['code']), 'temp': int(m['t']), 'logg': int(m['g'])/100,
                'mass': int(m['m'])/100, 'c_o': 10**(val(m['c']) - val(m['o'])),
                'feh': val(m['fe']) - self.COMARCS_FE_SUN, 'xi': int(m['xi'])/10, 'version': m['v'],
                'extra': m['extra'] or '', 'suffix': m['suf'] or ''}

    def load_comarcs(self, logg, logz, mass, c_o, xi=2.5, version=None):
        # Load one 1D Teff sequence of COMARCS R~200 spectra  at fixed
        # log g, mass, [Fe/H], C/O and microturbulence. 
        models = [p for p in map(self.parse_comarcs_name, os.listdir(self.rsg_modeldir))
                  if p is not None and not p['extra'] and not p['suffix']]
        match = [p for p in models if abs(p['logg'] - logg) < 0.005 and abs(p['mass'] - mass) < 0.005
                 and abs(p['feh'] - logz) < 0.01 and abs(p['xi'] - xi) < 0.01
                 and (c_o is not None and abs(p['c_o']/c_o - 1) < 0.01)]
        if version is not None:
            match = [p for p in match if p['version'] == version]
        if not match:
            avail = sorted({round(p['c_o'], 2) for p in models if abs(p['logg'] - logg) < 0.005 and
                            abs(p['mass'] - mass) < 0.005 and abs(p['feh'] - logz) < 0.01})
            raise ValueError(f'No COMARCS models for logg={logg}, M={mass}, [Fe/H]={logz}, C/O={c_o}, xi={xi}'
                             + (f', version={version}' if version else '') + f'; C/O available there: {avail}')

        versions = sorted({p['version'] for p in match})
        if len(versions) > 1:
            temps = {v: sorted({p['temp'] for p in match if p['version'] == v}) for v in versions}
            raise ValueError(f'COMARCS models at these parameters come from several code versions {temps}; '
                             'pass comarcs_version to pick one rather than mixing them')
        # the same model computed with several code releases: keep the newest
        by_temp = {}
        for p in match:
            if p['temp'] not in by_temp or p['code'] > by_temp[p['temp']]['code']:
                by_temp[p['temp']] = p
        if len(by_temp) < 2:
            raise ValueError(f'Need at least two COMARCS temperatures to interpolate, found {sorted(by_temp)}')

        temps = sorted(by_temp)
        wave, flux = None, []
        for t in temps:
            d = np.loadtxt(os.path.join(self.rsg_modeldir, by_temp[t]['file']))
            if wave is None:
                wave = d[:, 0]
            elif len(d) != len(wave) or not np.allclose(d[:, 0], wave, rtol=1e-6):
                raise ValueError(f'{by_temp[t]["file"]} is on a different wavelength grid from the other models')
            flux.append(d[:, 2] / d[:, 0])                     # lambda*L_lambda -> L_lambda shape

        self.rsg_temp = np.array(temps, dtype=float) * u.K
        self.rsg_data = np.array(flux).T                         # (n_wavelength, n_temp), as for MARCS
        self.rsg_wavelength = wave * u.Angstrom
        self.rsg_files = [by_temp[t]['file'] for t in temps]
        self.comarcs_version = versions[0]

        self.rsg_join = wave[-1] / 1e4
        self.rsg_match = (0.96 * self.rsg_join, self.rsg_join)
        self.rsg_tag = (f'comarcs_g{logg:+.2f}_m{mass:.2f}_feh{logz:+.2f}_co{c_o:.2f}_xi{xi:.1f}_'
                        f'{self.comarcs_version}')

    def setup_temp_interp(self, interp_method='pchip'):
        # All model spectra share one wavelength grid, so interpolate in temperature
        # only, in log flux. This is exact at the grid temperatures and, with PCHIP,
        # monotone between them.
        logflux = np.log10(np.clip(self.rsg_data, self.rsg_data[self.rsg_data > 0].min(), None))
        if interp_method in ('pchip', 'cubic'):
            return interpolate.PchipInterpolator(self.rsg_temp.value, logflux, axis=1, extrapolate=False)
        elif interp_method == 'linear':
            return interpolate.interp1d(self.rsg_temp.value, logflux, axis=1, bounds_error=False)
        else:
            raise ValueError('interp_method must be "pchip" (or "cubic") or "linear"')

    def get_rsg(self, temp, model='g00'):

        if model == 'g00': 
            flux = 10**self.rsg_interp(np.atleast_1d(temp.to(u.K).value)[0])
            flux = flux.flatten()
            flux[flux<0] = self.rsg_minflux
        else: 
            raise NotImplementedError('Only model "g00" is currently available')

        return flux
    
    def make_lambda_grid(self, npts=500):
        # DUSTY wavelength grid in three log-spaced segments
        n_out = max(22, int(round(0.175*npts)))
        n_mid = npts - 2*n_out
        uv = np.logspace(np.log10(0.01), np.log10(0.6), n_out, endpoint=False)
        mid = np.logspace(np.log10(0.6), np.log10(30), n_mid, endpoint=False)
        fir = np.logspace(np.log10(30), np.log10(3.6e4), n_out)
        # pin the end points exactly; logspace rounding otherwise fails DUSTY's checks
        uv[0], fir[-1] = 0.01, 3.6e4
        grid = np.concatenate([uv, mid, fir])
        if (grid[1:]/grid[:-1]).max() >= 1.5:
            raise ValueError(f'{npts} points is too few for DUSTY (step ratio >= 1.5)')

        return grid

    def read_lambda_grid(self):
        # the wavelength grid the compiled DUSTY binary actually uses
        grid = np.loadtxt(os.path.join(self.dusty_basedir, 'lambda_grid.dat'), skiprows=1)
        return np.unique(grid)

    def validate_dusty_dir(self, rewrite_lambda=False):
        if rewrite_lambda:
            gridfile = os.path.join(self.dusty_basedir, 'lambda_grid.dat')
            parfile = os.path.join(self.dusty_basedir, 'userpar.inc')
            nline = f'npL = {len(self.dusty_lambda_grid)}\n'

            # edit wavelength grid file
            with open(gridfile, 'w') as f:
                f.write(nline)
            with open(gridfile, 'ab') as f:
                np.savetxt(f, self.dusty_lambda_grid)

            # update user parameter file
            with open(parfile, 'r') as f:
                par_lines = f.readlines()
            par_lines[15] = f'      PARAMETER (npL={len(self.dusty_lambda_grid)})\n'
            with open(parfile, 'w') as f:
                f.writelines(par_lines)

            # recompile dusty
            curdir = os.getcwd()
            os.chdir(self.dusty_basedir)
            subprocess.run(['gfortran', 'dustyV2.f', '-std=legacy', '-o', 'dusty'], check=True)
            os.chdir(curdir)

        default_tau = np.array(list(np.arange(0.0, 1.1, 0.1)) + list(np.arange(1.5, 6.0, 0.5)) + list(np.arange(6.0, 13.0, 1.0)) +\
                        list(np.arange(15.0, 33.0, 2.0)) + list(np.arange(35.0, 61.0, 5.0))) 
        # If the first entry is 0, dusty will throw a non-invertible matrix error. Minimum value is 1e-4
        default_tau[0] = 1e-4
        taufile = os.path.join(self.dusty_basedir, 'taugrid.dat')
        exist_tau = np.loadtxt(taufile, skiprows=1, dtype=float)
        if len(exist_tau) != len(default_tau):
            replace_tau = True
        elif (np.abs(exist_tau - default_tau) > 1e-5).any():
            replace_tau = True
        else:
            replace_tau = False

        if replace_tau:
            with open(taufile, 'w') as f:
                f.write('lambda0 = 0.55 micron\n')
                for t in default_tau:
                    f.write(f'   {t}\n')
            self.dusty_n_taugrid = len(default_tau)

        # make sure all required files are present in the dusty directory
        req_files = ['dusty', 'dustyV2.f', 'userpar.inc', 'lambda_grid.dat', 'dusty.inp']
        for fl_ in req_files:
            if not os.path.exists(os.path.join(self.dusty_basedir, fl_)):
                raise ValueError(f"{fl_} not found in {self.dusty_basedir}")

    def extend_bb_tail(self, wv, flux, temp, lam_join=20.0, lam_match=(10.0, 20.0), lam_max=3.6e4, npts=300):
        # Extend the model atmosphere beyond lam_join (micron) with a blackbody at
        # temp, scaled to the model's continuum over lam_match so the join does not
        # hinge on a single flux point. Otherwise DUSTY extrapolates a 
        # lambda*F_lambda ~ lambda^-3 tail from the last input point.
        bb = lambda w: 1.0 / (w**5 * np.expm1(14387.77 / (w * temp)))   # B_lambda, w in micron
        m = (wv >= lam_match[0]) & (wv <= lam_match[1])
        scale = np.median(flux[m] / bb(wv[m]))

        keep = wv <= lam_join
        tail_wv = np.logspace(np.log10(lam_join), np.log10(lam_max), npts)[1:]
        return np.concatenate([wv[keep], tail_wv]), np.concatenate([flux[keep], scale * bb(tail_wv)])

    def cell_average(self, wv, flux, grid):
        # Average F_lambda over each DUSTY grid cell (edges at the geometric midpoints
        # between grid points). DUSTY linearly interpolates the input spectrum at its
        # grid points, so passing cell averages makes it flux-conserving instead of
        # point-sampling the molecular bands. Cells only partly covered by the model
        # are averaged over the covered part; cells outside it are dropped (DUSTY sets
        # the stellar flux to zero below the shortest input wavelength).
        edges = np.concatenate([[grid[0]], np.sqrt(grid[1:]*grid[:-1]), [grid[-1]]])
        cum = np.concatenate([[0.0], np.cumsum(0.5*(flux[1:] + flux[:-1])*np.diff(wv))])
        lo, hi = np.clip(edges[:-1], wv[0], wv[-1]), np.clip(edges[1:], wv[0], wv[-1])
        covered = hi > lo
        integral = np.interp(hi, wv, cum) - np.interp(lo, wv, cum)

        return grid[covered], integral[covered] / (hi - lo)[covered]

    def setup_input_spec(self, temp, outdir, redo=False):
        grid = self.read_lambda_grid()
        # the file depends on the DUSTY grid it was averaged onto
        specfilename = os.path.join(outdir, f'{self.rsg_tag}_{temp.value}_npL{len(grid)}.dat')
        if redo or not os.path.exists(specfilename):
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # get marcs model flux, extend with a blackbody beyond 20 micron and
            # average onto the DUSTY wavelength grid
            rsg_flux = self.get_rsg(temp, model='g00')
            rsg_wv = self.rsg_wavelength.to(u.um).value
            rsg_wv, rsg_flux = self.extend_bb_tail(rsg_wv, rsg_flux, temp.to(u.K).value,
                                                   lam_join=self.rsg_join, lam_match=self.rsg_match)
            spec_input = np.array(self.cell_average(rsg_wv, rsg_flux, grid)).T

            # write marcs model as input spectrum for dusty
            with open(specfilename, 'w') as f:
                f.write(f'{self.modeltype} model atmosphere for T={temp.value} K\n')
                f.write(f'  lambda    L_lambda\n')
                f.write(f' (micron)  (arbitrary)\n')
            with open(specfilename, 'ab') as f:
                np.savetxt(f, spec_input)

        return specfilename

    def setup_dusty(self, outdir, p):
        #p - specfilename (path), temp (in K), dust_temp (in K), dust_comp ('sil' or 'grf'), shell_thickness (float), tau (at 0.55 micron)
        #setup input file for dusty
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # input file - outdir/rsg_temp_dusttemp.inp
        inp_file = os.path.join(outdir, f'rsg_{p['temp'].value}_{p['dust_temp'].value}.inp')

        #write input file
        with open(inp_file, 'w') as f:
            f.write('  I PHYSICAL PARAMETERS\n')
            f.write('     1) External radiation:\n')
            f.write('                Spectrum = 5\n')
            f.write(f'                {p['input_spec']}\n')
            f.write('     2) Dust Properties\n\n')
            f.write('       2.1 Chemical composition\n')
            f.write('           Optical properties index = 1\n')
            f.write('           Abundances for supported grain types:\n')
            f.write('               Sil-Ow  Sil-Oc  Sil-DL  grf-DL  amC-Hn  SiC-Pg\n')
            if p['dust_comp'].lower() == 'sil' or p['dust_comp'].lower() == 'silicate': # use Draine and Lee silicate dust
                f.write('           x =  0.00    0.00    1.00    0.00    0.00    0.00\n\n')
            elif p['dust_comp'].lower() == 'grf' or p['dust_comp'].lower() == 'graphite': # use Draine and Lee graphite dust
                f.write('           x =  0.00    0.00    0.00    1.00    0.00    0.00\n\n')
            f.write('       2.2 Grain size distribution\n\n')
            f.write('          Size distribution = 2 % arbitrary MRN\n')
            f.write('          q = 3.0, a(min) = 0.05 micron, a(max) = 1.00 micron\n\n')
            f.write('       2.3 Dust temperature on inner boundary:\n\n')
            f.write(f'        - temperature = {p['dust_temp'].value} K\n\n')
            f.write('     3) Density Distribution\n')
            f.write('        - density type = 1\n')
            f.write('        - number of powers = 1\n')
            f.write(f'        - shell\'s relative thickness = {p['shell_thickness']}\n')
            f.write('        - power = 2\n\n')
            f.write('     4) Optical Depth\n')
            f.write('        - grid type = 3\n')
            f.write('            taugrid.dat\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  II NUMERICS\n\n')
            f.write('     - accuracy for flux conservation = 0.05\n\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  III OUTPUT PARAMETERS\n\n')
            f.write('        FILE DESCRIPTION                               FLAG\n')
            f.write('       ------------------------------------------------------------\n')
            f.write('       - verbosity flag;                               verbose = 1\n')
            f.write('       - properties of emerging spectra;             fname.spp = 0\n')
            f.write('       - detailed spectra for each model;           fname.s### = 2\n')
            f.write('       - images at specified wavelengths;           fname.i### = 0\n')
            f.write('       - visibility function at spec. wavelengths;  fname.v### = 0\n')
            f.write('       - radial profiles for each model;            fname.r### = 0\n')
            f.write('       - detailed run-time messages;                fname.m### = 0\n')
            f.write('       -------------------------------------------------------------\n\n')
            f.write('  The end of the input parameters listing.\n')


    def scale_flux(self, wv, flux, scale):
        # Renormalize the RSG spectrum so it's in units of erg/s/cm2/angstrom for
        # synphot to interpret
        normalize = simpson(flux, x=wv.to(u.um).value)
        flux = scale.value * self.FLUX_SCALE * flux/normalize / u.um
        flux = flux.to(u.erg/u.s/u.cm**2/u.Angstrom)

        return flux

    def run_dusty(self, temp, dust_temp,
                  dust_comp='sil', 
                  shell_thickness=2, 
                  filedir=None, 
                  redo_input=False,
                  redo_dusty=False,
                  tb_overwrite=False):
        curdir = os.getcwd()

        self.dust_comp, self.shell_thickness = dust_comp, shell_thickness
        if filedir is None:
            filedir = self.dusty_datadir

        # create input spectrum
        input_spec = self.setup_input_spec(temp=temp, outdir=os.path.join(filedir, 'model_spec'), redo=redo_input)
        # parameter list
        p = {'input_spec': input_spec, 
             'temp' : temp, 
             'dust_temp': dust_temp, 
             'dust_comp': dust_comp, 
             'shell_thickness' : shell_thickness}
        
        # directory structure
        # - root (filedir)
        #   - model_spec
        #       - spectra
        #   - dusty_out
        #       - basename: rsg_temp_dusttemp (outdir)
        #           - basename.inp (inpfile)
        #           - bsaename.out (outfile)
        #           - basename.s### (spec_files)
        #           - basename.hdf5 (dusty_tb_file)

        basename = f'rsg_{p['temp'].value}_{p['dust_temp'].value}'
        outdir = os.path.join(filedir, f'dusty_{dust_comp}_out', basename)
        outfile = os.path.join(outdir, basename + '.out')

        # run dusty
        if redo_dusty or not os.path.exists(outfile):
            self.setup_dusty(outdir=outdir, p=p)

            # write input file in dusty directory
            os.chdir(self.dusty_basedir)
            with open('dusty.inp', 'w') as f:
                f.write(os.path.join(outdir, basename))

            try:
                _ = subprocess.run(['./dusty'], check=True, capture_output=True).stdout.decode()
            except subprocess.CalledProcessError as e:
                raise e
            os.chdir(curdir)

        # read outfile and spectra files
        # BUG: self.dusty_n_taugrid is not updated if taugrid.dat is changed, so we read the number of rows in the outfile instead
        outfile_rows = np.loadtxt(outfile, skiprows=42, max_rows=self.dusty_n_taugrid)
        idx, taus = outfile_rows[:, 0], outfile_rows[:, 1]
        spec_files = [f"{os.path.join(outdir, basename)}.s{int(i):03}" for i in idx]
        taus = outfile_rows[:, 1]

        # create output hdf5 table for spectrum at each tau
        dusty_tb = QTable()
        dusty_tb_file = os.path.join(outdir, basename) + '.hdf5'

        for i, fl_ in enumerate(spec_files):
            t_ = ascii.read(fl_,  names=['lambda', 'fTot', 'xAtt', 'xDs', 'xDe', 'fInp', 'tauT', 'albedo'], 
                            format='basic', data_start=0)
            if i == 0:
                dusty_tb['lambda'] = t_['lambda'] * u.um
                dusty_tb[f'flam_{taus[i]}'] = t_['fTot']/t_['lambda']
            else:
                dusty_tb[f'flam_{taus[i]}'] = t_['fTot']/t_['lambda']

        dusty_tb.write(dusty_tb_file, path='data', serialize_meta=True, overwrite=tb_overwrite)

        return dusty_tb, dusty_tb_file
    
    def dust_spec(self, lum, temp, dust_temp, **kwargs):
        dusty_tb, dusty_tb_file = self.run_dusty(temp, dust_temp, **kwargs)
        scale = 10**lum * u.Lsun
        wv = dusty_tb['lambda']

        # scale spectra from dusty output to the correct luminosity
        for col_ in dusty_tb.colnames[1:]:
            dusty_tb[col_] = self.scale_flux(wv, dusty_tb[col_], scale)

        dusty_tb.write(dusty_tb_file, path='data', serialize_meta=True, overwrite=True)

        return dusty_tb
    
    def gen_grid(self, grid_temps=None, grid_dust_temps=None, grid_taus='default', **kwargs):
        print(f'Creating a {self.dust_comp} DUSTY grid at log(Z)={self.rsg_logz} using {self.modeltype} input spectra')
        if grid_temps is None:
            if self.modeltype=='MARCS':
                grid_temps = np.array(list(np.arange(2500., 4100., 100.)) + list(np.arange(4250., 5250., 250.)))
            if self.modeltype=='MARCS15':
                grid_temps = np.array(list(np.arange(3300., 4100., 100.)) + list(np.array([4250, 4500])))
            if self.modeltype=='NewEra':
                grid_temps = np.arange(2500, 5100, 100)
            if self.modeltype=='COMARCS':
                grid_temps = self.rsg_temp.value
                
        if grid_dust_temps is None:
            grid_dust_temps = np.array([200.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 1800.0])
        
        if grid_taus != 'default':
            if not (isinstance(grid_taus, list) or isinstance(grid_taus, np.ndarray)):
                raise ValueError('grid_taus has to be a list or numpy array, if not "default"')
            taufile = os.path.join(self.dusty_basedir, 'taugrid.dat')
            with open(taufile, 'w') as f:
                f.write('lambda0 = 0.55 micron\n')
                for t in grid_taus:
                    f.write(f'   {t}\n')
            self.dusty_n_taugrid = len(grid_taus)

        for rsg_temp in grid_temps:
            for dust_temp in grid_dust_temps:
                print(f'Generating grid for {rsg_temp}, {dust_temp}')
                tb = self.dust_spec(4.0, rsg_temp*u.K, dust_temp*u.K, dust_comp=self.dust_comp, 
                                    shell_thickness=self.shell_thickness, **kwargs)

class DustyError(RuntimeError):
    '''
    Raised when a DUSTY model fails or does not reach the requested flux
    conservation accuracy, so that a sampler can catch it and reject the point.
    '''
    pass

# ----------------------------------------------------------------------------
# Exponential mass-loss density family (see DUSTY_Grid_Design.md)
#
# A constant-velocity wind with Mdot(t) ~ exp(t/t_e) has, in y = r/R_in,
#     rho(y) ~ y^-2 exp[-u (y - 1)],    u = R_in / (v t_e)
# u > 0 is rising mass loss (steeper than r^-2), u = 0 constant (pure r^-2) and
# u < 0 falling mass loss (shallower). Single models only (dusty_likelihood
# density='exp'); the precomputed grids use power laws (see build_grid).

# DUSTY aborts when eta_min/eta_max < 1e-12 anywhere in the shell; keep a margin
ETA_FLOOR = 1e-11
def exp_density_ymax(u_, Y, floor=ETA_FLOOR):
    '''
    Outer radius DUSTY can handle for the exponential profile. For rising mass loss
    (u > 0) eta falls below floor * eta(1) before Y once 2 ln y + u (y - 1) > -ln(floor),
    and DUSTY then aborts. The dust beyond that radius holds a fraction ~ floor of tau,
    so the profile is truncated there (Y_eff); for u <= 0 the full Y is kept.
    '''
    from scipy.optimize import brentq
    f = lambda y: -2*np.log(y) - u_*(y - 1) - np.log(floor)
    if f(Y) >= 0:
        return float(Y)
    return brentq(f, 1.0, Y, xtol=1e-6, rtol=1e-10)

def exp_density_profile(u_, Y, npts=300, floor=ETA_FLOOR):
    '''
    Tabulated (y, eta) for DUSTY's user-supplied density (input density type 5), and
    the effective outer radius Y_eff (see exp_density_ymax). The table merges a
    log-spaced and a linear-spaced grid, since DUSTY interpolates it linearly: the
    log spacing resolves the y^-2 fall near R_in and the linear spacing the
    exponential rise near R_out for u < 0. DUSTY allows at most 1000 rows.
    '''
    if Y <= 1:
        raise ValueError('Y = R_out/R_in must be > 1')
    if 2*npts > 1000:
        raise ValueError('DUSTY reads at most 1000 density points (npts <= 500)')
    Y_eff = exp_density_ymax(u_, Y, floor)
    y = np.unique(np.concatenate([np.geomspace(1.0, Y_eff, npts), np.linspace(1.0, Y_eff, npts)]))
    # eta relative to its maximum, so the table never underflows for u < 0
    log_eta = -2*np.log(y) - u_*(y - 1)
    log_eta -= log_eta.max()
    # truncated profiles end at the floor itself
    if log_eta.min() < np.log(floor) - 1e-6:
        raise ValueError(f'Density contrast of u={u_}, Y={Y} exceeds DUSTY\'s dynamic range '
                         f'(eta_min/eta_max = {np.exp(log_eta.min()):.1e})')
    return y, np.exp(log_eta), Y_eff

class dusty_likelihood(dusty_gen):
    '''
    Run single DUSTY models on the fly, so that the output SED can be used directly
    as the model in an MCMC / nested sampling fit.

    Unlike dusty_gen, which writes a full tau grid per (temp, dust_temp) pair, every
    call to model_sed runs exactly one DUSTY model at the requested tau and shell
    thickness. Each process gets its own DUSTY working directory (dusty.inp and
    lambda_grid.dat are read from the cwd), so the model can be evaluated from a
    multiprocessing pool.

    Parameters (theta), in order of PARAM_NAMES, minus any that are fixed:
        logl            : log10(L/Lsun)
        temp            : photospheric Teff (K)
        dust_temp       : dust temperature at the inner boundary T_in (K)
        tau             : optical depth at 0.55 micron
        shell_thickness : relative shell thickness Y = R_out/R_in (fixed at 1e4 by default;
                          pass shell_thickness=None to sample it)
        u               : only for density='exp', the exponential mass-loss shape
                          rho ~ y^-2 exp[-u (y-1)], u = R_in/(v t_e) (see exp_density_profile)

    For precomputed grids over (temp, dust_temp, tau, s, Y), with rho ~ r^-s, see
    build_grid and DustyGrid.
    '''
    PARAM_NAMES = ['logl', 'temp', 'dust_temp', 'tau', 'shell_thickness']
    DENSITIES = ('rdw', 'powerlaw', 'exp')

    # Built-in DUSTY grain types, in the column order of the input file
    DUSTY_GRAINS = ['Sil-Ow', 'Sil-Oc', 'Sil-DL', 'grf-DL', 'amC-Hn', 'SiC-Pg']
    GRAIN_ALIASES = {'sil': 'Sil-DL', 'silicate': 'Sil-DL',
                     'grf': 'grf-DL', 'graphite': 'grf-DL',
                     'amc': 'amC-Hn', 'carbon': 'amC-Hn',
                     'sic': 'SiC-Pg'}

    # DUSTY reports r1 for a source with L = 1e4 Lsun
    DUSTY_LUM = 1e4 * u.Lsun

    def __init__(self,
                 dist=10*u.Mpc,
                 logz=0.00,
                 logg=0.00,
                 comp=None,
                 nk_comp=None,
                 size_dist='mrn',
                 size_par=None,
                 density='rdw',
                 density_power=2,
                 shell_thickness=1e4,
                 fixed=None,
                 interp_method='pchip',
                 mass=1.0,
                 c_o=None,
                 xi=2.5,
                 comarcs_version=None,
                 workdir=None,
                 keep_files=False,
                 accuracy=0.05,
                 outdir='../data/dusty_sil_grid',
                 modeldir='../data/marcs/m1.00_g0.00'):

        super().__init__(dist=dist, logz=logz, logg=logg, interp_method=interp_method,
                         rewrite_lambda_grid=False, modeldir=modeldir, outdir=outdir,
                         mass=mass, c_o=c_o, xi=xi, comarcs_version=comarcs_version)
        self.logz = logz
        self.outdir = Path(outdir)
        self.modeldir = Path(modeldir)

        # Dust properties are fixed per fit 
        # comp sets the built-in grains; nk_comp optionally adds grains from
        # optical-constant (.nk) files, e.g. comp={'sic': 0.2}, nk_comp={'amC-zb2.nk': 0.8}.
        # With no composition given at all, fall back to pure silicate.
        if comp is None and not nk_comp:
            comp = 'sil'
        self.comp = comp
        self.abundances, self.nk_abundances = self.parse_comp(comp, nk_comp)
        self.size_dist = size_dist.lower()
        if self.size_dist == 'mrn':
            # modified MRN; amin = amax gives a single grain size
            self.size_par = {'q': 3.0, 'amin': 0.05, 'amax': 1.00}
        elif self.size_dist == 'kmh':
            self.size_par = {'q': 3.5, 'amin': 0.005, 'a0': 0.2}
        else:
            raise ValueError('size_dist must be "mrn" or "kmh"')
        if size_par is not None:
            self.size_par.update(size_par)
        self.density = density.lower()
        if self.density not in self.DENSITIES:
            raise ValueError('density must be "rdw" (analytic radiatively driven wind), "powerlaw" '
                             'or "exp" (r^-2 x exponential mass-loss history)')
        self.density_power = density_power
        # the exponential family adds the shape parameter u
        self.param_names = self.PARAM_NAMES + (['u'] if self.density == 'exp' else [])

        # required fractional accuracy of bolometric flux conservation (DUSTY q_acc).
        # DUSTY reports errors < 1% as 0, so the post-run check only bites for accuracy >= 0.01
        self.accuracy = accuracy

        # Fixed parameters are removed from theta, e.g. fixed={'temp': 3000*u.K}
        self.fixed = {} if shell_thickness is None else {'shell_thickness': shell_thickness}
        for k, v in (fixed or {}).items():
            if k not in self.param_names:
                raise ValueError(f'Unknown parameter {k}, must be one of {self.param_names}')
            self.fixed[k] = v.value if isinstance(v, u.Quantity) else v
        self.free_params = [p for p in self.param_names if p not in self.fixed]

        # Per-process DUSTY working directories are created lazily (after a pool forks)
        self.workdir_base = workdir
        self.keep_files = keep_files
        self._workdirs = {}
        self._input_spec = {}

    def parse_comp(self, comp, nk_comp=None):
        # comp is either a single built-in grain type ('sil', 'amc', ...) or a dict
        # of abundances, e.g. {'amc': 0.9, 'sic': 0.1}. nk_comp is a dict of
        # {.nk file: abundance}. Abundances are normalized over both together, as
        # DUSTY does.
        if comp is None:
            comp = {}
        elif isinstance(comp, str):
            comp = {comp: 1.0}

        abundances = dict.fromkeys(self.DUSTY_GRAINS, 0.0)
        for grain, frac in comp.items():
            name = self.GRAIN_ALIASES.get(grain.lower(), grain)
            if name not in abundances:
                raise ValueError(f'Unknown grain type {grain}, must be one of '
                                 f'{self.DUSTY_GRAINS} or {list(self.GRAIN_ALIASES)}')
            if frac < 0:
                raise ValueError(f'Abundance of {grain} must be non-negative')
            abundances[name] += frac

        nk_abundances = {}
        for nkfile, frac in (nk_comp or {}).items():
            if frac < 0:
                raise ValueError(f'Abundance of {nkfile} must be non-negative')
            path = self.find_nk_file(nkfile)
            nk_abundances[path] = nk_abundances.get(path, 0.0) + frac
        # DUSTY reads at most 10 additional components (nameNK(10))
        if len(nk_abundances) > 10:
            raise ValueError('DUSTY supports at most 10 additional (.nk) components')

        total = sum(abundances.values()) + sum(nk_abundances.values())
        if total <= 0:
            raise ValueError('At least one grain type needs a non-zero abundance')

        return ({k: v/total for k, v in abundances.items()},
                {k: v/total for k, v in nk_abundances.items()})

    def find_nk_file(self, nkfile):
        # DUSTY opens .nk files relative to its working directory (a temp dir here),
        # so resolve to an absolute path: as given, then $DUSTY_PATH/Lib_nk, then $DUSTY_PATH
        for path in [Path(nkfile),
                     Path(self.dusty_basedir) / 'Lib_nk' / nkfile,
                     Path(self.dusty_basedir) / nkfile]:
            if path.is_file():
                path = str(path.resolve())
                # FileMSG in dustyV2.f reads file names into a CHARACTER*230
                if len(path) >= 230:
                    raise ValueError(f'Path to {nkfile} is too long for DUSTY (>= 230 characters)')
                return path

        raise FileNotFoundError(f'{nkfile} not found as given, in {self.dusty_basedir}/Lib_nk '
                                f'or in {self.dusty_basedir}')

    def get_params(self, theta):
        # map the free parameter vector onto the full parameter dictionary
        if len(theta) != len(self.free_params):
            raise ValueError(f'Expected {len(self.free_params)} free parameters '
                             f'{self.free_params}, got {len(theta)}')
        p = dict(self.fixed)
        p.update(zip(self.free_params, theta))

        if not (self.rsg_temp.value.min() <= p['temp'] <= self.rsg_temp.value.max()):
            raise ValueError(f'temp = {p['temp']} K is outside the {self.modeltype} grid '
                             f'({self.rsg_temp.value.min()}-{self.rsg_temp.value.max()} K)')
        if p['tau'] <= 0:
            raise ValueError('tau must be positive (DUSTY fails at tau = 0)')
        if p['shell_thickness'] <= 1:
            raise ValueError('shell_thickness (Y = R_out/R_in) must be > 1')

        return p

    def get_workdir(self):
        # one DUSTY working directory per process, since DUSTY reads dusty.inp and
        # lambda_grid.dat from its cwd
        pid = os.getpid()
        if pid not in self._workdirs:
            if self.workdir_base is not None:
                os.makedirs(self.workdir_base, exist_ok=True)
            workdir = tempfile.mkdtemp(prefix=f'dusty_{pid}_', dir=self.workdir_base)
            for fl_ in ['dusty', 'lambda_grid.dat']:
                os.symlink(os.path.join(self.dusty_basedir, fl_), os.path.join(workdir, fl_))
            self._workdirs[pid] = workdir

        return self._workdirs[pid]

    def get_input_spec(self, temp, workdir):
        # input spectra are cached per process, since temp is usually fixed or
        # tightly priored
        key = (os.getpid(), float(temp))
        if key not in self._input_spec:
            self._input_spec[key] = self.setup_input_spec(temp=temp*u.K, outdir=workdir)

        return self._input_spec[key]

    def dusty_comp_block(self):
        # 2.1 chemical composition. Built-in grains only -> optical properties
        # index 1; with additional .nk files -> index 2, which appends the file
        # list and their abundances after the built-in abundances (DUSTY manual 3.2.1)
        top = 2 if self.nk_abundances else 1
        lines = ['       2.1 Chemical composition\n',
                 f'           Optical properties index = {top}\n',
                 '           Abundances for supported grain types:\n',
                 '               Sil-Ow  Sil-Oc  Sil-DL  grf-DL  amC-Hn  SiC-Pg\n',
                 '           x = ' + '  '.join(f'{self.abundances[g]:.4f}' for g in self.DUSTY_GRAINS) + '\n\n']

        if self.nk_abundances:
            lines.append(f'           Number of additional components = {len(self.nk_abundances)}, '
                         'properties listed in files\n')
            for path in self.nk_abundances:
                lines.append(f'               {path}\n')
            lines.append('           Abundances for these components = ' +
                         ', '.join(f'{x:.4f}' for x in self.nk_abundances.values()) + '\n\n')

        return ''.join(lines)

    def dusty_size_block(self):
        # 2.2 grain size distribution
        lines = ['       2.2 Grain size distribution\n\n']
        if self.size_dist == 'mrn':
            lines.append('          Size distribution = 2 % arbitrary MRN\n')
            lines.append(f'          q = {self.size_par['q']}, a(min) = {self.size_par['amin']} micron, '
                         f'a(max) = {self.size_par['amax']} micron\n\n')
        else:
            lines.append('          Size distribution = 3 % KMH\n')
            lines.append(f'          q = {self.size_par['q']}, a(min) = {self.size_par['amin']} micron, '
                         f'a0 = {self.size_par['a0']} micron\n\n')

        return ''.join(lines)

    def dusty_density_block(self, p):
        # 3) density distribution. DUSTY input density type 4 is the analytic
        # radiatively driven wind and only takes the relative thickness Y; 
        # type 1 is a single power law
        lines = ['     3) Density Distribution\n']
        if self.density == 'rdw':
            lines.append('        - density type = 4\n')
            lines.append(f'        - Y = {p['shell_thickness']}\n\n')
        elif self.density == 'exp':
            # input density type 5 reads a (y, eta) table from file; DUSTY takes Y
            # from its last row (see write_density_file)
            lines.append('        - density type = 5\n')
            lines.append(f'            {p['eta_file']}\n\n')
        else:
            lines.append('        - density type = 1\n')
            lines.append('        - number of powers = 1\n')
            lines.append(f'        - shell\'s relative thickness = {p['shell_thickness']}\n')
            lines.append(f'        - power = {p.get('density_power', self.density_power)}\n\n')

        return ''.join(lines)

    def write_density_file(self, p, workdir):
        # tabulated exponential mass-loss profile for density type 5. Rising profiles
        # are truncated where eta reaches DUSTY's dynamic range; p['Y_eff'] records it
        y, eta, p['Y_eff'] = exp_density_profile(p['u'], p['shell_thickness'])
        p['eta_file'] = os.path.join(workdir, 'model.eta')
        with open(p['eta_file'], 'w') as f:
            f.write(f'# rho ~ y^-2 exp[-u (y-1)], u = {p['u']}, Y = {p['shell_thickness']}, '
                    f'Y_eff = {p['Y_eff']}\n#   y   eta (relative)\n#\n')
            np.savetxt(f, np.c_[y, eta], fmt='%.8e')
        # FileMSG in dustyV2.f reads file names into a CHARACTER*230
        if len(p['eta_file']) >= 230:
            raise ValueError(f'Path to {p['eta_file']} is too long for DUSTY (>= 230 characters)')

    def write_tau_file(self, taus, workdir):
        # tau grid for grid type 3. DUSTY sorts it and segfaults on a single entry
        taus = np.sort(np.asarray(taus, dtype=float))
        if len(taus) < 2:
            raise ValueError('A DUSTY tau sequence needs at least two optical depths')
        if len(taus) > 999:
            raise ValueError('DUSTY runs at most 999 models per input file')
        with open(os.path.join(workdir, 'model.tau'), 'w') as f:
            f.write('lambda0 = 0.55 micron\n')
            for t in taus:
                f.write(f'   {t:.6e}\n')
        # dustyV2.f declares nameTAU as CHARACTER*70, so pass the name relative to
        # the DUSTY working directory rather than as a full path
        return 'model.tau'

    def setup_dusty_single(self, inp_file, p):
        # setup input file for a single DUSTY model at p['tau'], or for the tau
        # sequence in p['tau_file'] (grid type 3)
        with open(inp_file, 'w') as f:
            f.write('  I PHYSICAL PARAMETERS\n')
            f.write('     1) External radiation:\n')
            f.write('                Spectrum = 5\n')
            f.write(f'                {p['input_spec']}\n')
            f.write('     2) Dust Properties\n\n')
            f.write(self.dusty_comp_block())
            f.write(self.dusty_size_block())
            f.write('       2.3 Dust temperature on inner boundary:\n\n')
            f.write(f'        - temperature = {p['dust_temp']} K\n\n')
            f.write(self.dusty_density_block(p))
            f.write('     4) Optical Depth\n')
            if p.get('tau_file'):
                f.write('        - grid type = 3\n')
                f.write(f'            {p['tau_file']}\n')
            else:
                f.write('        - grid type = 1\n')
                f.write('        - lambda0 = 0.55 micron\n')
                f.write(f'        - tau(min) = {p['tau']}; tau(max) = {p['tau']}\n')
                f.write('        - number of models = 1\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  II NUMERICS\n\n')
            f.write(f'     - accuracy for flux conservation = {self.accuracy}\n\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  III OUTPUT PARAMETERS\n\n')
            f.write('        FILE DESCRIPTION                               FLAG\n')
            f.write('       ------------------------------------------------------------\n')
            f.write('       - verbosity flag;                               verbose = 0\n')
            f.write('       - properties of emerging spectra;             fname.spp = 0\n')
            f.write('       - detailed spectra for each model;           fname.s### = 2\n')
            f.write('       - images at specified wavelengths;           fname.i### = 0\n')
            f.write('       - visibility function at spec. wavelengths;  fname.v### = 0\n')
            f.write('       - radial profiles for each model;            fname.r### = 0\n')
            f.write('       - detailed run-time messages;                fname.m### = 2\n')
            f.write('       -------------------------------------------------------------\n\n')
            f.write('  The end of the input parameters listing.\n')

    def read_dusty_results_all(self, outfile):
        # read every row of the RESULTS table in the .out file as {model number:
        # {column: value}}, keyed by the column names in the header (e.g. 'r1(cm)',
        # 'Td(Y)', 'err'). Models DUSTY did not finish have no row.
        with open(outfile) as f:
            lines = f.readlines()

        for n, line in enumerate(lines):
            if line.lstrip().startswith('###') and 'tau0' in line:
                names = line.split()[1:]
                break
        else:
            raise DustyError(f'No RESULTS table in {outfile}, the DUSTY run probably failed')

        rows = {}
        for line in lines[n+1:]:
            s = line.split()
            if s and s[0].isdigit():
                rows[int(s[0])] = dict(zip(names, [float(v) for v in s[1:len(names)+1]]))
            elif rows and s and s[0].startswith('='):
                break

        return rows

    def read_dusty_results(self, outfile):
        # the single-model row of the RESULTS table
        rows = self.read_dusty_results_all(outfile)
        if not rows:
            raise DustyError(f'RESULTS table in {outfile} has no models')
        return rows[min(rows)]

    def read_dusty_status(self, outfile):
        # per-model error and warning flags from the .out file: DUSTY writes
        # 'There are some error messages for model: N' (or 'warnings for model')
        with open(outfile) as f:
            out = f.read()
        errors = {int(m) for m in re.findall(r'error messages for model:\s*(\d+)', out)}
        warnings = {int(m) for m in re.findall(r'warnings for model:\s*(\d+)', out)}
        return 'FATAL ERROR' in out, errors, warnings

    def check_dusty_status(self, outfile, msgfile, results, p):
        # DUSTY flags problems in the .out file and details them in the .m### file.
        # Errors and failed flux conservation raise DustyError; warnings are returned.
        with open(outfile) as f:
            out = f.read()
        messages = ''
        if os.path.exists(msgfile):
            with open(msgfile) as f:
                messages = f.read().strip()

        if 'FATAL ERROR' in out or 'error messages for model' in out:
            raise DustyError(f'DUSTY reported errors for {p}:\n{messages}')

        # err is the maximum flux conservation error in %, (1-r)/(1+r) with r the
        # ratio of smallest to largest bolometric flux in the shell (< 1% is listed as 0)
        if results.get('err', 0.0) > 100*self.accuracy:
            raise DustyError(f'DUSTY flux conservation error {results['err']}% exceeds '
                             f'the required {100*self.accuracy}% for {p}')

        warnings = messages if 'warnings for model' in out else ''
        return warnings

    def clear_dusty_outputs(self, basename):
        # remove the outputs of a previous run, so stale spectra are never read
        for fl_ in glob.glob(glob.escape(basename) + '.*'):
            if not fl_.endswith(('.inp', '.dat')):
                os.remove(fl_)

    def run_dusty_single(self, p):
        workdir = self.get_workdir()
        p['input_spec'] = self.get_input_spec(p['temp'], workdir)

        basename = os.path.join(workdir, 'model')
        inp_file, outfile, specfile = basename + '.inp', basename + '.out', basename + '.s001'
        msgfile = basename + '.m001'
        self.clear_dusty_outputs(basename)

        if self.density == 'exp':
            self.write_density_file(p, workdir)
        self.setup_dusty_single(inp_file, p)
        with open(os.path.join(workdir, 'dusty.inp'), 'w') as f:
            f.write(basename)

        run = subprocess.run(['./dusty'], cwd=workdir, capture_output=True)
        if run.returncode != 0 or not os.path.exists(specfile):
            raise DustyError(f'DUSTY failed for {p}:\n{run.stdout.decode()}{run.stderr.decode()}')

        results = self.read_dusty_results(outfile)
        results['warnings'] = self.check_dusty_status(outfile, msgfile, results, p)
        if 'Y_eff' in p:
            results['Y_eff'] = p['Y_eff']
        spec = self.read_dusty_spec(specfile)

        if not self.keep_files:
            self.clear_dusty_outputs(basename)

        return spec, results

    @staticmethod
    def read_dusty_spec(specfile):
        return ascii.read(specfile, names=['lambda', 'fTot', 'xAtt', 'xDs', 'xDe', 'fInp', 'tauT', 'albedo'],
                          format='basic', data_start=0)

    def run_dusty_sequence(self, p, taus):
        '''
        Run DUSTY once over a sequence of optical depths (grid type 3) at the other
        parameters in p, which is much cheaper than one run per tau.

        Returns a list over the sorted taus of (spec, results), with (None, results)
        for models that failed or missed the flux conservation accuracy; results
        then holds the reason under 'failed'. Raises DustyError if the whole run
        failed (e.g. a density profile DUSTY cannot handle).
        '''
        workdir = self.get_workdir()
        p = dict(p)
        p['input_spec'] = self.get_input_spec(p['temp'], workdir)
        taus = np.sort(np.asarray(taus, dtype=float))

        basename = os.path.join(workdir, 'model')
        inp_file, outfile = basename + '.inp', basename + '.out'
        self.clear_dusty_outputs(basename)

        if self.density == 'exp':
            self.write_density_file(p, workdir)
        p['tau_file'] = self.write_tau_file(taus, workdir)
        self.setup_dusty_single(inp_file, p)
        with open(os.path.join(workdir, 'dusty.inp'), 'w') as f:
            f.write(basename)

        run = subprocess.run(['./dusty'], cwd=workdir, capture_output=True)
        if run.returncode != 0 or not os.path.exists(outfile):
            raise DustyError(f'DUSTY failed for {p}:\n{run.stdout.decode()}{run.stderr.decode()}')
        fatal, errors, warnings = self.read_dusty_status(outfile)
        if fatal:
            raise DustyError(f'DUSTY reported a fatal error for {p}')
        rows = self.read_dusty_results_all(outfile)
        if not rows:
            with open(outfile) as f:
                tail = f.read()[-600:]
            raise DustyError(f'DUSTY produced no models for {p}:\n{tail}')

        out = []
        for i, tau in enumerate(taus):
            model = i + 1
            res = dict(rows.get(model, {}))
            res['tau'] = tau
            if 'Y_eff' in p:
                res['Y_eff'] = p['Y_eff']
            res['warnings'] = model in warnings
            specfile = f'{basename}.s{model:03d}'
            if model not in rows:
                res['failed'] = 'not computed'
            elif model in errors:
                res['failed'] = 'DUSTY error'
            elif not np.isclose(res['tau0'], tau, rtol=0.02):
                res['failed'] = f'tau mismatch ({res["tau0"]} vs {tau})'
            elif res.get('err', 0.0) > 100*self.accuracy:
                res['failed'] = f'flux conservation error {res["err"]}%'
            elif not os.path.exists(specfile):
                res['failed'] = 'no spectrum'
            out.append((None if 'failed' in res else self.read_dusty_spec(specfile), res))

        if not self.keep_files:
            self.clear_dusty_outputs(basename)

        return out

    def spec_to_sed(self, spec, logl):
        # DUSTY spectrum -> QTable with 'lambda' (micron), 'flam' (erg/s/cm2/A) and
        # 'fnu' (uJy) at self.dist for luminosity 10**logl Lsun
        # the stitched DUSTY wavelength grid repeats the boundary wavelengths
        wv, idx = np.unique(np.asarray(spec['lambda']), return_index=True)
        wv = wv * u.um

        # fTot is lambda*F_lambda/F_bol; scale to the requested luminosity
        flam = np.asarray(spec['fTot'])[idx] / wv.value
        flam = self.scale_flux(wv, flam, 10**logl * u.Lsun)
        fnu = flam.to(u.uJy, equivalencies=u.spectral_density(wv))

        sed = QTable()
        sed['lambda'] = wv
        sed['flam'] = flam
        sed['fnu'] = fnu
        return sed

    def config_meta(self):
        # the discrete configuration of the models (photosphere, dust, density law)
        meta = {'photosphere': self.rsg_tag,
                'comp': self.abundances,
                'nk_comp': self.nk_abundances,
                'size_dist': self.size_dist,
                'size_par': self.size_par,
                'density': self.density,
                'accuracy': self.accuracy}
        if self.density == 'powerlaw':
            meta['density_power'] = self.density_power
        if self.density == 'exp':
            meta['eta_floor'] = ETA_FLOOR
        return meta

    def model_sed(self, theta):
        '''
        Primary model function: free parameters -> observed SED at self.dist.

        Returns a QTable with 'lambda' (micron), 'flam' (erg/s/cm2/A) and 'fnu' (uJy)
        for the total (attenuated star + scattered + dust emission) spectrum. The
        DUSTY results are stored in the table meta, with r_in scaled to logl.
        '''
        p = self.get_params(theta)
        spec, results = self.run_dusty_single(p)
        sed = self.spec_to_sed(spec, p['logl'])

        sed.meta = {k: p[k] for k in self.param_names}
        sed.meta.update(self.config_meta())
        sed.meta['dusty_results'] = results
        # R_in scales as L^1/2 at fixed T_in
        r1 = results.get('r1(cm)', np.nan) * u.cm
        sed.meta['r_in'] = r1 * np.sqrt((10**p['logl'] * u.Lsun / self.DUSTY_LUM).decompose().value)
        sed.meta['dusty_err'] = results.get('err', 0)
        sed.meta['dusty_warnings'] = results['warnings']

        return sed

    @staticmethod
    def tapered(bp):
        # stsynphot bandpasses (e.g. HST ACS/WFC3 from stsynphot.band) disable taper(), so
        # tabulate them as a plain SpectralElement first
        try:
            return bp.taper()
        except NotImplementedError:
            return SpectralElement(Empirical1D, points=bp.waveset, lookup_table=bp(bp.waveset)).taper()

    def model_phot(self, theta, bandpasses, sed=None):
        '''
        Synthetic photometry (uJy) of model_sed(theta) through a dict of
        {filter name: synphot.SpectralElement}.
        '''
        if sed is None:
            sed = self.model_sed(theta)

        wv = sed['lambda'].to(u.Angstrom)
        sp = synphot.SourceSpectrum(Empirical1D, points=wv, lookup_table=sed['flam'])

        phot = {}
        for filt, bp in bandpasses.items():
            # taper the bandpass to zero at its ends: curves that stop at non-zero
            # transmission (pandeia MIRI F2550W ends at 30 um with T ~ 0.6) are otherwise
            # extrapolated over the whole long-wavelength model grid
            obs = synphot.Observation(sp, self.tapered(bp), force='taper', binset=wv)
            try:
                phot[filt] = obs.effstim(u.uJy).value
            except synphot.exceptions.SynphotError:
                # the most obscured models have exactly zero flux across the bluest
                # bands; that is a property of the model, not a failure (cf. FAINT_MAG
                # in rsg.py)
                phot[filt] = 0.0

        return phot

    # ------------------------------------------------------------------------
    # Precomputed grids
    #
    # The grid is built at DUSTY's reference luminosity, 1e4 Lsun: for a fixed input
    # spectral shape, T_in, tau and dimensionless density profile the SED shape does
    # not depend on L, so at fit time flux scales as L/1e4 and R_in as sqrt(L/1e4).
    # One grid holds one discrete configuration (photosphere log g and C/O, dust
    # mixture, grain sizes); compare configurations by evidence.
    #
    # The density is a single power law rho ~ r^-s from R_in to Y R_in, with s a grid
    # axis (DUSTY density type 1), so the grid is a full rectangle in
    # (temp, dust_temp, log tau, s, log Y).

    GRID_LOGL = 4.0
    GRID_AXES = ('temp', 'dust_temp', 'tau', 's', 'Y')

    def grid_cell(self, temp, dust_temp, s, Y, taus, bandpasses, save_sed=False):
        '''
        One grid cell: a single DUSTY run over the tau sequence at fixed (temp,
        dust_temp, s, Y), with synthetic photometry (uJy at self.dist, L = 1e4 Lsun),
        r1, flux conservation error and Td(Y) for every tau. Failed models are NaN
        with ok = False; a failed run leaves the whole cell NaN with 'msg' set.
        '''
        taus = np.sort(np.asarray(taus, dtype=float))
        nt, nb = len(taus), len(bandpasses)
        out = {'phot': np.full((nt, nb), np.nan), 'r1': np.full(nt, np.nan), 'err': np.full(nt, np.nan),
               'td_y': np.full(nt, np.nan), 'ok': np.zeros(nt, bool), 'warn': np.zeros(nt, bool),
               'msg': '', 'runtime': 0.0}
        p = {'logl': self.GRID_LOGL, 'temp': temp, 'dust_temp': dust_temp, 'shell_thickness': Y,
             'density_power': s}
        t0 = time.time()
        try:
            seq = self.run_dusty_sequence(p, taus)
        except (DustyError, ValueError) as e:
            out['msg'] = str(e)[:1000]
            out['runtime'] = time.time() - t0
            return out

        for i, (spec, res) in enumerate(seq):
            out['warn'][i] = res['warnings']
            out['r1'][i] = res.get('r1(cm)', np.nan)
            out['err'][i] = res.get('err', np.nan)
            out['td_y'][i] = res.get('Td(Y)', np.nan)
            if spec is None:
                continue
            sed = self.spec_to_sed(spec, self.GRID_LOGL)
            phot = self.model_phot(None, bandpasses, sed=sed)
            out['phot'][i] = [phot[b] for b in bandpasses]
            out['ok'][i] = True
            if save_sed:
                if 'sed' not in out:
                    out['sed_lambda'] = sed['lambda'].value
                    out['sed'] = np.full((nt, len(sed)), np.nan, dtype=np.float32)
                out['sed'][i] = sed['fnu'].value
        out['runtime'] = time.time() - t0
        return out

    def default_grid_axes(self):
        '''
        Default grid axes, chosen so that DustyGrid's PCHIP interpolation stays within
        ~0.02 mag of DUSTY for detectable fluxes (> 1e-3 uJy at 19 Mpc for 1e4 Lsun).
        The spacings come from hold-out tests on 1D cuts through COMARCS + amC/SiC
        power-law models (leave nodes out and interpolate them from the rest); the
        largest PCHIP errors over NIRCam/MIRI bands were:
            temp       every model atmosphere in 2500-4000 K (100 K for COMARCS);
                       200 K steps already give <= 0.04 mag
            dust_temp  70 K steps over 400-1400 K (<= 0.01 mag; 100 K gives 0.03),
                       ~110 K above, where the SED barely changes with T_in
                       (105 K steps give 0.001 mag)
            tau        0.01-60 in 0.145 dex steps; 0.15 dex gives <= 0.011 mag at
                       s = 1 and 3.5 (linear interpolation would need < 0.05 dex)
            s          0.1 steps over 1-2.5, 0.25 over 2.5-3.5 (<= 0.02 mag)
            Y          10-1e4 in 0.2 dex steps; 0.25 dex gives 0.031 mag at s = 1,
                       where most dust is far out, and 0.009 at s = 1.5
        '''
        temps = self.rsg_temp.value
        return {'temp': temps[(temps >= 2500 - 1e-6) & (temps <= 4000 + 1e-6)],
                'dust_temp': np.concatenate([np.arange(400., 1400., 70.), [1500., 1600., 1700., 1800.]]),
                'tau': np.round(np.logspace(-2, np.log10(60.), 27), 5),
                's': np.round(np.concatenate([np.arange(1.0, 2.5 + 1e-6, 0.1), np.arange(2.75, 3.5 + 1e-6, 0.25)]), 3),
                'Y': np.round(np.logspace(1, 4, 16), 4)}

    def build_grid(self, outfile, bandpasses, temps=None, dust_temps=None, taus=None, ss=None, Ys=None,
                   nproc=1, save_sed=False, assemble=True, retry_failed=False):
        '''
        Build a rectangular grid over (temp, dust_temp, tau, s, Y) for density='powerlaw',
        in the style of dusty_gen.gen_grid: every (temp, dust_temp, s, Y) cell is one
        DUSTY run over all taus, at L = 1e4 Lsun. Axes left as None take the values of
        default_grid_axes.

        Each cell is written to its own shard in <outfile>.shards/ as it finishes, so
        an interrupted run resumes where it stopped; the shards are then merged into
        outfile (see assemble_grid). retry_failed=True also reruns cells whose DUSTY
        run failed as a whole.

        bandpasses is a dict {filter name: synphot.SpectralElement}. The cost is set by
        the largest taus: DUSTY's run time per model grows roughly linearly with tau
        above tau ~ 10, and with s: the default 27-tau sequence takes ~1.5-2 min at
        s <= 2, ~4 min at s = 2.25-3.25 and ~12 min at s = 3.5 (one of 10 parallel
        processes).
        '''
        if self.density != 'powerlaw':
            raise ValueError('build_grid needs density="powerlaw" (the power s is a grid axis)')
        given = {'temp': temps, 'dust_temp': dust_temps, 'tau': taus, 's': ss, 'Y': Ys}
        axes = self.default_grid_axes()
        axes.update({k: v for k, v in given.items() if v is not None})
        axes = {k: np.unique(np.asarray(axes[k], dtype=float)) for k in self.GRID_AXES}
        if (axes['tau'] <= 0).any() or (axes['Y'] <= 1).any():
            raise ValueError('taus must be > 0 and Ys > 1')
        if len(axes['tau']) < 2:
            raise ValueError('A DUSTY tau sequence needs at least two optical depths')
        lo, hi = self.rsg_temp.value.min(), self.rsg_temp.value.max()
        if axes['temp'].min() < lo or axes['temp'].max() > hi:
            raise ValueError(f'temps must lie within the {self.modeltype} grid ({lo}-{hi} K)')

        shard_dir = Path(str(outfile) + '.shards')
        shard_dir.mkdir(parents=True, exist_ok=True)
        config = {'axes': {k: v.tolist() for k, v in axes.items()},
                  'bands': list(bandpasses), 'save_sed': save_sed, 'dist_mpc': self.dist.to(u.Mpc).value,
                  'logl': self.GRID_LOGL, 'meta': self.grid_meta(),
                  'dusty_path': self.dusty_basedir, 'npL': len(self.read_lambda_grid())}
        config = json.loads(json.dumps(config, default=float))
        config_file = shard_dir / 'config.json'
        if config_file.exists():
            old = json.loads(config_file.read_text())
            if old != config:
                raise ValueError(f'{shard_dir} holds shards of a different grid; '
                                 'use another outfile or remove the directory')
        else:
            config_file.write_text(json.dumps(config, indent=1))

        cells = [(it, jt, js, iy)
                 for it in range(len(axes['temp'])) for jt in range(len(axes['dust_temp']))
                 for js in range(len(axes['s'])) for iy in range(len(axes['Y']))]
        def done(c):
            fl_ = shard_dir / shard_name(c)
            return fl_.exists() and not (retry_failed and str(np.load(fl_)['msg']))
        todo = [c for c in cells if not done(c)]
        print(f'{len(cells)} cells ({len(cells)*len(axes["tau"])} models); {len(todo)} still to run')

        # bandpasses go to the workers as plain arrays: pickling a synphot Empirical1D
        # changes how it extrapolates (see _init_worker in rsg.py)
        args = (self, bandpass_tables(bandpasses), axes, save_sed, str(shard_dir))
        t0 = time.time()
        if nproc > 1:
            import multiprocessing as mp
            with mp.get_context('spawn').Pool(nproc, initializer=_grid_worker_init, initargs=args) as pool:
                for n, msg in enumerate(pool.imap_unordered(_grid_worker, todo), 1):
                    report_progress(n, len(todo), t0, msg)
        else:
            _grid_worker_init(*args)
            for n, cell in enumerate(todo, 1):
                report_progress(n, len(todo), t0, _grid_worker(cell))

        if assemble:
            return assemble_grid(outfile)

    def grid_meta(self):
        # configuration of a grid: as config_meta, without the fixed power-law index,
        # since s is a grid axis
        meta = self.config_meta()
        meta.pop('density_power', None)
        return meta

def bandpass_tables(bandpasses):
    # {name: (wavelength in Angstrom, throughput)} of the tapered bandpasses
    out = {}
    for k, bp in bandpasses.items():
        bp = dusty_likelihood.tapered(bp)
        wv = bp.waveset.to(u.Angstrom).value
        out[k] = (wv, np.asarray(bp(wv * u.Angstrom).value))
    return out

def bandpasses_from_tables(tables):
    return {k: SpectralElement(Empirical1D, points=wv * u.Angstrom, lookup_table=thru)
            for k, (wv, thru) in tables.items()}

def shard_name(cell):
    return 'cell_{:03d}_{:03d}_{:03d}_{:03d}.npz'.format(*cell)

def report_progress(n, ntot, t0, msg):
    if msg or n == ntot or n % max(1, ntot // 100) == 0:
        el = time.time() - t0
        print(f'[{n}/{ntot}] {el/3600:.2f} h elapsed, ~{el/n*(ntot-n)/3600:.2f} h left'
              + (f'  {msg}' if msg else ''), flush=True)

# state of a grid worker process, set once by _grid_worker_init
_GRID_WORKER = {}

def _grid_worker_init(builder, tables, axes, save_sed, shard_dir):
    _GRID_WORKER.update(builder=builder, axes=axes, save_sed=save_sed, shard_dir=Path(shard_dir),
                        bandpasses=bandpasses_from_tables(tables))

def _grid_worker(cell):
    w = _GRID_WORKER
    it, jt, js, iy = cell
    ax = w['axes']
    out = w['builder'].grid_cell(ax['temp'][it], ax['dust_temp'][jt], ax['s'][js], ax['Y'][iy],
                                 ax['tau'], w['bandpasses'], save_sed=w['save_sed'])
    # write atomically, so a killed run never leaves a truncated shard
    tmp = w['shard_dir'] / ('tmp_' + shard_name(cell))
    np.savez(tmp, **out)
    os.replace(tmp, w['shard_dir'] / shard_name(cell))
    if out['msg']:
        return f'cell {cell} failed: {out["msg"].splitlines()[0][:150]}'
    nbad = int((~out['ok']).sum())
    return f'cell {cell}: {nbad} of {len(out["ok"])} models failed' if nbad else ''

# status codes of grid models
GRID_OK, GRID_FAILED, GRID_MISSING = 1, 2, 3

def assemble_grid(outfile):
    '''
    Merge the shards in <outfile>.shards/ into one .npz grid file holding the axes,
    photometry 'phot' (uJy at dist_mpc, L = 1e4 Lsun; shape temp x dust_temp x tau x
    s x Y x band), 'r1' (cm, for 1e4 Lsun), 'err' (%), 'td_y' (K), 'warn', 'status'
    (GRID_OK, GRID_FAILED, or GRID_MISSING if not run yet) and the metadata as a JSON
    string.
    '''
    shard_dir = Path(str(outfile) + '.shards')
    config = json.loads((shard_dir / 'config.json').read_text())
    axes = {k: np.asarray(v) for k, v in config['axes'].items()}
    shape = tuple(len(axes[k]) for k in DustyGrid.AXES)
    nb = len(config['bands'])
    phot = np.full(shape + (nb,), np.nan)
    arrs = {k: np.full(shape, np.nan) for k in ['r1', 'err', 'td_y']}
    warn = np.zeros(shape, bool)
    status = np.full(shape, GRID_MISSING, dtype=np.int8)
    runtime = 0.0
    seds, sed_lambda = None, None

    for it in range(shape[0]):
        for jt in range(shape[1]):
            for js in range(shape[3]):
                for iy in range(shape[4]):
                    fl_ = shard_dir / shard_name((it, jt, js, iy))
                    if not fl_.exists():
                        continue
                    idx = np.s_[it, jt, :, js, iy]
                    d = np.load(fl_)
                    phot[idx] = d['phot']
                    for k in arrs:
                        arrs[k][idx] = d[k]
                    warn[idx] = d['warn']
                    status[idx] = np.where(d['ok'], GRID_OK, GRID_FAILED)
                    runtime += float(d['runtime'])
                    if 'sed' in d:
                        if seds is None:
                            sed_lambda = d['sed_lambda']
                            seds = np.full(shape + (len(sed_lambda),), np.nan, dtype=np.float32)
                        seds[idx] = d['sed']

    meta = dict(config)
    meta.update({'units': {'phot': 'uJy at dist_mpc for L = 1e4 Lsun', 'r1': 'cm for L = 1e4 Lsun',
                           'err': 'percent', 'td_y': 'K'},
                 'n_ok': int((status == GRID_OK).sum()), 'n_failed': int((status == GRID_FAILED).sum()),
                 'n_missing': int((status == GRID_MISSING).sum()), 'cpu_hours': runtime / 3600})
    out = dict(phot=phot, warn=warn, status=status, meta=json.dumps(meta), **arrs,
               **{k: axes[k] for k in DustyGrid.AXES})
    if seds is not None:
        out.update(sed=seds, sed_lambda=sed_lambda)
    np.savez(outfile, **out)
    print(f'{outfile}: {meta["n_ok"]} models ok, {meta["n_failed"]} failed, {meta["n_missing"]} missing; '
          f'{meta["cpu_hours"]:.1f} CPU hours')
    return outfile

def _pchip_edge(h0, h1, m0, m1):
    # one-sided three-point end slope, as in scipy.interpolate.PchipInterpolator
    d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
    d = np.where(np.sign(d) != np.sign(m0), 0.0, d)
    return np.where((np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3*np.abs(m0)), 3*m0, d)

def _pchip_interior(h0, h1, m0, m1):
    # weighted harmonic mean of the neighbouring secants, and 0 at a local extremum
    w1, w2 = 2*h1 + h0, h1 + 2*h0
    d = (w1 + w2) / (w1/m0 + w2/m1)
    return np.where((np.sign(m0) != np.sign(m1)) | (m0 == 0) | (m1 == 0), 0.0, d)

def local_pchip(axes, values, pts):
    '''
    Tensor-product PCHIP on a rectilinear grid, at points pts (npts, ndim) inside it.
    values has shape (len(axes[0]), ..., len(axes[-1]), nval).

    PCHIP slopes at a node only depend on its two neighbours, so the interpolant in
    any cell only needs the 4 nodes around it along each axis. Each axis is reduced in
    turn over that 4^ndim stencil, which costs O(4^ndim) per point. This is the same
    interpolant as scipy's RegularGridInterpolator(method='pchip') (up to the order the
    axes are reduced in), which instead works on the whole grid for every point and is
    far too slow for a likelihood. Points outside the axes are extrapolated, so check
    the bounds first. A NaN anywhere in a point's stencil makes its result NaN.
    '''
    pts = np.atleast_2d(pts)
    npts, ndim = pts.shape
    stencils = []
    for d, x in enumerate(axes):
        n = len(x)
        i = np.clip(np.searchsorted(x, pts[:, d], side='right') - 1, 0, n - 2)
        # nodes i-1 .. i+2, repeated at the ends of the axis
        st = np.clip(i[:, None] + np.arange(-1, 3), 0, n - 1)
        stencils.append((st, i, n))
    block = values[tuple(st[(slice(None),) + (None,)*d + (slice(None),) + (None,)*(ndim - d - 1)]
                         for d, (st, _, _) in enumerate(stencils))]

    # block is (npts, 4, ..., 4, nval); reduce the first remaining stencil axis each time
    with np.errstate(divide='ignore', invalid='ignore'):
        for d, (st, i, n) in enumerate(stencils):
            r = lambda a: a.reshape((npts,) + (1,)*(block.ndim - 2))
            xn = axes[d][st]
            y = [block[:, k] for k in range(4)]
            h = [r(xn[:, k+1] - xn[:, k]) for k in range(3)]
            m = [(y[k+1] - y[k]) / h[k] for k in range(3)]
            if n == 2:
                d0 = d1 = m[1]
            else:
                # at the ends of the axis the repeated node has h = 0, so use the
                # one-sided end slope there
                d0 = np.where(r(i == 0), _pchip_edge(h[1], h[2], m[1], m[2]),
                              _pchip_interior(h[0], h[1], m[0], m[1]))
                d1 = np.where(r(i + 1 == n - 1), _pchip_edge(h[1], h[0], m[1], m[0]),
                              _pchip_interior(h[1], h[2], m[1], m[2]))
            s = r((pts[:, d] - xn[:, 1]) / (xn[:, 2] - xn[:, 1]))
            s2, s3 = s*s, s*s*s
            block = ((2*s3 - 3*s2 + 1)*y[1] + (s3 - 2*s2 + s)*h[1]*d0
                     + (3*s2 - 2*s3)*y[2] + (s3 - s2)*h[1]*d1)

    return block

class DustyGrid(object):
    '''
    Interpolation of a grid from dusty_likelihood.build_grid in (temp, dust_temp,
    log tau, s, log Y), of log10 flux per band and log10 r1.

    method = 'pchip' (default) is tensor-product PCHIP (see local_pchip): C1, exact
    at the nodes and monotone between them, so it cannot overshoot where the flux
    changes fast (high tau, the SiC feature turning over). Where a point's 4^5
    stencil holds a failed model it falls back to linear. method = 'linear' is
    multilinear, like the rsg.py grids.

    Luminosity is the sixth axis, with nodes at log L = 3 and 6 only (LOGL_RANGE): the
    SED shape does not depend on L, so log flux is exactly linear in log L, and
    interpolating between the two nodes is the analytic shift log F + (log L - 4).
    It is applied that way rather than as another interpolation axis, which would
    double the cost of every call. r1 scales as sqrt(L / 1e4).

    Points next to failed models (both methods: in the cell itself) and points
    outside the axes interpolate to NaN and count as outside the grid.
    '''
    AXES = dusty_likelihood.GRID_AXES
    LOG_AXES = ('tau', 'Y')
    METHODS = ('pchip', 'linear')
    LOGL_RANGE = (3.0, 6.0)
    # uJy at the grid distance; far below any detection
    FLUX_FLOOR = 1e-30

    def __init__(self, gridfile, method='pchip'):
        if method not in self.METHODS:
            raise ValueError(f'method must be one of {self.METHODS}')
        self.method = method
        self.gridfile = str(gridfile)
        d = np.load(gridfile)
        self.meta = json.loads(str(d['meta']))
        self.bands = list(self.meta['bands'])
        self.axes = {k: d[k] for k in self.AXES}
        self.dist = self.meta['dist_mpc'] * u.Mpc
        self.status = d['status']

        ok = self.status == GRID_OK
        with np.errstate(invalid='ignore', divide='ignore'):
            # zero flux in a band (fully obscured star) is floored, so log flux stays
            # finite; PCHIP is monotone, so it never interpolates above the neighbours
            phot = np.where(ok[..., None], np.maximum(d['phot'], self.FLUX_FLOOR), np.nan)
            logf = np.log10(phot)
            logr1 = np.where(ok & (d['r1'] > 0), np.log10(d['r1']), np.nan)
        values = np.concatenate([logf, logr1[..., None]], axis=-1)

        # axes with a single value are fixed: drop them from the interpolation
        coords = [self.coord(k, self.axes[k]) for k in self.AXES]
        self._free = [i for i, c in enumerate(coords) if len(c) > 1]
        self._fixed = {i: c[0] for i, c in enumerate(coords) if len(c) == 1}
        self._coords = [coords[i] for i in self._free]
        self._lo = np.array([c[0] for c in self._coords])
        self._hi = np.array([c[-1] for c in self._coords])
        self._values = values.reshape([len(c) for c in self._coords] + [values.shape[-1]])
        # RegularGridInterpolator multiplies zero-weight corners too (0 * nan = nan),
        # which would make points on the faces of failed cells nan. Interpolate
        # nan-filled values together with the weight of the nan corners, and only
        # reject points where that weight is non-zero.
        bad = ~np.isfinite(self._values)
        kw = dict(method='linear', bounds_error=False, fill_value=np.nan)
        self._interp = interpolate.RegularGridInterpolator(self._coords, np.where(bad, 0.0, self._values), **kw)
        self._interp_bad = interpolate.RegularGridInterpolator(self._coords, bad.astype(float), **kw)

    @classmethod
    def coord(cls, name, x):
        return np.log10(x) if name in cls.LOG_AXES else np.asarray(x, dtype=float)

    def interp(self, temp, dust_temp, tau, s, Y):
        '''
        log10 flux (uJy at self.dist, L = 1e4 Lsun) per band, shape (..., nband),
        and log10 r1 (cm), broadcast over the inputs. NaN outside the grid.
        '''
        x = np.broadcast_arrays(*[self.coord(k, v) for k, v in zip(self.AXES, (temp, dust_temp, tau, s, Y))])
        shape = x[0].shape
        pts = np.stack([x[i].ravel() for i in self._free], axis=-1)
        # nan outside the axes (fill_value) or with weight on a failed model
        vals = self._interp(pts)
        vals[~(self._interp_bad(pts) < 1e-9)] = np.nan
        if self.method == 'pchip':
            inside = np.all((pts >= self._lo) & (pts <= self._hi), axis=1)
            if inside.any():
                vp = local_pchip(self._coords, self._values, pts[inside])
                # failed models in the stencil but not the cell: keep the linear value
                vals[inside] = np.where(np.isfinite(vp), vp, vals[inside])
        for i, c in self._fixed.items():
            vals[~np.isclose(x[i].ravel(), c)] = np.nan
        vals = vals.reshape(shape + (vals.shape[-1],))
        return vals[..., :-1], vals[..., -1]

    def in_logl_range(self, logl):
        return (np.asarray(logl) >= self.LOGL_RANGE[0]) & (np.asarray(logl) <= self.LOGL_RANGE[1])

    def phot(self, logl, temp, dust_temp, tau, s, Y, dist=None):
        '''
        Synthetic photometry (uJy), shape (..., nband), at luminosity 10**logl and dist
        (default the grid distance). NaN outside the grid, including log L outside
        LOGL_RANGE.
        '''
        logf, _ = self.interp(temp, dust_temp, tau, s, Y)
        logl = np.where(self.in_logl_range(logl), logl, np.nan)
        scale = 10**(logl - self.meta['logl'])
        if dist is not None:
            scale = scale * (self.dist / dist).decompose().value**2
        return 10**logf * np.asarray(scale)[..., None]

    def r1(self, temp, dust_temp, tau, s, Y):
        '''Inner radius (cm) for L = 1e4 Lsun'''
        return 10**self.interp(temp, dust_temp, tau, s, Y)[1]

    def r_in(self, logl, temp, dust_temp, tau, s, Y):
        '''Inner radius (cm) at luminosity 10**logl: r1 sqrt(L / 1e4 Lsun)'''
        logl = np.where(self.in_logl_range(logl), logl, np.nan)
        return self.r1(temp, dust_temp, tau, s, Y) * np.sqrt(10**(logl - self.meta['logl']))

def _validate_worker(task):
    builder, p, tables = task
    try:
        spec, res = builder.run_dusty_single(dict(p))
    except (DustyError, ValueError) as e:
        return None, str(e)[:300]
    bandpasses = bandpasses_from_tables(tables)
    phot = builder.model_phot(None, bandpasses, sed=builder.spec_to_sed(spec, p['logl']))
    return ([phot[b] for b in bandpasses], res.get('r1(cm)', np.nan)), ''

def validate_grid(builder, grid, bandpasses, n=300, points=None, seed=None, nproc=1):
    '''
    Compare the grid interpolation with direct DUSTY runs at off-grid points.

    points is an optional (N, 5) array of (temp, dust_temp, tau, s, Y), e.g. drawn from
    a posterior; otherwise n points are drawn uniformly in the interpolation
    coordinates (log tau, log Y) within the axes, keeping only points with a finite
    interpolation. builder must have the grid's configuration.

    Returns a QTable with the parameters, dmag per band (grid minus DUSTY, in mag)
    and the r1 ratio. Interpolation errors below ~0.02-0.03 mag are negligible against
    the photometric errors; otherwise add them in quadrature in the likelihood.
    '''
    if list(bandpasses) != grid.bands:
        raise ValueError(f'bandpasses must match the grid bands {grid.bands}')
    if json.loads(json.dumps(builder.grid_meta(), default=float)) != grid.meta['meta']:
        print('Warning: the builder configuration differs from the grid metadata')

    rng = np.random.default_rng(seed)
    if points is None:
        lo = [grid.coord(k, grid.axes[k].min()) for k in grid.AXES]
        hi = [grid.coord(k, grid.axes[k].max()) for k in grid.AXES]
        pts = []
        while len(pts) < n:
            x = rng.uniform(lo, hi, size=(4*n, 5))
            x[:, 2], x[:, 4] = 10**x[:, 2], 10**x[:, 4]
            keep = np.isfinite(grid.interp(*x.T)[1])
            pts.extend(x[keep])
        points = np.array(pts[:n])
    points = np.atleast_2d(points)

    tables = bandpass_tables(bandpasses)
    tasks = [(builder, {'logl': grid.meta['logl'], 'temp': t, 'dust_temp': td, 'tau': tau,
                        'density_power': s, 'shell_thickness': Y}, tables) for t, td, tau, s, Y in points]
    if nproc > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(nproc) as pool:
            results = pool.map(_validate_worker, tasks)
    else:
        results = [_validate_worker(t) for t in tasks]

    logf_grid, logr1_grid = grid.interp(*points.T)
    tb = QTable()
    for k, col in zip(grid.AXES, points.T):
        tb[k] = col
    dmag = np.full((len(points), len(grid.bands)), np.nan)
    r1_ratio = np.full(len(points), np.nan)
    msgs = []
    for i, (res, msg) in enumerate(results):
        msgs.append(msg)
        if res is None:
            continue
        f_dusty, r1 = res
        dmag[i] = -2.5 * (logf_grid[i] - np.log10(f_dusty))
        r1_ratio[i] = 10**logr1_grid[i] / r1
    for j, b in enumerate(grid.bands):
        tb[f'dmag_{b}'] = dmag[:, j]
    tb['r1_ratio'] = r1_ratio
    tb['msg'] = msgs
    return tb


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    rewrite_lambda_grid = args.rewrite_lambda_grid

    if args.gen:
        logz = 0.0 if args.logz is None else args.logz
        comp = args.comp
        outdir = args.outdir or 'data/dusty_sil_grid'
        modeldir = args.modeldir or 'data/marcs/m1.00_g0.00'

        dustgen = dusty_gen(dist = 10*u.Mpc, logz=logz, outdir=outdir, modeldir=modeldir, rewrite_lambda_grid=rewrite_lambda_grid)
        dustgen.dust_comp = comp

        dustgen.gen_grid(redo_dusty=False, redo_input=False, tb_overwrite=True)
    else:
        from rsg_phot.rsg import rsg_phot
        q, amin, amax = args.size
        builder = dusty_likelihood(dist=args.dist*u.Mpc, modeldir=args.modeldir or 'data/marcs/comarcs', logg=0.0,
                                   logz=-0.5 if args.logz is None else args.logz, mass=args.mass, c_o=args.c_o,
                                   density='powerlaw', comp={'sic': args.sic}, nk_comp={args.amc_nk: 1 - args.sic},
                                   size_dist='mrn', size_par={'q': q, 'amin': amin, 'amax': amax},
                                   outdir=args.outdir or 'data/dusty_grid', workdir=args.workdir)
        # grid bands: every filter of the photometry table (pandeia names MIRI filters F0560W etc.)
        tb = ascii.read(args.phot, format='csv', comment=r'\s*#', header_start=0, data_start=1)
        rp = rsg_phot(verbose=False)
        bandpasses = {}
        for f, inst in zip(tb['filter'], tb['instrument']):
            f, inst = str(f), str(inst).lower()
            bandpasses[f] = rp.get_jwst_filters(f'{inst},{f if inst == "nircam" else f"F{int(f[1:-1]):04d}W"}')
        axes = {k: SHELL_GRID_AXES[k] if v is None else v
                for k, v in zip(dusty_likelihood.GRID_AXES, [args.teff, args.tin, args.tau, args.s, args.Y])}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        print(' | '.join(f'{k}: {list(np.round(v, 5))}' for k, v in axes.items()), flush=True)
        builder.build_grid(args.out, bandpasses, temps=axes['temp'], dust_temps=axes['dust_temp'], taus=axes['tau'],
                           ss=axes['s'], Ys=axes['Y'], nproc=args.nproc, retry_failed=args.retry_failed,
                           save_sed=args.save_sed)