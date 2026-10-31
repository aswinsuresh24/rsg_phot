'''
Wind + detached shell DUSTY grids (SN 2026sqf).

RADMC-3D finds that a spherical r^-2 wind inside a detached r^-2 shell fits the SN 2026sqf SED as well as the
wedge + shell (2026sqf_tmp/, Oct 2026), and DUSTY reproduces those RADMC models with a tabulated density
(2026sqf_tmp/windshell_dusty_vs_radmc.py). This module builds and interpolates grids of that family, reusing
the DUSTY machinery of rsg_phot.dust (dusty_likelihood, sharded builds, PCHIP interpolation):

    twoshell_likelihood   dusty_likelihood with the wind + shell density (DUSTY density type 5)
    build_grid            sharded, resumable grid over TWOSHELL_GRID_AXES (one DUSTY tau sequence per cell)
    TwoShellGrid          DustyGrid over those axes, for rsg_phot.mcmc_nest.nest
    validate_grid         grid interpolation against direct DUSTY runs

Geometry, in y = r / R_in (R_in where the dust is at T_in): a wind rho ~ y^-s_w from 1 to y_shell / gap, a gap
(gap = 1: none), and a shell rho ~ y^-s_s from y_shell to y_shell Y_shell. f_wind is the wind's share of the
radial tau_V, so each DUSTY tau sequence scales both components together. Teff is fixed per grid.

Run from the repo root, like rsg_phot.dust (cells are sharded in <out>.shards/, so rerunning resumes):
    python -m rsg_phot.dust_twoshell --out data/dusty_grids/2026sqf/twoshell_amC_SiC20.npz --nproc 20 --save_sed
'''
import os, json, time, argparse
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import QTable
from rsg_phot.dust import (dusty_likelihood, DustyGrid, local_pchip, merge_grids, table_bandpasses, bandpass_tables,
                           bandpasses_from_tables, report_progress, _validate_worker, GRID_OK, GRID_FAILED, GRID_MISSING)
from rsg_phot.dust_mixes import MIXES, builder_kwargs

# DUSTY aborts when eta_min/eta_max < 1e-12 anywhere in the shell; keep a margin
ETA_FLOOR = 1e-11

# Default grid (Teff 4000 K, amC + 20% SiC, MRN q 3.5, 0.005-0.25 um). Around the RADMC best fit (T_in 700 K,
# tau 8, f_wind 0.19, shell at 13-52 R_in, no gap; chi2 rises to 41 at T_in 1000 K, to 24-36 at f_wind 0.13 / 0.24,
# and the gap and Y_shell are weakly constrained). dust_temp in x1.19 steps (<= 0.017 mag in the single-shell
# hold-out), tau in 0.1 dex steps (0.15 dex gave <= 0.011 mag); the geometry axes are coarse: check them with a
# hold-out (validate_grid) before trusting the posterior
TWOSHELL_GRID_AXES = {'dust_temp': np.round(np.geomspace(500., 1000., 5)),
                      'tau': np.round(np.logspace(np.log10(2.5), np.log10(25.), 11), 4),
                      'f_wind': np.round(np.geomspace(0.06, 0.4, 6), 3),
                      'y_shell': np.round(np.geomspace(5., 32., 5), 2),
                      'gap': np.array([1., 1.6, 2.5]),
                      'Y_shell': np.array([2., 4., 8.])}

def windshell_density_profile(f_w, y_s, g, Y_s, s_w=2., s_s=2., edge=0.01, gap_rel=1e-2, npts=500, nedge=60):
    '''
    Tabulated (y, eta) of the wind + detached shell for DUSTY's user-supplied density (input density type 5),
    and the outer radius y_s Y_s. DUSTY normalises eta itself; the relative amplitudes give the wind tau
    fraction f_w (the gap fill adds to neither).

    DUSTY's radial grid (Ygrid) only allows a factor EtaRat = 4 between neighbouring points and at most npY
    points, and hangs on true steps. So both edges are logistic in ln y of width `edge`, and the gap is filled
    at gap_rel x the wind's density at y_w = y_s / g (< 0.5% of the wind's tau for g <= 2.5); the profile is
    continuous in g down to g = 1, where the edges coincide. DUSTY interpolates the table linearly, so it is
    log-spaced (npts) plus nedge points across each edge.
    '''
    if not 0 < f_w < 1:
        raise ValueError('f_w = tau_wind / tau must be in (0, 1)')
    if g < 1 or Y_s <= 1 or y_s/g <= 1:
        raise ValueError('need g >= 1, Y_s > 1 and a wind outer edge y_s/g > 1')
    if npts + 2*nedge > 1000:
        raise ValueError('DUSTY reads at most 1000 density points')
    y_w, y_o = y_s/g, y_s*Y_s
    ledge = lambda c: c*np.exp(np.linspace(-8*edge, 8*edge, nedge))
    y = np.unique(np.concatenate([np.geomspace(1., y_o, npts), ledge(y_w), ledge(y_s)]))
    y = y[(y >= 1) & (y <= y_o)]
    down = 1/(1 + np.exp((np.log(y) - np.log(y_w))/edge))
    up = 1/(1 + np.exp(-(np.log(y) - np.log(y_s))/edge))
    wind, shell = y**-s_w*down, y**-s_s*up
    a_w, a_s = f_w/np.trapezoid(wind, y), (1 - f_w)/np.trapezoid(shell, y)
    eta = a_w*wind + a_s*shell
    eta = np.maximum(eta, gap_rel*a_w*y_w**-s_w*(y <= y_s))
    eta = np.maximum(eta/eta.max(), ETA_FLOOR)
    return y, eta, y_o

class twoshell_likelihood(dusty_likelihood):
    '''
    dusty_likelihood for the wind + detached shell (see windshell_density_profile). Parameters (theta), in
    order of param_names, minus fixed ones: logl, temp, dust_temp, tau and the geometry f_wind, y_shell, gap,
    Y_shell. powers = (s_w, s_s) are the density powers of wind and shell. Other keywords as dusty_likelihood
    (dust, photosphere, workdir, ...).
    '''
    GEOMETRY = ['f_wind', 'y_shell', 'gap', 'Y_shell']
    GRID_AXES = tuple(TWOSHELL_GRID_AXES)

    def __init__(self, powers=(2., 2.), edge=0.01, gap_rel=1e-2, fixed=None, **kwargs):
        super().__init__(density='powerlaw', shell_thickness=None, **kwargs)
        self.density = 'windshell'
        self.powers = tuple(float(s) for s in powers)
        self.edge, self.gap_rel = edge, gap_rel
        self.param_names = self.PARAM_NAMES[:-1] + self.GEOMETRY
        self.fixed = {}
        for k, v in (fixed or {}).items():
            if k not in self.param_names:
                raise ValueError(f'Unknown parameter {k}, must be one of {self.param_names}')
            self.fixed[k] = v.value if isinstance(v, u.Quantity) else v
        self.free_params = [p for p in self.param_names if p not in self.fixed]

    def setup_dusty_single(self, inp_file, p):
        # the density table goes next to the input file, in the DUSTY working directory
        y, eta, _ = windshell_density_profile(p['f_wind'], p['y_shell'], p['gap'], p['Y_shell'], *self.powers,
                                              edge=self.edge, gap_rel=self.gap_rel)
        p['eta_file'] = os.path.join(os.path.dirname(inp_file), 'model.eta')
        # FileMSG in dustyV2.f reads file names into a CHARACTER*230
        if len(p['eta_file']) >= 230:
            raise ValueError(f'Path to {p["eta_file"]} is too long for DUSTY (>= 230 characters)')
        with open(p['eta_file'], 'w') as f:
            f.write(f'# wind rho ~ y^-{self.powers[0]:g} to y = {p["y_shell"]/p["gap"]:.6g} (tau fraction {p["f_wind"]}), '
                    f'shell rho ~ y^-{self.powers[1]:g} from {p["y_shell"]} to {p["y_shell"]*p["Y_shell"]:.6g}\n'
                    '#   y   eta (relative)\n#\n')
            np.savetxt(f, np.c_[y, eta], fmt='%.8e')
        super().setup_dusty_single(inp_file, p)

    def dusty_density_block(self, p):
        # input density type 5 reads a (y, eta) table from file; DUSTY takes Y from its last row
        return f'     3) Density Distribution\n        - density type = 5\n            {p["eta_file"]}\n\n'

    def config_meta(self):
        meta = super().config_meta()
        meta.pop('density_power', None)
        meta.update(density='windshell', powers=self.powers, edge=self.edge, gap_rel=self.gap_rel, eta_floor=ETA_FLOOR,
                    density_law=f'wind rho ~ r^-{self.powers[0]:g} from R_in to y_shell/gap R_in, gap (filled at gap_rel x '
                                f'the wind edge density), shell rho ~ r^-{self.powers[1]:g} from y_shell to y_shell*Y_shell R_in; '
                                f'logistic edges of width {self.edge} in ln r; f_wind = wind share of radial tau_V')
        return meta

    def grid_meta(self):
        return self.config_meta()

    def grid_cell(self, temp, dust_temp, f_wind, y_shell, gap, Y_shell, taus, bandpasses, save_sed=False):
        # one DUSTY tau sequence at fixed (temp, dust_temp, geometry); see dusty_likelihood.grid_cell
        p = {'logl': self.GRID_LOGL, 'temp': temp, 'dust_temp': dust_temp, 'f_wind': f_wind, 'y_shell': y_shell,
             'gap': gap, 'Y_shell': Y_shell}
        return self.sequence_cell(p, taus, bandpasses, save_sed)

    def build_grid(self, outfile, bandpasses, temp=4000., axes=None, nproc=1, save_sed=False, assemble=True,
                   retry_failed=False):
        '''
        Build a rectangular grid over GRID_AXES at Teff = temp, as dusty_likelihood.build_grid: every cell of the
        non-tau axes is one DUSTY run over all taus at L = 1e4 Lsun, written to its own shard in <outfile>.shards/
        (so an interrupted run resumes) and merged into outfile by assemble_grid. Axes missing from `axes` take
        the TWOSHELL_GRID_AXES values.
        '''
        axes = {k: np.unique(np.asarray((axes or {}).get(k, TWOSHELL_GRID_AXES[k]), dtype=float)) for k in self.GRID_AXES}
        if len(axes['tau']) < 2:
            raise ValueError('A DUSTY tau sequence needs at least two optical depths')
        if (axes['f_wind'] <= 0).any() or (axes['f_wind'] >= 1).any() or (axes['gap'] < 1).any() or (axes['Y_shell'] <= 1).any():
            raise ValueError('need 0 < f_wind < 1, gap >= 1 and Y_shell > 1')
        if axes['y_shell'].min() <= axes['gap'].max():
            raise ValueError('every y_shell must exceed every gap (the wind must reach beyond R_in)')
        lo, hi = self.rsg_temp.value.min(), self.rsg_temp.value.max()
        if not lo <= temp <= hi:
            raise ValueError(f'temp must lie within the {self.modeltype} grid ({lo}-{hi} K)')

        shard_dir = Path(str(outfile) + '.shards')
        shard_dir.mkdir(parents=True, exist_ok=True)
        config = {'axes': {k: v.tolist() for k, v in axes.items()}, 'temp': float(temp),
                  'bands': list(bandpasses), 'save_sed': save_sed, 'dist_mpc': self.dist.to(u.Mpc).value,
                  'logl': self.GRID_LOGL, 'meta': self.grid_meta(),
                  'dusty_path': self.dusty_basedir, 'npL': len(self.read_lambda_grid())}
        config = json.loads(json.dumps(config, default=float))
        config_file = shard_dir / 'config.json'
        if config_file.exists():
            if json.loads(config_file.read_text()) != config:
                raise ValueError(f'{shard_dir} holds shards of a different grid; use another outfile or remove the directory')
        else:
            config_file.write_text(json.dumps(config, indent=1))

        cells = [tuple(c) for c in np.ndindex(*[len(axes[k]) for k in CELL_AXES])]
        def done(c):
            fl_ = shard_dir / shard_name(c)
            return fl_.exists() and not (retry_failed and str(np.load(fl_)['msg']))
        todo = [c for c in cells if not done(c)]
        print(f'{len(cells)} cells ({len(cells)*len(axes["tau"])} models); {len(todo)} still to run')

        # bandpasses go to the workers as plain arrays (see dusty_likelihood.build_grid)
        args = (self, bandpass_tables(bandpasses), axes, float(temp), save_sed, str(shard_dir))
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

# the axes that index cells (one DUSTY run each); tau is the sequence inside a cell
CELL_AXES = tuple(k for k in TWOSHELL_GRID_AXES if k != 'tau')

def shard_name(cell):
    return 'cell_' + '_'.join(f'{i:03d}' for i in cell) + '.npz'

# state of a grid worker process, set once by _grid_worker_init
_GRID_WORKER = {}

def _grid_worker_init(builder, tables, axes, temp, save_sed, shard_dir):
    _GRID_WORKER.update(builder=builder, axes=axes, temp=temp, save_sed=save_sed, shard_dir=Path(shard_dir),
                        bandpasses=bandpasses_from_tables(tables))

def _grid_worker(cell):
    w = _GRID_WORKER
    vals = {k: w['axes'][k][i] for k, i in zip(CELL_AXES, cell)}
    out = w['builder'].grid_cell(w['temp'], taus=w['axes']['tau'], bandpasses=w['bandpasses'], save_sed=w['save_sed'], **vals)
    # write atomically, so a killed run never leaves a truncated shard
    tmp = w['shard_dir'] / ('tmp_' + shard_name(cell))
    np.savez(tmp, **out)
    os.replace(tmp, w['shard_dir'] / shard_name(cell))
    if out['msg']:
        return f'cell {cell} failed: {out["msg"].splitlines()[0][:150]}'
    nbad = int((~out['ok']).sum())
    return f'cell {cell}: {nbad} of {len(out["ok"])} models failed' if nbad else ''

def assemble_grid(outfile):
    '''
    Merge the shards in <outfile>.shards/ into one .npz grid file, as rsg_phot.dust.assemble_grid: the axes,
    'phot' (uJy at dist_mpc, L = 1e4 Lsun; shape dust_temp x tau x f_wind x y_shell x gap x Y_shell x band),
    'r1' (cm, for 1e4 Lsun), 'err' (%), 'td_y' (K), 'warn', 'status' (GRID_OK / GRID_FAILED / GRID_MISSING),
    optionally 'sed' / 'sed_lambda', and the metadata as a JSON string.
    '''
    shard_dir = Path(str(outfile) + '.shards')
    config = json.loads((shard_dir / 'config.json').read_text())
    axes = {k: np.asarray(v) for k, v in config['axes'].items()}
    shape = tuple(len(axes[k]) for k in TwoShellGrid.AXES)
    itau = TwoShellGrid.AXES.index('tau')
    phot = np.full(shape + (len(config['bands']),), np.nan)
    arrs = {k: np.full(shape, np.nan) for k in ['r1', 'err', 'td_y']}
    warn = np.zeros(shape, bool)
    status = np.full(shape, GRID_MISSING, dtype=np.int8)
    runtime, seds = 0.0, None
    for cell in np.ndindex(*[len(axes[k]) for k in CELL_AXES]):
        fl_ = shard_dir / shard_name(cell)
        if not fl_.exists():
            continue
        idx = tuple(cell[:itau]) + (slice(None),) + tuple(cell[itau:])
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
    meta.update({'units': {'phot': 'uJy at dist_mpc for L = 1e4 Lsun', 'r1': 'cm for L = 1e4 Lsun', 'err': 'percent', 'td_y': 'K'},
                 'n_ok': int((status == GRID_OK).sum()), 'n_failed': int((status == GRID_FAILED).sum()),
                 'n_missing': int((status == GRID_MISSING).sum()), 'cpu_hours': runtime / 3600})
    out = dict(phot=phot, warn=warn, status=status, meta=json.dumps(meta), **arrs, **{k: axes[k] for k in TwoShellGrid.AXES})
    if seds is not None:
        out.update(sed=seds, sed_lambda=sed_lambda)
    np.savez(outfile, **out)
    print(f'{outfile}: {meta["n_ok"]} models ok, {meta["n_failed"]} failed, {meta["n_missing"]} missing; '
          f'{meta["cpu_hours"]:.1f} CPU hours')
    return outfile

class TwoShellGrid(DustyGrid):
    '''
    DustyGrid (tensor-product PCHIP of log flux and log r1, analytic in log L) over the wind + shell axes
    (dust_temp, tau, f_wind, y_shell, gap, Y_shell) at the grid's fixed Teff (meta['temp']). The methods take
    the axes in that order, after logl where there is one.
    '''
    AXES = twoshell_likelihood.GRID_AXES
    LOG_AXES = ('tau', 'f_wind', 'y_shell', 'gap', 'Y_shell')

    def interp(self, *params):
        '''log10 flux (uJy at self.dist, L = 1e4 Lsun) per band, shape (..., nband), and log10 r1 (cm); NaN outside'''
        x = np.broadcast_arrays(*[self.coord(k, v) for k, v in zip(self.AXES, params)])
        shape = x[0].shape
        pts = np.stack([x[i].ravel() for i in self._free], axis=-1)
        vals = self._interp(pts)
        vals[~(self._interp_bad(pts) < 1e-9)] = np.nan
        if self.method == 'pchip':
            inside = np.all((pts >= self._lo) & (pts <= self._hi), axis=1)
            if inside.any():
                vp = local_pchip(self._coords, self._values, pts[inside])
                vals[inside] = np.where(np.isfinite(vp), vp, vals[inside])
        for i, c in self._fixed.items():
            vals[~np.isclose(x[i].ravel(), c)] = np.nan
        vals = vals.reshape(shape + (vals.shape[-1],))
        return vals[..., :-1], vals[..., -1]

    def phot(self, logl, *params, dist=None):
        '''Synthetic photometry (uJy), shape (..., nband), at 10**logl Lsun and dist (default the grid's)'''
        logf, _ = self.interp(*params)
        logl = np.where(self.in_logl_range(logl), logl, np.nan)
        scale = 10**(logl - self.meta['logl'])
        if dist is not None:
            scale = scale * (self.dist / dist).decompose().value**2
        return 10**logf * np.asarray(scale)[..., None]

    def r1(self, *params):
        '''Inner radius (cm) for L = 1e4 Lsun'''
        return 10**self.interp(*params)[1]

    def r_in(self, logl, *params):
        '''Inner radius (cm) at luminosity 10**logl: r1 sqrt(L / 1e4 Lsun)'''
        logl = np.where(self.in_logl_range(logl), logl, np.nan)
        return self.r1(*params) * np.sqrt(10**(logl - self.meta['logl']))

def load_grid(gridfile, **kwargs):
    # DustyGrid or TwoShellGrid, by the density law in the grid metadata
    meta = json.loads(str(np.load(gridfile)['meta']))
    return (TwoShellGrid if meta['meta'].get('density') == 'windshell' else DustyGrid)(gridfile, **kwargs)

def validate_grid(builder, grid, bandpasses, points, nproc=1):
    '''
    Grid interpolation against direct DUSTY runs at points (N, 6) of (dust_temp, tau, f_wind, y_shell, gap,
    Y_shell), as rsg_phot.dust.validate_grid. Returns a QTable with the parameters, dmag per band (grid minus
    DUSTY) and the r1 ratio. builder must have the grid's configuration.
    '''
    if list(bandpasses) != grid.bands:
        raise ValueError(f'bandpasses must match the grid bands {grid.bands}')
    if json.loads(json.dumps(builder.grid_meta(), default=float)) != grid.meta['meta']:
        print('Warning: the builder configuration differs from the grid metadata')
    points = np.atleast_2d(points)
    tables = bandpass_tables(bandpasses)
    tasks = [(builder, dict(logl=grid.meta['logl'], temp=grid.meta['temp'], **dict(zip(grid.AXES, x))), tables) for x in points]
    if nproc > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(nproc) as pool:
            results = pool.map(_validate_worker, tasks)
    else:
        results = [_validate_worker(t) for t in tasks]

    logf_grid, logr1_grid = grid.interp(*points.T)
    tb = QTable({k: col for k, col in zip(grid.AXES, points.T)})
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

def create_parser():
    parser = argparse.ArgumentParser(description='Build the wind + detached shell DUSTY grid (twoshell_likelihood.build_grid) '
                                     'for SN 2026sqf. Paths are relative to the repo root.')
    parser.add_argument('--mix', type=str, default='carbon', choices=list(MIXES),
                        help='Photosphere + dust preset from dust_mixes.py (carbon: COMARCS + amC/SiC; silicate: MARCS + Sil-Ow/corundum)')
    parser.add_argument('--out', type=str, default=None,
                        help='Grid file (default data/dusty_grid/twoshell_grid_<mix tag>_ism.npz); cells are sharded in '
                             '<out>.shards/, so rerunning resumes an interrupted build')
    parser.add_argument('--phot', type=str, default='data/dolphot/sn2026sqf/sn2026sqf_photometry_new_ujy.txt',
                        help='Photometry table whose filter and instrument columns set the grid bands')
    parser.add_argument('--dist', type=float, default=19.0, help='Distance (Mpc)')
    parser.add_argument('--teff', type=float, default=None, help='Photospheric Teff (K), fixed for the grid (default the --mix value: '
                        'carbon 4000, silicate 4500)')
    parser.add_argument('--logz', type=float, default=None, help='Photosphere log metallicity (default the --mix value)')
    parser.add_argument('--c_o', type=float, default=None, help='COMARCS C/O (default the --mix value)')
    parser.add_argument('--mass', type=float, default=None, help='COMARCS mass (Msun; default the --mix value)')
    parser.add_argument('--modeldir', type=str, default=None, help='Model directory for DUSTY input spectra (default the --mix photosphere)')
    parser.add_argument('--outdir', type=str, default='data/dusty_grid', help='Output directory for input spectra')
    parser.add_argument('--size', type=float, nargs=3, default=None, metavar=('Q', 'AMIN', 'A'),
                        help='Size distribution of the --mix type (KMH for both presets): power q, amin and amax (MRN) or the '
                             'exponential scale a0 (KMH), in um; default the --mix value')
    parser.add_argument('--powers', type=float, nargs=2, default=[2., 2.], metavar=('S_WIND', 'S_SHELL'),
                        help='Density powers of the wind and the shell (rho ~ r^-s)')
    for k, name in [('dust_temp', 'tin'), ('tau', 'tau'), ('f_wind', 'f_wind'), ('y_shell', 'y_shell'), ('gap', 'gap'),
                    ('Y_shell', 'Y_shell')]:
        parser.add_argument(f'--{name}', type=float, nargs='+', default=None, help=f'{k} axis (default {TWOSHELL_GRID_AXES[k].tolist()})')
    parser.add_argument('--nproc', type=int, default=10, help='Parallel DUSTY processes')
    parser.add_argument('--retry_failed', default=False, action='store_true', help='Rerun cells whose DUSTY run failed as a whole')
    parser.add_argument('--merge', type=str, nargs='+', default=None, metavar='NPZ',
                        help='Instead of building, merge these assembled grids (differing only in their --merge_axis nodes) into --out')
    parser.add_argument('--merge_axis', type=str, default='dust_temp', choices=list(TWOSHELL_GRID_AXES), help='Axis to merge along')
    parser.add_argument('--save_sed', default=False, action='store_true',
                        help='Also store the full DUSTY spectrum of every model (fnu in uJy at --dist for 1e4 Lsun)')
    parser.add_argument('--workdir', type=str, default='data/dusty_grid/work_twoshell',
                        help='DUSTY working directories (keep the path short: DUSTY has fixed-length file names)')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    args.out = args.out or f'data/dusty_grid/twoshell_grid_{MIXES[args.mix]["tag"]}_ism.npz'
    args.teff = MIXES[args.mix]['teff'] if args.teff is None else args.teff
    if args.merge:
        merge_grids(args.merge, args.out, axis=args.merge_axis, grid_axes=TwoShellGrid.AXES)
    else:
        a3 = 'a0' if MIXES[args.mix]['size_dist'] == 'kmh' else 'amax'
        size = None if args.size is None else dict(zip(('q', 'amin', a3), args.size))
        builder = twoshell_likelihood(dist=args.dist*u.Mpc, powers=args.powers, outdir=args.outdir, workdir=args.workdir,
                                      **builder_kwargs(args.mix, modeldir=args.modeldir, logz=args.logz, mass=args.mass,
                                                       c_o=args.c_o, size_par=size))
        axes = {k: TWOSHELL_GRID_AXES[k] if v is None else np.asarray(v)
                for k, v in zip(twoshell_likelihood.GRID_AXES, [args.tin, args.tau, args.f_wind, args.y_shell, args.gap, args.Y_shell])}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        print(f'mix {args.mix}: {builder.grid_meta()}', flush=True)
        print(f'Teff {args.teff:g} | ' + ' | '.join(f'{k}: {np.round(v, 5).tolist()}' for k, v in axes.items()), flush=True)
        builder.build_grid(args.out, table_bandpasses(args.phot), temp=args.teff, axes=axes, nproc=args.nproc,
                           retry_failed=args.retry_failed, save_sed=args.save_sed)
