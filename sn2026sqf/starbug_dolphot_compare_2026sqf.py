'''
Field-star check of the starbug2 background subtraction for SN 2026sqf MIRI, against dolphot.

dolphot (data/dolphot/sn2026sqf/miri_0_0.phot, reference coadd_0_0_f560w_i2d.fits) fits a local sky
per star on the same JHAT stage-2 frames. For every frame, this script rebuilds the starbug background map
that starbug_miri_2026sqf.py selected for the band (injection-selected box and BGD_R), subtracts it, and
measures the dolphot sources within R_CMP pix of the target. The background mask list is the starbug
detections plus every dolphot source with band S/N > 3, so the comparison stars are masked like the target
is (an unlisted faint star is partly absorbed by a small-box map):
  - starbug forced PSF at the dolphot position (every dolphot source with band S/N > 3 is in the fit list,
    so neighbours are modelled);
  - EE50 aperture on the background-subtracted frame (CRDS annulus as residual sky) and on the cal frame
    (plain annulus), for the comparison stars.
Per-dither values are combined with inverse-variance weights.

Comparison stars: dolphot type 1, band S/N > SNR_CMP, flag 0, crowding < CROWD_MAX, and no other listed
dolphot source brighter than ISO_FRAC x the star within ISO_FWHM x FWHM. If either code
leaves diffuse emission in or takes source flux out, starbug/dolphot trends with the local diffuse level
relative to the star, x = B_loc * pi FWHM^2 / F_dolphot, where B_loc is the starbug map at the star
minus its median over the region. The target's x is marked so the regime it sits in can be read off.

Outputs, under data/dolphot/sn2026sqf/starbug/:
  sn2026sqf_miri_dolphot_compare_stars.csv    per star and band
  sn2026sqf_miri_dolphot_compare_summary.csv  per band: median ratio, scatter, trend with x
  sn2026sqf_miri_dolphot_compare.png

Run from the repo root after starbug_miri_2026sqf.py:  python starbug_dolphot_compare_2026sqf.py [--plots-only]
'''
import os, sys, glob, shutil
from multiprocessing import Pool

import numpy as np

import starbug_miri_2026sqf as S

PHOT = f'{S.ROOT}/data/dolphot/sn2026sqf/miri_0_0.phot'
REF_IMAGE = f'{S.ROOT}/data/dolphot/sn2026sqf/coadd_0_0_f560w_i2d.fits'
R_CMP = 400          # pix (44") around the target
SNR_FIT, SNR_CMP, CROWD_MAX = 3, 10, 0.5
ISO_FWHM, ISO_FRAC = 2.5, 0.25
N_BASE, N_PER_FILTER = 12, 13        # dolphot .phot layout: 12 global columns, 13 per filter
STARS_CSV = f'{S.OUT}/sn2026sqf_miri_dolphot_compare_stars.csv'


def load_dolphot():
    '''Global columns plus per-filter mag, err, S/N, flag, crowding; RA/Dec from the reference WCS (0-based = X - 0.5).'''
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    ncol = N_BASE + N_PER_FILTER * len(S.FILTERS)
    d = np.loadtxt(PHOT, usecols=range(ncol))
    out = pd.DataFrame(dict(id=np.arange(len(d)), X=d[:, 2], Y=d[:, 3], type=d[:, 10].astype(int)))
    for k, f in enumerate(S.FILTERS):
        c = N_BASE + N_PER_FILTER * k
        out[f'{f}_mag'], out[f'{f}_err'], out[f'{f}_snr'] = d[:, c + 4], d[:, c + 6], d[:, c + 8]
        out[f'{f}_crowd'], out[f'{f}_flag'] = d[:, c + 11], d[:, c + 12].astype(int)
        out[f'{f}_ujy'] = np.where(d[:, c + 4] < 90, 3631e6 * 10 ** (-0.4 * d[:, c + 4]), np.nan)
    out['ra'], out['dec'] = WCS(fits.getheader(REF_IMAGE, 'SCI')).all_pix2world(out['X'] - 0.5, out['Y'] - 0.5, 0)
    return out


def selected_maps():
    import pandas as pd
    s = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_injection_summary.csv')
    s = s[s['primary']]
    return {r['filter']: (int(r['box']), int(r['bgdr'])) for _, r in s.iterrows()}


def target_position():
    '''The F560W position used by starbug_miri_2026sqf.py, from the per-frame table.'''
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    fr = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_frames.csv').iloc[0]
    w = WCS(fits.getheader(f'{S.JHAT_DIR}/{fr["file"]}.fits', 'SCI'))
    ra, dec = w.all_pix2world(fr['x_t'], fr['y_t'], 0)
    return float(ra), float(dec)


def run_frame(args):
    fname, dol, box, bf, ra_t, dec_t = args
    from astropy.io import fits
    from astropy.table import Table
    filt = fits.getheader(fname)['FILTER']
    fw = S.fwhm_pix(filt)
    bname = os.path.basename(fname).replace('.fits', '')
    psffrac = fits.getheader(f'{S.PSF_DIR}/{filt}.fits')['PSFFRAC']
    ap_rows = S.aperture_setup(filt)
    tmp = f'{S.OUT}/compare_tmp'
    os.makedirs(tmp, exist_ok=True)
    base_opts = dict(VERBOSE=0, FWHM=fw, SIGSRC=5.0, PSF_SIZE=S.fit_size(filt), CALC_CROWD=0, APCORR_FILE=S.CRDS_APCORR)
    det, xt, yt, w, _ = S.prepare(fname, ra_t, dec_t, dict(base_opts, OUTPUT=tmp + '/'))
    with fits.open(fname) as h:
        scale = S.jy_scale(h)
        err_jy = h['ERR'].data * scale
        ny, nx = h['SCI'].data.shape
    x, y = w.all_world2pix(dol['ra'].values, dol['dec'].values, 0)
    d = dol.assign(x=x, y=y)
    d = d[(d['x'] > 0) & (d['x'] < nx) & (d['y'] > 0) & (d['y'] < ny) & (d[f'{filt}_snr'] > SNR_FIT)]
    # mask every listed dolphot source too (skipping ones starbug already detected)
    sep = np.hypot(d['x'].values[:, None] - det['xcentroid'][None, :], d['y'].values[:, None] - det['ycentroid'][None, :]).min(axis=1)
    new = d[sep > fw]
    det = S.add_sources(det, new['x'].values, new['y'].values, w, 'is_inj')
    opts = dict(base_opts, BOX_SIZE=box, BGD_R=bf * fw, OUTPUT=tmp + '/', QUIETMODE=1)
    sb, raw = S.subtract_background(fname, det, opts)
    bgd = raw - sb.image.data
    dd = np.hypot(d['x'].values[:, None] - d['x'].values[None, :], d['y'].values[:, None] - d['y'].values[None, :])
    np.fill_diagonal(dd, np.inf)
    fl = d[f'{filt}_ujy'].fillna(0).values
    bright_nb = (dd < ISO_FWHM * fw) & (fl[None, :] > ISO_FRAC * fl[:, None])
    d = d.assign(isolated=~bright_nb.any(axis=1))
    d = d[(np.hypot(d['x'] - xt, d['y'] - yt) < R_CMP) & (d['x'] > 380) & (d['x'] < nx - 30) & (d['y'] > 30) & (d['y'] < ny - 30)]
    cat = S.psf_fit(sb, Table(dict(xcentroid=d['x'].values, ycentroid=d['y'].values)), 'forced')
    rows = []
    reg = np.s_[max(int(yt) - R_CMP, 0):int(yt) + R_CMP, max(int(xt) - R_CMP, 0):int(xt) + R_CMP]
    bgd_med = np.nanmedian(bgd[reg])
    cmp = ((d['type'] == 1) & (d[f'{filt}_snr'] > SNR_CMP) & (d[f'{filt}_flag'] == 0) & (d[f'{filt}_crowd'] < CROWD_MAX)
           & d['isolated'])
    for _, s in d[cmp].iterrows():
        v = S.psf_vals(cat, s['x'], s['y'], psffrac, 'sb')
        r = dict(id=int(s['id']), file=bname, filter=filt, x=s['x'], y=s['y'], r_target_pix=float(np.hypot(s['x'] - xt, s['y'] - yt)),
                 sb_ujy=v['sb_ujy'], sb_err_ujy=v['sb_err_ujy'],
                 bgd_loc_mjysr=float(bgd[int(round(s['y'])), int(round(s['x']))] - bgd_med))
        a = S.aperture_target(sb.image.data * scale, err_jy, s['x'], s['y'], ap_rows, 'ap', extra=False)
        b = S.aperture_target(raw * scale, err_jy, s['x'], s['y'], ap_rows, 'ap_rawann', extra=False)
        t = S.tag_ee(S.EE_PRIMARY)
        r.update(ap_ujy=a[f'ap_{t}_ujy'], ap_err_ujy=a[f'ap_{t}_err_ujy'], ap_rawann_ujy=b[f'ap_rawann_{t}_ujy'],
                 ap_rawann_err_ujy=b[f'ap_rawann_{t}_err_ujy'])
        rows.append(r)
    # the target, for its x value and its own starbug/dolphot ratio
    bt = float(bgd[int(round(yt)), int(round(xt))] - bgd_med)
    rows.append(dict(id=-1, file=bname, filter=filt, x=xt, y=yt, r_target_pix=0.0, bgd_loc_mjysr=bt))
    for f in glob.glob(f'{tmp}/{bname}-*'):
        os.remove(f)
    return rows


def combine(stars, dol, ra_t, dec_t):
    '''Inverse-variance mean over dithers; adds dolphot fluxes and the contamination ratio x.'''
    import pandas as pd
    from starbug2.utils import get_MJysr2Jy_scalefactor
    from astropy.io import fits
    rows = []
    # dolphot counterpart of the target: nearest source to the F560W position
    sep = np.hypot((dol['ra'] - ra_t) * np.cos(np.radians(dec_t)), dol['dec'] - dec_t) * 3.6e6
    tid = int(dol['id'][np.argmin(sep)])
    for (filt, sid), g in stars.groupby(['filter', 'id']):
        row = dict(filter=filt, id=sid, n=len(g), r_target_pix=g['r_target_pix'].mean(), bgd_loc_mjysr=g['bgd_loc_mjysr'].mean())
        for m in ('sb', 'ap', 'ap_rawann'):
            if f'{m}_ujy' in g and g[f'{m}_ujy'].notna().any():
                row[f'{m}_ujy'], row[f'{m}_err_ujy'] = S.wmean(g[f'{m}_ujy'], g[f'{m}_err_ujy'])
        di = tid if sid == -1 else sid
        dr = dol[dol['id'] == di].iloc[0]
        row.update(dol_id=di, dol_ujy=dr[f'{filt}_ujy'], dol_snr=dr[f'{filt}_snr'])
        rows.append(row)
    out = pd.DataFrame(rows)
    # B_loc in uJy per pix x pi FWHM^2, relative to the dolphot flux
    pix_ujy = {}
    for filt in S.FILTERS:
        f = sorted(glob.glob(f'{S.JHAT_DIR}/*.fits'))
        f = [x for x in f if fits.getheader(x)['FILTER'] == filt][0]
        with fits.open(f) as h:
            pix_ujy[filt] = 1e6 * get_MJysr2Jy_scalefactor(h['SCI'])
    out['x_contam'] = [b * pix_ujy[f] * np.pi * S.fwhm_pix(f) ** 2 / fd for b, f, fd in zip(out['bgd_loc_mjysr'], out['filter'], out['dol_ujy'])]
    for m in ('sb', 'ap', 'ap_rawann'):
        out[f'{m}_ratio'] = out[f'{m}_ujy'] / out['dol_ujy']
    return out


def summarise(comb):
    import pandas as pd
    rows = []
    for filt in S.FILTERS:
        g = comb[(comb['filter'] == filt) & (comb['id'] >= 0)]
        t = comb[(comb['filter'] == filt) & (comb['id'] == -1)]
        row = dict(filter=filt, n_stars=len(g), target_x_contam=float(t['x_contam'].iloc[0]) if len(t) else np.nan)
        for m in ('sb', 'ap', 'ap_rawann'):
            r = g[f'{m}_ratio']
            med = np.nanmedian(r)
            row[f'{m}_ratio'] = med
            row[f'{m}_scatter'] = 1.4826 * np.nanmedian(np.abs(r - med))
            # trend: median ratio below / above the median contamination, and for stars within a factor 2 of the target's
            hi = g['x_contam'] > np.nanmedian(g['x_contam'])
            row[f'{m}_ratio_lowx'] = np.nanmedian(r[~hi])
            row[f'{m}_ratio_highx'] = np.nanmedian(r[hi])
            tx = row['target_x_contam']
            like = (g['x_contam'] > tx / 2) & (g['x_contam'] < 2 * tx) if tx > 0 else np.zeros(len(g), bool)
            row[f'{m}_ratio_targetlike'] = np.nanmedian(r[like]) if like.any() else np.nan
            row[f'{m}_n_targetlike'] = int(like.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(comb, summ, fname):
    import matplotlib.pyplot as plt
    C = S.METHOD_COLOR
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9), facecolor=S.SURF)
    for ax, filt in zip(axes.flat, S.FILTERS):
        g = comb[(comb['filter'] == filt) & (comb['id'] >= 0)]
        t = comb[(comb['filter'] == filt) & (comb['id'] == -1)]
        ax.axhline(1, color=S.INK2, lw=0.6)
        ax.axvline(0, color=S.INK2, lw=0.4)
        for m, col, lab in (('sb', C['psf_forced'], 'starbug forced PSF'), ('ap', C['ap'], 'aperture, bgd-subtracted'),
                            ('ap_rawann', C['ap_rawann'], 'aperture, plain annulus')):
            ax.scatter(g['x_contam'], g[f'{m}_ratio'], s=8, color=col, alpha=0.45, lw=0, label=lab if filt == S.FILTERS[0] else None)
            # running median in x bins
            if len(g) > 15:
                xs = np.sort(g['x_contam'].values)
                edges = np.unique(np.quantile(xs, np.linspace(0, 1, 7)))
                xc = 0.5 * (edges[1:] + edges[:-1])
                med = [np.nanmedian(g[f'{m}_ratio'][(g['x_contam'] >= a) & (g['x_contam'] < b)]) for a, b in zip(edges[:-1], edges[1:])]
                ax.plot(xc, med, '-', color=col, lw=2.2)
        if len(t):
            ax.axvline(t['x_contam'].iloc[0], color=S.SERIES[1], lw=1.5, ls='--', label='target' if filt == S.FILTERS[0] else None)
        ax.set_xscale('symlog', linthresh=0.05)
        ax.set_xlim(-1.5, 12)
        ax.set_xticks([-1, -0.1, 0, 0.1, 1, 10], ['−1', '−0.1', '0', '0.1', '1', '10'])
        ax.set_xticks([], minor=True)
        ax.set_ylim(0, 2.5)
        ax.tick_params(labelsize=7, colors=S.INK2)
        sm = summ[summ['filter'] == filt].iloc[0]
        ax.set_title(f'{filt}: {int(sm["n_stars"])} stars, median {sm["sb_ratio"]:.2f} (σ {sm["sb_scatter"]:.2f})', fontsize=8.5, color=S.INK)
    for ax in axes[-1]:
        ax.set_xlabel('local diffuse level × πFWHM² / F$_{dolphot}$', fontsize=8, color=S.INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel('starbug / dolphot', fontsize=8, color=S.INK2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, loc='lower center', ncol=3)
    fig.suptitle(f'Isolated field stars within {R_CMP} pix ({R_CMP * 0.11:.0f}″) of the target: dolphot type 1, S/N > {SNR_CMP}, flag 0, '
                 f'crowding < {CROWD_MAX},\nno neighbour > {ISO_FRAC:g}× within {ISO_FWHM} FWHM; dolphot sources masked; line: running median; '
                 'dashed: target', fontsize=9, color=S.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(fname, dpi=140, facecolor=S.SURF)
    plt.close(fig)


def main():
    import pandas as pd
    from astropy.io import fits
    dol = load_dolphot()
    ra_t, dec_t = target_position()
    if '--plots-only' not in sys.argv:
        sel = selected_maps()
        files = sorted(glob.glob(f'{S.JHAT_DIR}/*_mirimage_jhat.fits'))
        args = []
        for f in files:
            filt = fits.getheader(f)['FILTER']
            cols = ['id', 'ra', 'dec', 'type', f'{filt}_snr', f'{filt}_flag', f'{filt}_crowd', f'{filt}_ujy']
            args.append((f, dol[cols], *sel[filt], ra_t, dec_t))
        with Pool(S.NCORES) as p:
            out = p.map(run_frame, args)
        pd.DataFrame([r for rows in out for r in rows]).to_csv(STARS_CSV, index=False)
        shutil.rmtree(f'{S.OUT}/compare_tmp', ignore_errors=True)
    stars = pd.read_csv(STARS_CSV)
    comb = combine(stars, dol, ra_t, dec_t)
    comb.to_csv(STARS_CSV.replace('_stars.csv', '_stars_combined.csv'), index=False, float_format='%.4f')
    summ = summarise(comb)
    summ.to_csv(f'{S.OUT}/sn2026sqf_miri_dolphot_compare_summary.csv', index=False, float_format='%.4f')
    plot(comb, summ, f'{S.OUT}/sn2026sqf_miri_dolphot_compare.png')
    pd.set_option('display.width', 250)
    print(summ.round(3).to_string())
    print(comb[comb['id'] == -1][['filter', 'dol_ujy', 'x_contam', 'bgd_loc_mjysr']].round(3).to_string())


if __name__ == '__main__':
    main()
