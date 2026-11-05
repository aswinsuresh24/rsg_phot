'''
Fold the starbug2 background map into the dither-stack PSF fit of f1800w_recheck_2026sqf.py and check whether the
long-band SN 2026sqf fluxes agree with the map-free fits.

For each band and each starbug map (MAPS: the map tuned in starbug_pah_injection_2026sqf.py plus corners of its
sweep), the map is rebuilt for every dither (mask list = starbug detections + target [+ dolphot sources]), converted
to uJy/pix and stacked with the same shifts and weights as the cal image. Fits within R_FIT, position fixed:
  nomap_plane / nomap_quad   image = F PSF + plane / quadratic          (reference: the map-free fits)
  sub_const / sub_plane      image - map = F PSF + constant / plane      (map subtracted, residual sky free)
  tmpl_const / tmpl_plane    image = F PSF + alpha map + constant / plane (map as a template with free scale alpha)
Calibration as before, at the PAH-complex sites: empty (map from the cal frames) and with an injected stpsf source
(map rebuilt on the injected frames, injection positions in the mask list, as starbug would see them).

Outputs, under data/dolphot/sn2026sqf/starbug/f1800w_recheck/: map_fit_results.csv, map_fit_summary.csv,
map_fit.png
Run from the repo root after f1800w_recheck_2026sqf.py:  python sn2026sqf/f1800w_map_fit_2026sqf.py [--plots-only]
'''
import os, glob, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import starbug_miri_2026sqf as S
import starbug_dolphot_compare_2026sqf as C
import starbug_pah_injection_2026sqf as P
import f1800w_recheck_2026sqf as R

BANDS = ['F1280W', 'F1500W', 'F1800W', 'F2100W']
MAPS = [('sbdol', 5, 2), ('sbdol', 25, 2), ('sbdol', 5, 5), ('sbdol', 25, 5), ('sb', 25, 2)]   # + the tuned map
R_FIT = [6, 8, 11]
VARIANTS = {   # name: (subtract map, use map as template, background terms)
    'nomap_plane': (False, False, 'plane'), 'nomap_quad': (False, False, 'quad'),
    'sub_const': (True, False, 'const'), 'sub_plane': (True, False, 'plane'),
    'tmpl_const': (False, True, 'const'), 'tmpl_plane': (False, True, 'plane'),
}
OUTD = R.OUTD


def fit(img, err, psf, bmap, rfit, variant):
    '''Weighted linear least squares; returns F, sigma_F, alpha (template scale, NaN otherwise).'''
    sub, tmpl, bg = VARIANTS[variant]
    yy, xx = np.mgrid[:2 * R.HALF + 1, :2 * R.HALF + 1] - R.HALF
    data = img - bmap if sub else img
    m = (np.hypot(xx, yy) <= rfit) & np.isfinite(data) & np.isfinite(err) & np.isfinite(bmap)
    cols = [psf[m]] + ([bmap[m]] if tmpl else []) + [np.ones(m.sum())]
    if bg in ('plane', 'quad'):
        cols += [xx[m], yy[m]]
    if bg == 'quad':
        cols += [xx[m] ** 2, yy[m] ** 2, (xx * yy)[m]]
    A = np.vstack(cols).T
    w = 1 / err[m]
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, data[m] * w, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    chi = (data[m] - A @ coef) * w
    scale = np.sqrt(max(np.sum(chi ** 2) / (m.sum() - A.shape[1]), 1))
    return coef[0], np.sqrt(cov[0, 0]) * scale, (coef[1] if tmpl else np.nan)


def stack_map(frames, maps, ra, dec):
    '''Stack per-dither maps with the image's shifts and weights (pixels masked in the image masked here too).'''
    fake = [dict(fr, img=np.where(np.isfinite(fr['img']), mp, np.nan)) for fr, mp in zip(frames, maps)]
    return R.stack(fake, ra, dec)[0]


def band_run(filt):
    from astropy.io import fits
    dol = C.load_dolphot()
    ra_t, dec_t = C.target_position()
    tuned = pd.read_csv(f'{P.OUTP}/tuned_photometry.txt', comment='#').set_index('filter').loc[filt]
    maps = [(tuned['mask'], int(tuned['box']), int(tuned['bgdr']))] + [m for m in MAPS if m != (tuned['mask'], int(tuned['box']), int(tuned['bgdr']))]
    frames = R.load_frames(filt)
    psf = R.psf_image(filt)
    fw = S.fwhm_pix(filt)
    sites = pd.read_csv(f'{P.OUTP}/sites.csv')
    sites = sites[sites['filter'] == filt]
    f_inj = P.target_flux(filt)
    tmp = f'{OUTD}/tmp_{filt}'
    os.makedirs(tmp, exist_ok=True)
    setup = []
    for fr in frames:
        base, det, xt, yt, w, shape = P.frame_setup(fr['file'], filt, ra_t, dec_t)
        ks = P.known_sources(fr['file'], filt, dol, w, shape)
        sx, sy = w.all_world2pix(sites['ra'].values, sites['dec'].values, 0)
        add = P.inject(fr['file'], filt, list(zip(sx, sy)), f_inj)
        with fits.open(fr['file']) as h:
            sc = S.jy_scale(h) * 1e6
        setup.append(dict(base=base, det=det, w=w, ks=ks, sx=sx, sy=sy, add=add, sc=sc))
    add_ujy = [s['add'] * s['sc'] for s in setup]
    rows = []
    for k, (mask, box, bgdr) in enumerate(maps):
        m0, mi = [], []
        for fr, s in zip(frames, setup):
            mdet = P.mask_list(s['det'], s['ks'], fw, s['w'], mask)
            opts = dict(s['base'], BOX_SIZE=box, BGD_R=bgdr * fw, OUTPUT=tmp + '/', QUIETMODE=1)
            sb, raw = S.subtract_background(fr['file'], mdet, opts)
            m0.append((raw - sb.image.data) * s['sc'])
            mdet_i = S.add_sources(mdet, list(s['sx']), list(s['sy']), s['w'], 'is_inj')
            sbi, rawi = S.subtract_background(fr['file'], mdet_i, opts, s['add'])
            mi.append((rawi - sbi.image.data) * s['sc'])
        key = dict(filter=filt, mask=mask, box=box, bgdr=bgdr, tuned=k == 0)
        todo = [('target', -1, ra_t, dec_t, None, m0, np.nan)]
        todo += [('site_empty', j, ra, dec, None, m0, 0.0) for j, (ra, dec) in enumerate(zip(sites['ra'], sites['dec']))]
        todo += [('site_inj', j, ra, dec, add_ujy, mi, f_inj) for j, (ra, dec) in enumerate(zip(sites['ra'], sites['dec']))]
        for kind, pos, ra, dec, add, mp, f_in in todo:
            img, err = R.stack(frames, ra, dec, add)
            bmap = stack_map(frames, mp, ra, dec)
            for v in VARIANTS:
                for r in R_FIT:
                    f, e, a = fit(img, err, psf, bmap, r, v)
                    rows.append(dict(key, kind=kind, pos=pos, f_in=f_in, variant=v, rfit=r, flux=f, err=e, alpha=a))
    for f in glob.glob(f'{tmp}/*'):
        os.remove(f)
    os.rmdir(tmp)
    print(f'done {filt}', flush=True)
    return rows


def summarise(d):
    mad = lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
    out = []
    for key, g in d.groupby(['filter', 'mask', 'box', 'bgdr', 'tuned', 'variant', 'rfit']):
        t = g[g['kind'] == 'target'].iloc[0]
        e = g[g['kind'] == 'site_empty']
        i = g[g['kind'] == 'site_inj']
        rec = i['flux'] / i['f_in']
        out.append(dict(zip(['filter', 'mask', 'box', 'bgdr', 'tuned', 'variant', 'rfit'], key), target=t['flux'], target_err=t['err'],
                        alpha=t['alpha'], empty_med=np.median(e['flux']), empty_sig=mad(e['flux']), rec=np.median(rec), rec_sig=mad(rec),
                        alpha_empty=np.nanmedian(e['alpha'])))
    return pd.DataFrame(out)


def plot(s, fname):
    import matplotlib.pyplot as plt
    adopted = pd.read_csv(f'{S.ROOT}/data/dolphot/sn2026sqf/sn2026sqf_photometry_adopted_ujy.txt', comment='#').set_index('filter')
    variants = list(VARIANTS)
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.2 * len(BANDS), 4.6), facecolor=S.SURF, sharey=False)
    for ax, filt in zip(axes, BANDS):
        g = s[(s['filter'] == filt) & (s['rfit'] == 8)]
        maps = g[['mask', 'box', 'bgdr', 'tuned']].drop_duplicates().sort_values('tuned', ascending=False)
        for k, (_, mp) in enumerate(maps.iterrows()):
            h = g[(g['mask'] == mp['mask']) & (g['box'] == mp['box']) & (g['bgdr'] == mp['bgdr'])].set_index('variant').loc[variants]
            x = np.arange(len(variants)) + 0.1 * (k - (len(maps) - 1) / 2)
            e = np.hypot(h['target_err'], h['empty_sig'])
            col = S.INK if mp['tuned'] else S.BLUE_RAMP[2 + k]
            ax.errorbar(x, h['target'], yerr=e, fmt='o', color=col, ms=4 if not mp['tuned'] else 6, lw=1.2,
                        label=f"{'tuned: ' if mp['tuned'] else ''}{mp['mask']}, box {mp['box']}, {mp['bgdr']} FWHM")
        a = adopted.loc[filt]
        if a['detected']:
            ax.axhspan(a['flux_ujy'] - a['err_lo_ujy'], a['flux_ujy'] + a['err_hi_ujy'], color=S.SERIES[1], alpha=0.12, lw=0)
            ax.axhline(a['flux_ujy'], color=S.SERIES[1], lw=1.5, label='adopted (aperture)')
        ax.set_xticks(range(len(variants)), [v.replace('_', '\n') for v in variants], fontsize=7)
        ax.tick_params(labelsize=7, colors=S.INK2)
        ax.set_title(filt, fontsize=10, color=S.INK)
        ax.set_ylabel('target F$_ν$ (µJy)', fontsize=8, color=S.INK2)
        if filt == BANDS[0]:
            ax.legend(fontsize=6.5, frameon=False, loc='best')
    fig.suptitle('Dither-stack PSF fit (r = 8 pix) with the starbug map: none / subtracted / free-scale template; '
                 'errors = fit ⊕ empty-site scatter', fontsize=9.5, color=S.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fname, dpi=140, facecolor=S.SURF)


def main():
    if '--plots-only' not in sys.argv:
        with Pool(len(BANDS)) as p:
            out = p.map(band_run, BANDS)
        pd.DataFrame([r for rows in out for r in rows]).to_csv(f'{OUTD}/map_fit_results.csv', index=False)
    d = pd.read_csv(f'{OUTD}/map_fit_results.csv')
    s = summarise(d)
    s.to_csv(f'{OUTD}/map_fit_summary.csv', index=False, float_format='%.4f')
    plot(s, f'{OUTD}/map_fit.png')
    pd.set_option('display.width', 250)
    for filt in BANDS:
        g = s[(s['filter'] == filt) & s['rfit'].isin([8, 11])]
        print(f'== {filt}')
        print(g.pivot_table(index=['mask', 'box', 'bgdr', 'tuned'], columns=['variant', 'rfit'], values='target').round(1).to_string())
        print(g.groupby(['variant', 'rfit'])[['empty_sig', 'rec', 'rec_sig', 'alpha']].median().round(2).to_string())


if __name__ == '__main__':
    main()
