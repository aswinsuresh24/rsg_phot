'''
SN 2026sqf F770W: the target sits on a narrow PAH filament. Injection test along the filament and a filament template in
the dither-stack PSF fit (f1800w_recheck_2026sqf.py machinery).

At any position (target or candidate site) the four F770W cal dithers are stacked on it (HALF pix box), then:
  local plane   fitted to the pixels at R_PLANE pix (3-sigma clipped), subtracted -> sub
  filament      the angle through the centre that maximises median(sub) along the line at S_SIDE pix minus the same at
                +-PERP pix sideways
  template      the filament cross-section P(d) (d: distance across the ridge), median over the S_SIDE pixels on both
                sides of the centre, extended along the ridge through the centre (the source itself is excluded)
Models within R_FIT, position fixed:
  psf_plane, psf_quad   F PSF + plane / quadratic (as before)
  psf_tmpl_fix          (image - template) = F PSF + plane         (filament at its measured strength under the source)
  psf_tmpl_free         image = F PSF + alpha template + plane      (filament strength free)
  ap                    EE50 aperture, stpsf apcorr, CRDS annulus (plain)
Sites: candidates along the target's filament line at S_CAND pix (and their own local filament), kept when their
filament peak is AMP_RANGE x the target's and they are >= MIN_SEP pix apart. Empty sites give the filament-only
bias of each method; the same sites with an injected stpsf source of INJ_UJY give its recovery.

Outputs, under data/dolphot/sn2026sqf/starbug/f770w_recheck/: filament_results.csv, filament_sites.csv,
filament_summary.txt, filament.png
Run from the repo root:  python sn2026sqf/f770w_filament_2026sqf.py
'''
import numpy as np
import pandas as pd

import starbug_miri_2026sqf as S
import starbug_dolphot_compare_2026sqf as C
import starbug_pah_injection_2026sqf as P
import f1800w_recheck_2026sqf as R

FILT = 'F770W'
OUTD = f'{S.OUT}/f770w_recheck'
HALF = 22
R_PLANE = (9, 16)
S_SIDE = np.concatenate([np.arange(-12, -3.5, 0.5), np.arange(4, 12.5, 0.5)])
PERP = 7
R_FIT = [6, 8]
S_CAND = [s for s in range(-40, 41, 2) if abs(s) >= 8]
AMP_RANGE = (0.5, 2.0)
MIN_SEP = 7
INJ_UJY = 4.0


def grid():
    yy, xx = np.mgrid[:2 * HALF + 1, :2 * HALF + 1] - HALF
    return xx, yy


def sample(img, x, y):
    from scipy.ndimage import map_coordinates
    return map_coordinates(np.nan_to_num(img, nan=np.nanmedian(img)), [np.atleast_1d(y), np.atleast_1d(x)], order=1)


def local_plane(img):
    from astropy.stats import sigma_clip
    xx, yy = grid()
    rr = np.hypot(xx, yy)
    m = (rr >= R_PLANE[0]) & (rr <= R_PLANE[1]) & np.isfinite(img)
    v = sigma_clip(img[m], sigma=3).filled(np.nan)
    g = np.isfinite(v)
    A = np.vstack([np.ones(g.sum()), xx[m][g], yy[m][g]]).T
    c, *_ = np.linalg.lstsq(A, v[g], rcond=None)
    return img - (c[0] + c[1] * xx + c[2] * yy)


def filament(sub):
    '''Angle (deg), cross-section profile P(d) on d_grid, and its peak amplitude.'''
    best = None
    for th in np.arange(0, 180, 2.0):
        t = np.radians(th)
        u, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        on = np.median(sample(sub, HALF + S_SIDE * u[0], HALF + S_SIDE * u[1]))
        off = np.median(np.concatenate([sample(sub, HALF + S_SIDE * u[0] + k * PERP * n[0], HALF + S_SIDE * u[1] + k * PERP * n[1])
                                        for k in (-1, 1)]))
        if best is None or on - off > best[1]:
            best = (th, on - off)
    th = best[0]
    t = np.radians(th)
    u, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    d_grid = np.arange(-10, 10.01, 0.5)
    prof = np.array([np.median(sample(sub, HALF + S_SIDE * u[0] + d * n[0], HALF + S_SIDE * u[1] + d * n[1])) for d in d_grid])
    edge = np.median(np.concatenate([prof[:4], prof[-4:]]))
    prof = prof - edge
    return th, d_grid, prof, prof[np.abs(d_grid) <= 1.5].max()


def template(th, d_grid, prof):
    xx, yy = grid()
    t = np.radians(th)
    d = -xx * np.sin(t) + yy * np.cos(t)
    return np.interp(d, d_grid, prof, left=0, right=0)


def fit(img, err, psf, rfit, kind, tmpl):
    xx, yy = grid()
    m = (np.hypot(xx, yy) <= rfit) & np.isfinite(img) & np.isfinite(err)
    data = img - tmpl if kind == 'psf_tmpl_fix' else img
    cols = [psf[m]] + ([tmpl[m]] if kind == 'psf_tmpl_free' else []) + [np.ones(m.sum()), xx[m], yy[m]]
    if kind == 'psf_quad':
        cols += [xx[m] ** 2, yy[m] ** 2, (xx * yy)[m]]
    A = np.vstack(cols).T
    w = 1 / err[m]
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, data[m] * w, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    chi2 = np.sum(((data[m] - A @ coef) * w) ** 2)
    scale = np.sqrt(max(chi2 / (m.sum() - A.shape[1]), 1))
    return coef[0], np.sqrt(cov[0, 0]) * scale, (coef[1] if kind == 'psf_tmpl_free' else np.nan)


def measure(img, err, psf, **key):
    sub = local_plane(img)
    th, d_grid, prof, amp = filament(sub)
    tm = template(th, d_grid, prof)
    a = [r for r in S.aperture_setup(FILT) if r['ee'] == S.EE_PRIMARY][0]
    ap = S.aperture_target(img * 1e-6, err * 1e-6, HALF, HALF, [a], 'ap', extra=False)
    t50 = S.tag_ee(S.EE_PRIMARY)
    info = dict(key, angle=th, fil_amp=amp)
    rows = [dict(info, method='ap', rfit=np.nan, flux=ap[f'ap_{t50}_ujy'], err=ap[f'ap_{t50}_err_ujy'], alpha=np.nan)]
    for r in R_FIT:
        for k in ('psf_plane', 'psf_quad', 'psf_tmpl_fix', 'psf_tmpl_free'):
            f, e, al = fit(img, err, psf, r, k, tm)
            rows.append(dict(info, method=k, rfit=r, flux=f, err=e, alpha=al))
    return rows, (sub, th, d_grid, prof, tm)


def main():
    from astropy.io import fits
    R.HALF = HALF
    ra_t, dec_t = C.target_position()
    frames = R.load_frames(FILT)
    psf = R.psf_image(FILT)
    img, err = R.stack(frames, ra_t, dec_t)
    rows_t, geo = measure(img, err, psf, kind='target', s=0, f_in=np.nan)
    th_t, amp_t = rows_t[0]['angle'], rows_t[0]['fil_amp']
    # candidate sites along the target's filament
    w0 = frames[0]['wcs']
    x0, y0 = [float(v) for v in w0.all_world2pix(ra_t, dec_t, 0)]
    t = np.radians(th_t)
    cands = []
    for s in S_CAND:
        ra, dec = w0.all_pix2world(x0 + s * np.cos(t), y0 + s * np.sin(t), 0)
        im, er = R.stack(frames, float(ra), float(dec))
        _, d_grid, prof, amp = filament(local_plane(im))
        cands.append(dict(s=s, ra=float(ra), dec=float(dec), fil_amp=amp))
    cands = pd.DataFrame(cands)
    ok = cands[(cands['fil_amp'] >= AMP_RANGE[0] * amp_t) & (cands['fil_amp'] <= AMP_RANGE[1] * amp_t)]
    sites = []
    for _, c in ok.iterrows():
        if all(abs(c['s'] - s['s']) >= MIN_SEP for s in sites):
            sites.append(c)
    sites = pd.DataFrame(sites)
    sites.to_csv(f'{OUTD}/filament_sites.csv', index=False)
    adds = []
    for fr in frames:
        pts = list(zip(*fr['wcs'].all_world2pix(sites['ra'].values, sites['dec'].values, 0)))
        with fits.open(fr['file']) as h:
            adds.append(P.inject(fr['file'], FILT, pts, INJ_UJY) * S.jy_scale(h) * 1e6)
    rows = list(rows_t)
    for _, st in sites.iterrows():
        im, er = R.stack(frames, st['ra'], st['dec'])
        rows += measure(im, er, psf, kind='ridge_empty', s=st['s'], f_in=0.0)[0]
        im, er = R.stack(frames, st['ra'], st['dec'], adds)
        rows += measure(im, er, psf, kind='ridge_inj', s=st['s'], f_in=INJ_UJY)[0]
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUTD}/filament_results.csv', index=False)

    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    lines = [f'target filament: angle {th_t:.0f} deg, peak {amp_t:.3f} uJy/pix; {len(cands)} candidates along it, '
             f'{len(ok)} with peak {AMP_RANGE[0]}-{AMP_RANGE[1]}x the target\'s, {len(sites)} sites used '
             f'(s = {sites["s"].tolist()} pix; peaks {np.round(sites["fil_amp"].values, 3).tolist()})']
    for (meth, r), g in d.groupby(['method', 'rfit'], dropna=False):
        tg = g[g['kind'] == 'target'].iloc[0]
        e = g[g['kind'] == 'ridge_empty']['flux']
        i = g[g['kind'] == 'ridge_inj']
        rec = (i['flux'] / i['f_in']).values
        net = (i['flux'].values - e.values) / i['f_in'].values
        corr = (tg['flux'] - np.median(e)) / np.median(net)
        lines.append(f'{meth:13s} r={r!s:>4}: target {tg["flux"]:5.2f} ± {tg["err"]:4.2f}'
                     f' | empty filament {np.median(e):+5.2f} (σ {mad(e):4.2f})'
                     f' | injected: recovered/in {np.median(rec):4.2f} (σ {mad(rec):4.2f}), net {np.median(net):4.2f}'
                     f' | target corrected {corr:5.2f} ± {np.hypot(tg["err"], mad(e)) / np.median(net):4.2f}'
                     + (f' | alpha target {tg["alpha"]:.2f}' if meth == 'psf_tmpl_free' else ''))
    txt = '\n'.join(lines)
    open(f'{OUTD}/filament_summary.txt', 'w').write(txt + '\n')
    print(txt)
    plot(img, err, psf, geo, sites, d)


def plot(img, err, psf, geo, sites, d):
    import matplotlib.pyplot as plt
    sub, th, d_grid, prof, tm = geo
    xx, yy = grid()
    t = np.radians(th)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), facecolor=S.SURF)
    sig = 1.4826 * np.nanmedian(np.abs(sub - np.nanmedian(sub)))
    ax = axes[0]
    ax.imshow(sub, origin='lower', cmap='Greys_r', vmin=-3 * sig, vmax=10 * sig)
    ax.plot(HALF + np.array([-HALF, HALF]) * np.cos(t), HALF + np.array([-HALF, HALF]) * np.sin(t), '--', color=S.SERIES[0], lw=1)
    vis = sites[sites['s'].abs() < HALF]
    ax.scatter(HALF + vis['s'] * np.cos(t), HALF + vis['s'] * np.sin(t), s=40, facecolor='none', edgecolor=S.SERIES[0], lw=1.4)
    ax.plot(HALF, HALF, '+', color=S.SERIES[1], ms=12, mew=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'F770W stack − local plane; filament {th:.0f}°\ncircles: injection sites on the filament', fontsize=9, color=S.INK)
    ax = axes[1]
    ax.plot(d_grid, prof, '-o', color=S.SERIES[0], ms=3, lw=2, label='filament cross-section (4–12 pix either side)')
    m = np.abs(xx * np.sin(t) - yy * np.cos(t)) < 0.6
    dd = (-xx * np.sin(t) + yy * np.cos(t))
    across = np.abs(xx * np.cos(t) + yy * np.sin(t)) < 0.8
    ax.plot(dd[across].ravel(), sub[across].ravel(), '.', color=S.SERIES[1], ms=4, label='across the target')
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_xlabel('distance across the filament (pix)', fontsize=8, color=S.INK2)
    ax.set_ylabel('µJy / pix', fontsize=8, color=S.INK2)
    ax.tick_params(labelsize=7.5, colors=S.INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('filament profile vs the target cut', fontsize=9, color=S.INK)
    ax = axes[2]
    tg = d[(d['kind'] == 'target') & (d['method'] == 'psf_tmpl_fix') & (d['rfit'] == 8)].iloc[0]
    rmask = np.hypot(xx, yy) <= 8
    A = np.vstack([psf[rmask], np.ones(rmask.sum()), xx[rmask], yy[rmask]]).T
    coef, *_ = np.linalg.lstsq(A, (img - tm)[rmask], rcond=None)
    res = np.full(img.shape, np.nan)
    res[rmask] = (img - tm)[rmask] - A @ coef
    ax.imshow(res, origin='lower', cmap='Greys_r', vmin=-4 * sig, vmax=4 * sig)
    ax.set_xlim(HALF - 10, HALF + 10); ax.set_ylim(HALF - 10, HALF + 10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'target residual: − filament template − plane − PSF\n(F = {tg["flux"]:.2f} µJy, r ≤ 8)', fontsize=9, color=S.INK)
    ax = axes[3]
    meths = [('ap', np.nan), ('psf_plane', 8), ('psf_quad', 8), ('psf_tmpl_fix', 8), ('psf_tmpl_free', 8),
             ('psf_plane', 6), ('psf_quad', 6), ('psf_tmpl_fix', 6), ('psf_tmpl_free', 6)]
    for k, (mt, r) in enumerate(meths):
        g = d[(d['method'] == mt) & (d['rfit'].isna() if np.isnan(r) else (d['rfit'] == r))]
        tgt = g[g['kind'] == 'target'].iloc[0]
        e = g[g['kind'] == 'ridge_empty']['flux']
        i = g[g['kind'] == 'ridge_inj']
        net = np.median((i['flux'].values - e.values) / i['f_in'].values)
        sig_e = 1.4826 * np.median(np.abs(e - np.median(e)))
        ax.errorbar([k], [(tgt['flux'] - np.median(e)) / net], yerr=[np.hypot(tgt['err'], sig_e) / net], fmt='D', color=S.SERIES[0],
                    ms=6, lw=1.3, label='corrected for filament bias and recovery' if k == 0 else None)
        ax.plot([k], [tgt['flux']], 'o', color=S.INK2, mfc=S.SURF, ms=6, label='raw' if k == 0 else None)
    ax.axhline(4.52, color=S.SERIES[1], lw=1.5, label='adopted (dolphot) 4.52')
    ax.set_xticks(range(len(meths)), [f'{m.replace("psf_", "")}\n{"" if np.isnan(r) else f"r={r}"}' for m, r in meths], fontsize=6.5)
    ax.set_ylabel('target F770W (µJy)', fontsize=8.5, color=S.INK2)
    ax.tick_params(labelsize=7.5, colors=S.INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('target flux, calibrated on the filament', fontsize=9, color=S.INK)
    fig.tight_layout()
    fig.savefig(f'{OUTD}/filament.png', dpi=140, facecolor=S.SURF)


if __name__ == '__main__':
    main()
