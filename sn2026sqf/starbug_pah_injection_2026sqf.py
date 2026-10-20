'''
Tune the starbug2 background map for SN 2026sqf MIRI with point sources injected in the target's own PAH complex.

Sites (per band, chosen once in sky coordinates from dither 1, then injected at the same sky positions in all
four dithers, so each site is combined over dithers exactly like the target):
  - diffuse level like the target's: the reference map (box REF_BOX, BGD_R REF_BGDR FWHM, starbug + dolphot mask)
    minus its median within R_SITE, within LEVEL_TOL of the target's value;
  - 4 FWHM to R_SITE from the target;
  - no dolphot source (band S/N > 3) or starbug detection brighter than NB_FRAC x the injected flux within
    NB_FWHM x FWHM. The complex is crowded (~1 dolphot source per 2 FWHM at 15 um within 16"), so fainter
    neighbours are allowed, as they are around the target; the summed neighbour flux within NB_FWHM is recorded
    for every site and the target;
  - mutually separated so apertures, sky annuli and PSF fits do not overlap.
Injected flux: the target's EE50 aperture flux from starbug_miri_2026sqf.py (dolphot / default if not positive).

Maps swept: BOX_SIZES x BGDR_FACTORS x MASKS, where MASKS is the source list masked in the map: starbug
detections + target ('sb'), or also every dolphot source with band S/N > 3 ('sbdol'). For every map:
  - injections: forced PSF and EE50 aperture (stpsf apcorr, CRDS annulus on the subtracted frame), combined per
    site over dithers -> recovered/injected bias, scatter, rms = sqrt(bias^2 + scatter^2);
  - residual ring: sum of the PSF-fit residual (forced) in 0.5-2 FWHM around each source / its flux (target and
    injections: the injected flux; field stars: their PSF flux), for the
    injections, the target, and low-background dolphot field stars (real PSFs, so PSF-model mismatch shows up
    there but not in stpsf injections).
The tuned map per band minimises rms(forced PSF) + rms(aperture). Its target photometry, with the site-to-site
scatter as the background error, is written to the tuned table.

Outputs, under data/dolphot/sn2026sqf/starbug/pah_injection/:
  sites.csv, frames.csv (per frame/map/source), summary.csv (per band/map), tuned_photometry.txt,
  pah_injection_maps.png, pah_injection_residuals.png, pah_injection_sites.png

Run from the repo root after starbug_miri_2026sqf.py and starbug_dolphot_compare_2026sqf.py:
  python starbug_pah_injection_2026sqf.py [--plots-only]
'''
import os, sys, glob, shutil, zlib
from multiprocessing import Pool

import numpy as np

import starbug_miri_2026sqf as S
import starbug_dolphot_compare_2026sqf as C

OUTP = f'{S.OUT}/pah_injection'
BOX_SIZES = [5, 10, 25, 40]
BGDR_FACTORS = [2, 3, 4, 5]
MASKS = ['sb', 'sbdol']
REF_BOX, REF_BGDR = 25, 3
R_SITE = 150           # pix (16")
LEVEL_TOL = 0.5        # |E - E_target| < LEVEL_TOL |E_target|
NB_FRAC, NB_FWHM = 0.5, 2.5
N_SITES, N_SITE_MIN = 10, 4
RING = (0.5, 2.0)      # residual ring, FWHM
X_LOW = 0.3            # low-background field stars for the residual ring


def target_flux(filt):
    import pandas as pd
    t = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_photometry.txt', comment='#')
    f = float(t.loc[t['filter'] == filt, 'ap_ujy'].iloc[0])
    return f if f > 0 else S.injection_flux(filt)


def known_sources(fname, filt, dol, w, shape):
    '''Pixel positions of dolphot sources for masking (band S/N > 3) and for site avoidance.'''
    x, y = w.all_world2pix(dol['ra'].values, dol['dec'].values, 0)
    inside = (x > -20) & (x < shape[1] + 20) & (y > -20) & (y < shape[0] + 20)
    return dict(x=x[inside], y=y[inside], band=dol[f'{filt}_snr'].values[inside] > 3,
                flux=np.nan_to_num(dol[f'{filt}_ujy'].values[inside]))


def mask_list(det, ks, fw, w, mode):
    if mode == 'sb':
        return det
    xs, ys = ks['x'][ks['band']], ks['y'][ks['band']]
    sep = np.hypot(xs[:, None] - det['xcentroid'][None, :], ys[:, None] - det['ycentroid'][None, :]).min(axis=1)
    keep = sep > fw
    return S.add_sources(det, xs[keep], ys[keep], w, 'is_inj')


def near_any(det, pts, radius):
    '''Rows of det within radius of any (x, y) in pts: the sources a PSF fit around those points needs.'''
    px, py = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    d = np.hypot(np.asarray(det['xcentroid'])[:, None] - px[None, :], np.asarray(det['ycentroid'])[:, None] - py[None, :])
    return det[d.min(axis=1) < radius]


def frame_setup(fname, filt, ra_t, dec_t):
    from astropy.io import fits
    fw = S.fwhm_pix(filt)
    base = dict(VERBOSE=0, FWHM=fw, SIGSRC=5.0, PSF_SIZE=S.fit_size(filt), CALC_CROWD=0, APCORR_FILE=S.CRDS_APCORR)
    det, xt, yt, w, _ = S.prepare(fname, ra_t, dec_t, base)
    with fits.open(fname) as h:
        shape = h['SCI'].data.shape
    return base, det, xt, yt, w, shape


def choose_sites(args):
    '''Sites in the target's complex, from dither 1 of a band (sky coordinates).'''
    fname, dol, ra_t, dec_t = args
    from astropy.io import fits
    filt = fits.getheader(fname)['FILTER']
    fw = S.fwhm_pix(filt)
    base, det, xt, yt, w, shape = frame_setup(fname, filt, ra_t, dec_t)
    ks = known_sources(fname, filt, dol, w, shape)
    tmp = f'{OUTP}/tmp'
    os.makedirs(tmp, exist_ok=True)
    opts = dict(base, BOX_SIZE=REF_BOX, BGD_R=REF_BGDR * fw, OUTPUT=tmp + '/', QUIETMODE=1)
    sb, raw = S.subtract_background(fname, mask_list(det, ks, fw, w, 'sbdol'), opts)
    bmap = raw - sb.image.data
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    reg = np.hypot(xx - xt, yy - yt) < R_SITE
    E = bmap - np.nanmedian(bmap[reg])
    Et = E[int(round(yt)), int(round(xt))]
    rout = S.aperture_setup(filt)[0]['rout']
    sep_min = max(5 * fw, rout + 2 * fw)
    rng = np.random.default_rng(zlib.crc32(filt.encode()))
    cand = np.argwhere(reg & (np.abs(E - Et) < LEVEL_TOL * abs(Et)) & (np.hypot(xx - xt, yy - yt) > 4 * fw)
                       & (xx > 400) & (xx < shape[1] - 40) & (yy > 40) & (yy < shape[0] - 40))
    rng.shuffle(cand)
    f_in = target_flux(filt)
    dflux = 1e6 * np.nan_to_num(np.asarray(det['flux'], float))
    real = ~np.asarray(det['is_target'])
    nbx = np.concatenate([ks['x'][ks['band']], np.asarray(det['xcentroid'])[real]])
    nby = np.concatenate([ks['y'][ks['band']], np.asarray(det['ycentroid'])[real]])
    nbf = np.concatenate([ks['flux'][ks['band']], dflux[real]])

    def neighbours(x, y, exclude=0.0):
        d = np.hypot(nbx - x, nby - y)
        m = (d < NB_FWHM * fw) & (d > exclude)
        return nbf[m].max(initial=0), nbf[m].sum()

    pts, nb = [], []
    for yc, xc in cand:
        mx, tot = neighbours(xc, yc)
        if mx > NB_FRAC * f_in:
            continue
        if pts and np.min([np.hypot(xc - a, yc - b) for a, b in pts]) < sep_min:
            continue
        if np.hypot(xc - xt, yc - yt) < sep_min:
            continue
        pts.append((float(xc), float(yc)))
        nb.append(tot)
        if len(pts) == N_SITES:
            break
    nb_t = neighbours(xt, yt, exclude=0.5 * fw)[1]      # the target itself is excluded
    ra, dec = w.all_pix2world([p[0] for p in pts], [p[1] for p in pts], 0)
    for f in glob.glob(f'{tmp}/{os.path.basename(fname).replace(".fits", "")}-*'):
        os.remove(f)
    return [dict(filter=filt, site=k, ra=float(a), dec=float(b), E=float(E[int(p[1]), int(p[0])]), E_target=float(Et),
                 r_target_pix=float(np.hypot(p[0] - xt, p[1] - yt)), nb_flux_ujy=float(n), nb_flux_target_ujy=float(nb_t),
                 f_in=f_in) for k, (a, b, p, n) in enumerate(zip(ra, dec, pts, nb))]


def inject(fname, filt, pts, f_ujy):
    '''MJy/sr image of stpsf point sources of total flux f_ujy at pixel positions pts.'''
    from astropy.io import fits
    from photutils.psf import FittableImageModel
    psffrac = fits.getheader(f'{S.PSF_DIR}/{filt}.fits')['PSFFRAC']
    model = FittableImageModel(fits.getdata(f'{S.PSF_DIR}/{filt}.fits'), normalize=True)
    with fits.open(fname) as h:
        shape, scale = h['SCI'].data.shape, S.jy_scale(h)
    add = np.zeros(shape, np.float32)
    half = S.PSF_FOV // 2
    for x, y in pts:
        model.flux, model.x_0, model.y_0 = f_ujy * 1e-6 * psffrac, x, y
        x0, y0 = int(round(x)), int(round(y))
        sl = np.s_[max(y0 - half, 0):min(y0 + half + 1, shape[0]), max(x0 - half, 0):min(x0 + half + 1, shape[1])]
        yy, xx = np.mgrid[sl]
        add[sl] += model(xx, yy) / scale[sl]
    return add


def ring_fraction(res_jy, x, y, fw, flux_ujy):
    from photutils.aperture import CircularAnnulus, aperture_photometry
    a = CircularAnnulus((x, y), RING[0] * fw, RING[1] * fw)
    return 1e6 * aperture_photometry(res_jy, a, method='exact')['aperture_sum'][0] / flux_ujy


def run_frame(args):
    fname, dol, sites, field, ra_t, dec_t = args
    from astropy.io import fits
    from starbug2.utils import get_MJysr2Jy_scalefactor
    filt = fits.getheader(fname)['FILTER']
    fw = S.fwhm_pix(filt)
    bname = os.path.basename(fname).replace('.fits', '')
    psffrac = fits.getheader(f'{S.PSF_DIR}/{filt}.fits')['PSFFRAC']
    ap_rows = S.aperture_setup(filt)
    t50 = S.tag_ee(S.EE_PRIMARY)
    f_in = target_flux(filt)
    base, det, xt, yt, w, shape = frame_setup(fname, filt, ra_t, dec_t)
    ks = known_sources(fname, filt, dol, w, shape)
    sx, sy = w.all_world2pix(sites['ra'].values, sites['dec'].values, 0)
    pts = list(zip(sx, sy))
    fx, fy = w.all_world2pix(field['ra'].values, field['dec'].values, 0)
    fin = (fx > 400) & (fx < shape[1] - 30) & (fy > 30) & (fy < shape[0] - 30)
    add = inject(fname, filt, pts, f_in)
    with fits.open(fname) as h:
        scale = S.jy_scale(h)
        err_jy = h['ERR'].data * scale
        sf = get_MJysr2Jy_scalefactor(h['SCI'])
    tmp = f'{OUTP}/tmp'
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for mode in MASKS:
        mdet = mask_list(det, ks, fw, w, mode)
        rfit = 1.5 * S.fit_size(filt)
        fit_t = near_any(mdet, [(xt, yt)] + list(zip(fx[fin], fy[fin])), rfit)
        mdet_i = S.add_sources(mdet, list(sx), list(sy), w, 'is_inj')
        fit_i = near_any(mdet_i, pts, rfit)
        for box in BOX_SIZES:
            for bf in BGDR_FACTORS:
                opts = dict(base, BOX_SIZE=box, BGD_R=bf * fw, OUTPUT=tmp + '/', QUIETMODE=1)
                key = dict(file=bname, filter=filt, mask=mode, box=box, bgdr=bf)
                # target and field stars (real sources)
                sb, raw = S.subtract_background(fname, mdet, opts)
                # field stars not already in the fit list (sbdol lists them; sb may have detected them)
                d0 = np.hypot(fx[:, None] - fit_t['xcentroid'][None, :], fy[:, None] - fit_t['ycentroid'][None, :]).min(axis=1)
                new = fin & (d0 > 1)
                fl = S.add_sources(fit_t, list(fx[new]), list(fy[new]), w, 'is_inj') if new.any() else fit_t
                resid = f'{tmp}/{bname}-res_t.fits'
                cat = S.psf_fit(sb, fl, 'forced', resid)
                res_jy = fits.getdata(resid) * sf
                v = S.psf_vals(cat, xt, yt, psffrac, 'p')
                a = S.aperture_target(sb.image.data * scale, err_jy, xt, yt, ap_rows, 'ap', extra=False)
                rows.append(dict(key, kind='target', site=-1, f_in=np.nan, psf_ujy=v['p_ujy'], psf_err_ujy=v['p_err_ujy'],
                                 ap_ujy=a[f'ap_{t50}_ujy'], ap_err_ujy=a[f'ap_{t50}_err_ujy'],
                                 ring=ring_fraction(res_jy, xt, yt, fw, f_in)))
                for k in np.where(fin)[0]:
                    v = S.psf_vals(cat, fx[k], fy[k], psffrac, 'p')
                    rows.append(dict(key, kind='field', site=int(field['id'].values[k]), f_in=np.nan, psf_ujy=v['p_ujy'],
                                     psf_err_ujy=v['p_err_ujy'], ap_ujy=np.nan, ap_err_ujy=np.nan,
                                     ring=ring_fraction(res_jy, fx[k], fy[k], fw, max(v['p_ujy'], 1e-3))))
                # injections at the complex sites
                sbi, _ = S.subtract_background(fname, mdet_i, opts, add)
                cat = S.psf_fit(sbi, fit_i, 'forced', resid)
                res_jy = fits.getdata(resid) * sf
                for k, (x, y) in enumerate(pts):
                    v = S.psf_vals(cat, x, y, psffrac, 'p')
                    a = S.aperture_target(sbi.image.data * scale, err_jy, x, y, ap_rows, 'ap', extra=False)
                    rows.append(dict(key, kind='inj', site=int(sites['site'].values[k]), f_in=f_in, psf_ujy=v['p_ujy'],
                                     psf_err_ujy=v['p_err_ujy'], ap_ujy=a[f'ap_{t50}_ujy'], ap_err_ujy=a[f'ap_{t50}_err_ujy'],
                                     ring=ring_fraction(res_jy, x, y, fw, f_in)))
    for f in glob.glob(f'{tmp}/{bname}-*'):
        os.remove(f)
    print(f'done {bname} ({filt}, {len(pts)} sites)', flush=True)
    return rows


def combine_sites(fr):
    '''Per band/map/source: inverse-variance mean over dithers.'''
    import pandas as pd
    out = []
    for key, g in fr.groupby(['filter', 'mask', 'box', 'bgdr', 'kind', 'site']):
        row = dict(zip(['filter', 'mask', 'box', 'bgdr', 'kind', 'site'], key), f_in=g['f_in'].iloc[0], ring=np.nanmean(g['ring']))
        row['psf_ujy'], row['psf_err_ujy'] = S.wmean(g['psf_ujy'], g['psf_err_ujy'])
        row['ap_ujy'], row['ap_err_ujy'] = S.wmean(g['ap_ujy'], g['ap_err_ujy']) if g['ap_ujy'].notna().any() else (np.nan, np.nan)
        out.append(row)
    return pd.DataFrame(out)


def summarise(cs):
    import pandas as pd
    rows = []
    for key, g in cs.groupby(['filter', 'mask', 'box', 'bgdr']):
        inj, tgt, fld = g[g['kind'] == 'inj'], g[g['kind'] == 'target'].iloc[0], g[g['kind'] == 'field']
        row = dict(zip(['filter', 'mask', 'box', 'bgdr'], key), n_sites=len(inj), f_in=inj['f_in'].iloc[0] if len(inj) else np.nan)
        for m in ('psf', 'ap'):
            r = inj[f'{m}_ujy'] / inj['f_in']
            med = np.nanmedian(r)
            row[f'{m}_bias'] = med - 1
            row[f'{m}_scatter'] = 1.4826 * np.nanmedian(np.abs(r - med))
            row[f'{m}_noise'] = np.nanmedian(inj[f'{m}_err_ujy'] / inj['f_in'])
            row[f'{m}_rms'] = np.hypot(row[f'{m}_bias'], row[f'{m}_scatter'])
            row[f'target_{m}_ujy'], row[f'target_{m}_err_ujy'] = tgt[f'{m}_ujy'], tgt[f'{m}_err_ujy']
        row['cost'] = row['psf_rms'] + row['ap_rms']
        row['inj_ring_med'] = np.nanmedian(inj['ring'])
        row['inj_ring_lo'], row['inj_ring_hi'] = np.nanpercentile(inj['ring'], [16, 84]) if len(inj) else (np.nan, np.nan)
        row['target_ring'] = tgt['ring']
        row['field_ring_med'] = np.nanmedian(fld['ring']) if len(fld) else np.nan
        row['n_field'] = len(fld)
        rows.append(row)
    s = pd.DataFrame(rows)
    s['tuned'] = False
    for filt, g in s.groupby('filter'):
        s.loc[g['cost'].idxmin(), 'tuned'] = True
    return s


def write_tuned(summ, fname):
    import pandas as pd
    import starbug2
    t = summ[summ['tuned']].set_index('filter').loc[S.FILTERS].reset_index()
    rows = []
    for _, r in t.iterrows():
        row = dict(filter=r['filter'], wave_um=starbug2.filters[r['filter']].wavelength, mask=r['mask'], box=r['box'], bgdr=r['bgdr'],
                   n_sites=r['n_sites'], f_injected_ujy=r['f_in'])
        for m in ('psf', 'ap'):
            f, e = r[f'target_{m}_ujy'], r[f'target_{m}_err_ujy']
            bgd = np.sqrt(max(r[f'{m}_scatter'] ** 2 - r[f'{m}_noise'] ** 2, 0)) * r['f_in']
            tot = np.hypot(e, bgd)
            key = 'psf_forced' if m == 'psf' else 'ap'
            row.update({f'{key}_ujy': f, f'{key}_err_ujy': e, f'{key}_bgd_ujy': bgd, f'{key}_tot_err_ujy': tot,
                        f'{key}_inj_bias': r[f'{m}_bias'], f'{key}_det_snr': f / e,
                        f'{key}_ul_ujy': max(f, 0) + 3 * tot if f / e < 3 else np.nan})
        row.update(target_ring=r['target_ring'], inj_ring_med=r['inj_ring_med'], inj_ring_hi=r['inj_ring_hi'], field_ring_med=r['field_ring_med'])
        rows.append(row)
    out = pd.DataFrame(rows)
    with open(fname, 'w') as fp:
        fp.write('# SN 2026sqf MIRI target photometry with the starbug map tuned on injections in the target PAH complex '
                 '(starbug_pah_injection_2026sqf.py)\n')
        fp.write('# mask: sb = starbug detections + target, sbdol = + dolphot sources (band S/N > 3); BGD_R = bgdr x FWHM\n')
        fp.write('# err: dither-combined statistical; bgd: site-to-site scatter beyond noise x injected flux (background structure in the '
                 'complex); tot_err = err (+) bgd; inj_bias: median recovered/injected - 1 (not applied); ul when det_snr < 3\n')
        fp.write(f'# rings: PSF-residual flux in {RING[0]}-{RING[1]} FWHM / source flux for the target, injections (median, 84th pct) '
                 'and low-background dolphot field stars\n')
        out.to_csv(fp, index=False, float_format='%.4f')
    return out


def plot_maps(summ, fname):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=S.SURF)
    for ax, filt in zip(axes.flat, S.FILTERS):
        g = summ[summ['filter'] == filt]
        for k, mode in enumerate(MASKS):
            for j, bf in enumerate(BGDR_FACTORS):
                h = g[(g['mask'] == mode) & (g['bgdr'] == bf)].sort_values('box')
                ax.plot(h['box'] * (1 + 0.02 * (j - 1.5)), h['psf_bias'], '-o' if mode == 'sbdol' else '--o',
                        color=S.BLUE_RAMP[2 + 2 * j], mfc=None if mode == 'sbdol' else S.SURF, ms=4, lw=1.5,
                        label=f'BGD_R {bf} FWHM, {mode}' if filt == S.FILTERS[0] else None)
        t = g[g['tuned']].iloc[0]
        ax.plot([t['box']], [t['psf_bias']], '*', color=S.INK, ms=12, mfc='none', mew=1.3, label='tuned' if filt == S.FILTERS[0] else None)
        ax.axhline(0, color=S.INK2, lw=0.6)
        ax.set_xscale('log'); ax.set_xticks(BOX_SIZES, [str(b) for b in BOX_SIZES]); ax.set_xticks([], minor=True)
        ax.set_ylim(-0.6, 0.6)
        ax.tick_params(labelsize=7, colors=S.INK2)
        ax.set_title(f'{filt}: {int(t["n_sites"])} sites, {t["f_in"]:.1f} µJy; tuned: {t["mask"]}, box {int(t["box"])}, '
                     f'{int(t["bgdr"])} FWHM\nforced bias {t["psf_bias"]:+.2f}, σ {t["psf_scatter"]:.2f}; '
                     f'ap bias {t["ap_bias"]:+.2f}, σ {t["ap_scatter"]:.2f}', fontsize=7.5, color=S.INK)
    for ax in axes[-1]:
        ax.set_xlabel('BOX_SIZE (pix)', fontsize=8, color=S.INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel('forced PSF: recovered/injected − 1', fontsize=8, color=S.INK2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc='lower center', ncol=5)
    fig.suptitle('Injections in the target PAH complex (same sky sites in all dithers, combined like the target): forced-PSF bias '
                 'vs map; solid = starbug + dolphot mask, dashed = starbug only', fontsize=9.5, color=S.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(fname, dpi=140, facecolor=S.SURF)
    plt.close(fig)


def plot_residuals(summ, fname):
    '''Residual-ring fraction for target vs injections vs field stars, tuned map and default-like map.'''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=S.SURF)
    xs = np.arange(len(S.FILTERS))
    t = summ[summ['tuned']].set_index('filter').loc[S.FILTERS]
    ax.fill_between(xs, t['inj_ring_lo'], t['inj_ring_hi'], color=S.SERIES[0], alpha=0.15, lw=0, step='mid')
    ax.plot(xs, t['inj_ring_med'], 'o-', color=S.SERIES[0], lw=2, ms=5, label='injections in the complex (median, 16–84%)')
    ax.plot(xs, t['field_ring_med'], 's-', color=S.SERIES[2], lw=2, ms=5, label='low-background field stars (median)')
    ax.plot(xs, t['target_ring'], 'D-', color=S.SERIES[1], lw=2, ms=6, label='target')
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_xticks(xs, S.FILTERS, fontsize=8)
    ax.set_ylim(-1, 2)
    ax.tick_params(labelsize=8, colors=S.INK2)
    ax.set_ylabel(f'forced-PSF residual in {RING[0]}–{RING[1]} FWHM / flux', fontsize=8.5, color=S.INK2)
    ax.set_title('Leftover flux around the source after PSF subtraction (tuned map per band)', fontsize=9.5, color=S.INK)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(fname, dpi=140, facecolor=S.SURF)
    plt.close(fig)


def plot_sites(sites, fname):
    '''Cutout of the default bgd map around the target with the injection sites, per band.'''
    import matplotlib.pyplot as plt
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    frames = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_frames.csv')
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10), facecolor=S.SURF)
    for ax, filt in zip(axes.flat, S.FILTERS):
        f = sorted(frames.loc[frames['filter'] == filt, 'file'].unique())[0]
        raw = fits.getdata(f'{S.JHAT_DIR}/{f}.fits', 'SCI')
        w = WCS(fits.getheader(f'{S.JHAT_DIR}/{f}.fits', 'SCI'))
        xt, yt = frames.loc[frames['file'] == f, ['x_t', 'y_t']].iloc[0]
        h = R_SITE + 10
        sl = np.s_[int(yt) - h:int(yt) + h, int(xt) - h:int(xt) + h]
        cut = raw[sl]
        lo, hi = np.nanpercentile(cut, [2, 99.5])
        ax.imshow(cut, origin='lower', cmap='Greys_r', vmin=lo, vmax=hi, extent=(int(xt) - h, int(xt) + h, int(yt) - h, int(yt) + h))
        s = sites[sites['filter'] == filt]
        x, y = w.all_world2pix(s['ra'], s['dec'], 0)
        ax.scatter(x, y, s=50, facecolor='none', edgecolor=S.SERIES[0], lw=1.5)
        ax.plot(xt, yt, '+', color=S.SERIES[1], ms=12, mew=2)
        ax.add_patch(plt.Circle((xt, yt), R_SITE, fill=False, ec=S.INK2, lw=0.7, ls=':'))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{filt}: {len(s)} sites', fontsize=9, color=S.INK)
    fig.suptitle('Injection sites (circles) at the target\'s diffuse level, clear of dolphot and starbug sources; + target; '
                 f'dotted: {R_SITE} pix', fontsize=9.5, color=S.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=120, facecolor=S.SURF)
    plt.close(fig)


def main():
    import pandas as pd
    from astropy.io import fits
    os.makedirs(OUTP, exist_ok=True)
    if '--plots-only' not in sys.argv:
        dol = C.load_dolphot()
        ra_t, dec_t = C.target_position()
        files = sorted(glob.glob(f'{S.JHAT_DIR}/*_mirimage_jhat.fits'))
        by_filt = {}
        for f in files:
            by_filt.setdefault(fits.getheader(f)['FILTER'], []).append(f)
        with Pool(S.NCORES) as p:
            sites = pd.DataFrame([r for rows in p.map(choose_sites, [(by_filt[f][0], dol, ra_t, dec_t) for f in S.FILTERS]) for r in rows])
        sites.to_csv(f'{OUTP}/sites.csv', index=False)
        print(sites.groupby('filter').size())
        # low-background isolated dolphot field stars from the comparison run, within R_SITE + 100 pix
        comb = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_dolphot_compare_stars_combined.csv')
        args = []
        for f in files:
            filt = fits.getheader(f)['FILTER']
            fl = comb[(comb['filter'] == filt) & (comb['id'] >= 0) & (comb['x_contam'] < X_LOW)]
            fl = dol[dol['id'].isin(fl['id'])][['id', 'ra', 'dec']]
            args.append((f, dol, sites[sites['filter'] == filt], fl, ra_t, dec_t))
        with Pool(S.NCORES) as p:
            out = p.map(run_frame, args)
        pd.DataFrame([r for rows in out for r in rows]).to_csv(f'{OUTP}/frames.csv', index=False)
        shutil.rmtree(f'{OUTP}/tmp', ignore_errors=True)
    sites = pd.read_csv(f'{OUTP}/sites.csv')
    fr = pd.read_csv(f'{OUTP}/frames.csv')
    summ = summarise(combine_sites(fr))
    summ.to_csv(f'{OUTP}/summary.csv', index=False, float_format='%.4f')
    tuned = write_tuned(summ, f'{OUTP}/tuned_photometry.txt')
    plot_maps(summ, f'{OUTP}/pah_injection_maps.png')
    plot_residuals(summ, f'{OUTP}/pah_injection_residuals.png')
    plot_sites(sites, f'{OUTP}/pah_injection_sites.png')
    pd.set_option('display.width', 250)
    print(tuned.round(3).to_string())


if __name__ == '__main__':
    main()
