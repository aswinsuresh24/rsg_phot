'''
Injection-recovery of the SN 2026sqf MIRI photometry in the target's PAH complex: background-structure errors and the
adopted MIRI fluxes.

Sites, per band (chosen on dither 1 in sky coordinates and used in all four dithers, so they are combined like the target):
  - R_MIN_FWHM FWHM to R_SITE pix from the target, clear of the frame edges and the coronagraph columns (x < 400);
  - local diffuse level like the target's: the band's starbug background map, minus its median within R_SITE, within
    LEVEL_TOL of its value at the target;
  - no listed source (starbug detection or dolphot S/N > 3) brighter than NB_FRAC x the injected flux within NB_FWHM
    FWHM (fainter neighbours are allowed, as around the target);
  - separated by more than the sky annulus + 2 FWHM, so apertures, annuli and PSF fits do not overlap.
stpsf point sources of the target's flux (its starbug.py aperture flux, at least 3 sigma) are injected at all sites
(nearest whole pixel) in each dither and measured by starbug.run_frame exactly like the target; per site, dithers are
combined. Pixel phase is therefore not tested (detector-sampled stamp: <~1% at EE >= 0.5).

Per band and method (forced PSF; aperture at the band's best EE), with r = recovered / injected over the sites, each
site weighted by 1 / (its distance from the target), so the statistics describe the target's surroundings:
  bias = wmedian(r) - 1,  scatter = 1.4826 wmedian(|r - wmedian(r)|),  noise = wmedian(error) / injected
  bgd  = sqrt(scatter^2 - noise^2) x injected: background structure, additive, does not average down over dithers
Adopted (aperture): F as measured (not bias-corrected). The bias enters as a one-sided term, bias_ujy = F (1 / (1 + bias) - 1),
the flux change if it were corrected: on the upper side when injections come back low (bias < 0), on the lower side when
they come back high (bias > 0). err_lo = stat (+) bgd [(+) |bias_ujy| if bias > 0]; err_hi = stat (+) bgd [(+) bias_ujy if
bias < 0]. Detected when F / stat >= 3 and not in FORCE_UL; otherwise
ul = max(F, 0) + 3 err_hi.

python injection.py   (after starbug.py) -> OUT/injection_sites.csv, injection_frames.csv, miri_adopted_ujy.csv, injection.png, injection_sites.png
'''
from multiprocessing import Pool
import numpy as np
import pandas as pd
import starbug as S

R_SITE, R_MIN_FWHM = 150, 4        # pix (16"), FWHM
LEVEL_TOL = 0.5                    # |E - E_target| < LEVEL_TOL |E_target|
NB_FRAC, NB_FWHM = 0.5, 2.5
N_SITES = 20
FORCE_UL = {'F2100W': 'formally detected, but no source is visible at the position by eye'}


def injected_flux(t):
    return max(t['ap_ujy'], 3 * t['ap_err_ujy'])


def choose_sites(filt, t, seed=1):
    # (ra, dec) of up to N_SITES sites around the target for band filt (t: its starbug.py row); seed: candidate order
    rows, x = S.run_frame(S.frames(filt)[0], [(t['ra'], t['dec'])], full=True)
    xt, yt, w, fw = rows[0]['x'], rows[0]['y'], x['wcs'], S.fwhm(filt)
    ny, nx = x['map'].shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - xt, yy - yt)
    E = x['map'] - np.nanmedian(x['map'][r < R_SITE])
    Et = E[int(round(yt)), int(round(xt))]
    sep_min = S.apertures(filt)[2][1] + 2 * fw
    cand = np.argwhere((r < R_SITE) & (r > max(R_MIN_FWHM * fw, sep_min)) & (np.abs(E - Et) < LEVEL_TOL * abs(Et))
                       & (xx > 400) & (xx < nx - 40) & (yy > 40) & (yy < ny - 40))
    np.random.default_rng(seed).shuffle(cand)
    dra, ddec, dflux = S.dolphot_sources(filt)
    dx, dy = w.all_world2pix(dra, ddec, 0)
    sx, sy = np.r_[x['det']['xcentroid'], dx], np.r_[x['det']['ycentroid'], dy]
    sf = np.r_[1e6 * np.nan_to_num(np.asarray(x['det']['flux'], float)), dflux]
    bright = sf > NB_FRAC * injected_flux(t)
    pts = []
    for yc, xc in cand:
        if (np.hypot(sx - xc, sy - yc)[bright] < NB_FWHM * fw).any() or any(np.hypot(xc - a, yc - b) < sep_min for a, b in pts):
            continue
        pts.append((xc, yc))
        if len(pts) == N_SITES:
            break
    ra, dec = w.all_pix2world(*np.transpose(pts), 0)
    return pd.DataFrame(dict(filter=filt, site=np.arange(1, len(pts) + 1), ra=ra, dec=dec, level=[E[b, a] for a, b in pts],
                             level_target=Et, r_pix=[np.hypot(a - xt, b - yt) for a, b in pts]))


def inject(fname, xy, f_ujy):
    # MJy/sr image with the detector-sampled stpsf stamp (total flux f_ujy) placed at whole-pixel positions xy, so the
    # injected PSF is exactly the fitted one (sub-pixel interpolation of the stamp would blur its core)
    from astropy.io import fits
    filt = fits.getheader(fname)['FILTER']
    psf = fits.getdata(f'{S.PSF_DIR}/{filt}.fits') * f_ujy * 1e-6
    with fits.open(fname) as h:
        scale = S.jy_scale(h)
    add, h = np.zeros(scale.shape, np.float32), psf.shape[0] // 2
    for x, y in xy:
        y0, y1, x0, x1 = max(y - h, 0), min(y + h + 1, add.shape[0]), max(x - h, 0), min(x + h + 1, add.shape[1])
        add[y0:y1, x0:x1] += psf[y0 - y + h:y1 - y + h, x0 - x + h:x1 - x + h] / scale[y0:y1, x0:x1]
    return add


def run_injected(fname, target, sites, f_ujy):
    # sites moved to the nearest whole pixel of this dither; target first (pos 0), sites as pos 1..N
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(fits.getheader(fname, 'SCI'))
    xy = np.round(w.all_world2pix(np.array(sites), 0)).astype(int)
    return S.run_frame(fname, [target] + [tuple(p) for p in w.all_pix2world(xy, 0)], inject(fname, xy, f_ujy))


def wmedian(x, w):
    o = np.argsort(x)
    return x[o][np.searchsorted(np.cumsum(w[o]) / np.sum(w), 0.5)]


def summarise(cs, ph, sites):
    rows = []
    for filt, g in cs.groupby('filter', sort=False):
        t = ph.loc[filt]
        inj, f_in, ap = g[g['pos'] > 0].sort_values('pos'), injected_flux(t), f'ap{int(100 * t["ee"])}'
        w = 1 / sites[sites['filter'] == filt].sort_values('site')['r_pix'].values
        row = dict(filter=filt, n_sites=len(inj), n_eff=w.sum() ** 2 / np.sum(w ** 2), f_in=f_in)
        for m, col in (('psf', 'psf'), ('ap', ap)):
            r = inj[f'{col}_ujy'].values / f_in
            med = wmedian(r, w)
            row.update({f'{m}_bias': med - 1, f'{m}_scatter': 1.4826 * wmedian(np.abs(r - med), w),
                        f'{m}_noise': wmedian(inj[f'{col}_err_ujy'].values, w) / f_in})
            row[f'{m}_bgd_ujy'] = np.sqrt(max(row[f'{m}_scatter'] ** 2 - row[f'{m}_noise'] ** 2, 0)) * f_in
        rows.append(row)
    return pd.DataFrame(rows).set_index('filter')


def adopt(ph, sm):
    rows = []
    for filt, t in ph.iterrows():
        s = sm.loc[filt]
        f, stat, bgd, b = t['ap_ujy'], t['ap_err_ujy'], s['ap_bgd_ujy'], s['ap_bias']
        shift = abs(f) * (1 / (1 + b) - 1)            # flux change if the bias were corrected: > 0 low recovery, < 0 high
        lo, hi = np.hypot(np.hypot(stat, bgd), min(shift, 0)), np.hypot(np.hypot(stat, bgd), max(shift, 0))
        det = f / stat >= 3 and filt not in FORCE_UL
        rows.append(dict(instrument='MIRI', filter=filt, wave_um=t['wave_um'], flux_ujy=f, err_lo_ujy=lo, err_hi_ujy=hi,
                         err_ujy=max(lo, hi), detected=det, ul_ujy=np.nan if det else max(f, 0) + 3 * hi, stat_ujy=stat,
                         bgd_ujy=bgd, bias_ujy=shift, inj_bias=b, n_sites=s['n_sites'], ee=t['ee']))
    return pd.DataFrame(rows)


def plot(sm, ph, fname):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = ph.loc[sm.index, 'wave_um'].values
    for m, c, dx in (('psf', 'C0', 0.98), ('ap', 'C1', 1.02)):
        ax.errorbar(w * dx, 1 + sm[f'{m}_bias'], sm[f'{m}_scatter'], fmt='o', color=c, capsize=2,
                    label={'psf': 'forced PSF', 'ap': 'aperture (best EE)'}[m])
    ax.axhline(1, color='k', lw=0.6)
    ax.set(xscale='log', xlabel=r'$\lambda$ ($\mu$m)', ylabel='recovered / injected (median, scatter)')
    ax.legend(fontsize=8)
    fig.savefig(fname, bbox_inches='tight', dpi=140)


def plot_sites(sites, cs, ph, fname):
    # dither-1 cal image around the target per band (detector orientation), sites coloured by aperture recovery
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.visualization import ImageNormalize, AsinhStretch
    fig, axs = plt.subplots(3, 3, figsize=(11, 11.6))
    hw = R_SITE + 15
    for ax, filt in zip(axs.flat, S.FILTERS):
        t, s = ph.loc[filt], sites[sites['filter'] == filt].sort_values('site')
        with fits.open(S.frames(filt)[0]) as h:
            img, w = h['SCI'].data, WCS(h['SCI'].header)
        xt, yt = [float(v) for v in w.all_world2pix(t['ra'], t['dec'], 0)]
        x0, y0 = int(xt) - hw, int(yt) - hw
        cut = img[y0:y0 + 2 * hw + 1, x0:x0 + 2 * hw + 1]
        yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
        near = cut[(np.hypot(xx - (xt - x0), yy - (yt - y0)) < R_SITE / 2) & np.isfinite(cut)]    # scale on the target's surroundings,
        lo, hi = np.percentile(near, [1, 99.5])                                                  # not the bright arm
        ax.imshow(cut, origin='lower', cmap='Greys', norm=ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(0.5)))
        inj = cs[(cs['filter'] == filt) & (cs['pos'] > 0)].sort_values('pos')
        rec = inj[f'ap{int(100 * t["ee"])}_ujy'].values / injected_flux(t)
        sx, sy = w.all_world2pix(s['ra'].values, s['dec'].values, 0)
        sc = ax.scatter(sx - x0, sy - y0, c=rec, cmap='coolwarm_r', vmin=0, vmax=2, s=70, edgecolors='k', linewidths=0.8)
        ax.plot(xt - x0, yt - y0, marker='+', color='lime', ms=16, mew=2)
        ax.add_patch(Circle((xt - x0, yt - y0), R_SITE, fill=False, ls='--', color='lime', lw=0.8))
        w = 1 / s['r_pix'].values
        med = wmedian(rec, w)
        ax.set_title(f'{filt}: {len(rec)} sites, recovery {med:.2f} ± {1.4826 * wmedian(np.abs(rec - med), w):.2f} (1/r)', fontsize=10)
        ax.set(xticks=[], yticks=[])
    fig.colorbar(sc, ax=axs, orientation='horizontal', fraction=0.03, pad=0.02, label='aperture recovered / injected')
    fig.suptitle(f'injection sites (dither 1, detector orientation); + target, dashed {R_SITE} pix = {0.11 * R_SITE:.0f}"', y=0.995)
    fig.savefig(fname, bbox_inches='tight', dpi=140)


if __name__ == '__main__':
    ph = pd.read_csv(f'{S.OUT}/starbug_photometry.csv').set_index('filter')
    with Pool(S.NCORES) as p:
        sites = pd.concat(p.starmap(choose_sites, [(f, ph.loc[f]) for f in S.FILTERS]))
    sites.to_csv(f'{S.OUT}/injection_sites.csv', index=False, float_format='%.7f')
    jobs = [(fn, (t['ra'], t['dec']), list(zip(*sites.loc[sites['filter'] == f, ['ra', 'dec']].values.T)), injected_flux(t))
            for f, t in ph.iterrows() for fn in S.frames(f)]
    with Pool(S.NCORES) as p:
        fr = pd.DataFrame([r for rows in p.starmap(run_injected, jobs) for r in rows])
    fr.to_csv(f'{S.OUT}/injection_frames.csv', index=False)
    cs = S.combine(fr)
    sm = summarise(cs, ph, sites)
    ad = adopt(ph, sm)
    with open(f'{S.OUT}/miri_adopted_ujy.csv', 'w') as fp:
        fp.write('# SN 2026sqf MIRI (2024-04-12), starbug aperture photometry at the best EE (starbug.py) with injection errors '
                 '(injection.py); err = stat (+) bgd (+) the one-sided bias term (bias_ujy: > 0 upper side, < 0 lower side); ul = 3 err_hi\n')
        fp.write('# forced upper limits: ' + '; '.join(f'{k} ({v})' for k, v in FORCE_UL.items()) + '\n')
        ad.to_csv(fp, index=False, float_format='%.4f')
    plot(sm, ph, f'{S.OUT}/injection.png')
    plot_sites(sites, cs, ph, f'{S.OUT}/injection_sites.png')
    pd.set_option('display.width', 200)
    print(sm.round(3).to_string())
    print(ad.round(3).to_string())
