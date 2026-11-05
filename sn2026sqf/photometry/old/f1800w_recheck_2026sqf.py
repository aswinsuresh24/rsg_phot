'''
Independent recheck of the SN 2026sqf long-wavelength MIRI photometry (F1800W, with F1500W and F2100W as controls),
without the starbug background map.

For each band the four cal dithers are cut out around a position, shifted to a common sub-pixel centre (same
position angle, so a pure shift) and averaged with inverse-variance weights. On the stack:
  - psf_plane:  F * PSF + plane (b0 + bx x + by y), linear least squares within R_FIT (several radii);
  - psf_quad:   F * PSF + quadratic surface (plane + x^2 + y^2 + xy), which can follow a diffuse ridge crest;
  - psf_ext:    as psf_plane plus a circular Gaussian of FWHM EXT_FWHM centred on the source (extended bump);
  - ap:         EE50 aperture with the stpsf correction and the CRDS annulus (plain annulus, no starbug map).
The same is done at the PAH-complex sites of starbug_pah_injection_2026sqf.py, empty (noise / background floor) and
with an injected stpsf source of INJ_UJY (bias), and at random empty positions within R_NULL of the target.

Outputs, under data/dolphot/sn2026sqf/starbug/f1800w_recheck/: results.csv, summary.txt, f1800w_recheck.png
Run from the repo root after the PAH-injection script:  python sn2026sqf/f1800w_recheck_2026sqf.py
'''
import os

import numpy as np
import pandas as pd

import starbug_miri_2026sqf as S
import starbug_dolphot_compare_2026sqf as C
import starbug_pah_injection_2026sqf as P

OUTD = f'{S.OUT}/f1800w_recheck'
BANDS = ['F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
HALF = 22                          # stack half-size (pix)
R_FIT = [4, 6, 8, 11]              # pix
EXT_FWHM = [4, 8]                  # pix, extended component
N_NULL, R_NULL = 30, 150           # random empty positions within R_NULL pix


def frames_for(filt):
    import glob
    from astropy.io import fits
    return [f for f in sorted(glob.glob(f'{S.JHAT_DIR}/*_mirimage_jhat.fits')) if fits.getheader(f)['FILTER'] == filt]


def load_frames(filt):
    '''Per dither: image and error in uJy/pix, WCS.'''
    from astropy.io import fits
    from astropy.wcs import WCS
    out = []
    for f in frames_for(filt):
        with fits.open(f) as h:
            sc = S.jy_scale(h) * 1e6
            img, err = h['SCI'].data * sc, h['ERR'].data * sc
            dq = h['DQ'].data
            img = np.where((dq & 1) > 0, np.nan, img)
            out.append(dict(file=f, img=img, err=err, wcs=WCS(h['SCI'].header)))
    return out


def stack(frames, ra, dec, add=None):
    '''Inverse-variance stack of cutouts shifted so (ra, dec) sits at the centre pixel. add: per-frame image to add.'''
    from scipy.ndimage import shift
    num = den = 0
    for k, fr in enumerate(frames):
        x, y = [float(v) for v in fr['wcs'].all_world2pix(ra, dec, 0)]
        xi, yi = int(round(x)), int(round(y))
        sl = np.s_[yi - HALF - 3:yi + HALF + 4, xi - HALF - 3:xi + HALF + 4]
        im = fr['img'][sl] + (0 if add is None else add[k][sl])
        er = fr['err'][sl]
        good = np.isfinite(im) & np.isfinite(er) & (er > 0)
        im, w = np.where(good, im, 0), np.where(good, 1 / er ** 2, 0)
        dx, dy = xi - x, yi - y
        ims = shift(im * w, (dy, dx), order=1, mode='nearest')[3:-3, 3:-3]
        ws = shift(w, (dy, dx), order=1, mode='nearest')[3:-3, 3:-3]
        num, den = num + ims, den + ws
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den, 1 / np.sqrt(den)


def psf_image(filt):
    '''stpsf PSF on the stack grid, centred, total flux 1 (stamp sum = PSFFRAC).'''
    from astropy.io import fits
    from photutils.psf import FittableImageModel
    p = fits.getdata(f'{S.PSF_DIR}/{filt}.fits')
    m = FittableImageModel(p, normalize=True)
    m.flux, m.x_0, m.y_0 = fits.getheader(f'{S.PSF_DIR}/{filt}.fits')['PSFFRAC'], HALF, HALF
    yy, xx = np.mgrid[:2 * HALF + 1, :2 * HALF + 1]
    return m(xx, yy)


def fit(img, err, psf, rfit, ext=None, quad=False):
    '''Linear least squares: F psf + [A gauss] + b0 + bx x + by y [+ x^2 + y^2 + xy] within rfit. Returns F, sigma_F, A.'''
    yy, xx = np.mgrid[:2 * HALF + 1, :2 * HALF + 1] - HALF
    m = (np.hypot(xx, yy) <= rfit) & np.isfinite(img) & np.isfinite(err)
    cols = [psf[m]]
    if ext is not None:
        s = ext / 2.3548
        g = np.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2))
        cols.append((g / g.sum())[m])
    cols += [np.ones(m.sum()), xx[m], yy[m]]
    if quad:
        cols += [xx[m] ** 2, yy[m] ** 2, (xx * yy)[m]]
    A = np.vstack(cols).T
    w = 1 / err[m]
    coef, *_ = np.linalg.lstsq(A * w[:, None], img[m] * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    resid = (img[m] - A @ coef) * w
    scale = np.sqrt(max(np.sum(resid ** 2) / (m.sum() - A.shape[1]), 1))
    return coef[0], np.sqrt(cov[0, 0]) * scale, (coef[1] if ext is not None else np.nan)


def aperture(img, err, filt):
    a = [r for r in S.aperture_setup(filt) if r['ee'] == S.EE_PRIMARY][0]
    out = S.aperture_target(img * 1e-6, err * 1e-6, HALF, HALF, [a], 'ap', extra=False)
    return out[f'ap_{S.tag_ee(S.EE_PRIMARY)}_ujy'], out[f'ap_{S.tag_ee(S.EE_PRIMARY)}_err_ujy']


def measure(img, err, psf, filt, **key):
    rows = []
    f, e = aperture(img, err, filt) if HALF > S.aperture_setup(filt)[0]['rout'] else (np.nan, np.nan)
    rows.append(dict(key, method='ap', rfit=np.nan, ext=np.nan, flux=f, err=e, ext_flux=np.nan))
    for r in R_FIT:
        f, e, _ = fit(img, err, psf, r)
        rows.append(dict(key, method='psf_plane', rfit=r, ext=np.nan, flux=f, err=e, ext_flux=np.nan))
        f, e, _ = fit(img, err, psf, r, quad=True)
        rows.append(dict(key, method='psf_quad', rfit=r, ext=np.nan, flux=f, err=e, ext_flux=np.nan))
        for x in EXT_FWHM:
            f, e, a = fit(img, err, psf, r, x)
            rows.append(dict(key, method='psf_ext', rfit=r, ext=x, flux=f, err=e, ext_flux=a))
    return rows


def main():
    os.makedirs(OUTD, exist_ok=True)
    ra_t, dec_t = C.target_position()
    sites = pd.read_csv(f'{P.OUTP}/sites.csv')
    rng = np.random.default_rng(1800)
    rows, stacks = [], {}
    for filt in BANDS:
        frames = load_frames(filt)
        psf = psf_image(filt)
        f_inj = P.target_flux(filt)
        img, err = stack(frames, ra_t, dec_t)
        stacks[filt] = (img, err, psf)
        rows += measure(img, err, psf, filt, filter=filt, kind='target', pos=-1, f_in=np.nan)
        s = sites[sites['filter'] == filt]
        from astropy.io import fits
        add = []
        for fr in frames:
            pts = list(zip(*fr['wcs'].all_world2pix(s['ra'].values, s['dec'].values, 0)))
            with fits.open(fr['file']) as h:
                add.append(P.inject(fr['file'], filt, pts, f_inj) * S.jy_scale(h) * 1e6)
        for k, (ra, dec) in enumerate(zip(s['ra'], s['dec'])):
            im, er = stack(frames, ra, dec)
            rows += measure(im, er, psf, filt, filter=filt, kind='site_empty', pos=k, f_in=0.0)
            im, er = stack(frames, ra, dec, add)
            rows += measure(im, er, psf, filt, kind='site_inj', filter=filt, pos=k, f_in=f_inj)
        # random positions near the target (no source avoidance: the generic local null)
        w0 = frames[0]['wcs']
        xt, yt = [float(v) for v in w0.all_world2pix(ra_t, dec_t, 0)]
        n = 0
        while n < N_NULL:
            r, th = rng.uniform(4 * S.fwhm_pix(filt), R_NULL), rng.uniform(0, 2 * np.pi)
            x, y = xt + r * np.cos(th), yt + r * np.sin(th)
            if not (400 < x < 1000 and 40 < y < 980):
                continue
            ra, dec = w0.all_pix2world(x, y, 0)
            im, er = stack(frames, float(ra), float(dec))
            rows += measure(im, er, psf, filt, filter=filt, kind='random', pos=n, f_in=0.0)
            n += 1
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUTD}/results.csv', index=False)
    np.savez_compressed(f'{OUTD}/stacks.npz', **{f'{k}_{n}': v for k, (a, b, c) in stacks.items() for n, v in zip(('img', 'err', 'psf'), (a, b, c))})
    summarise(d)
    plot(d, stacks)


def summarise(d):
    lines = []
    for filt in BANDS:
        g = d[d['filter'] == filt]
        lines.append(f'== {filt}')
        for (meth, rfit, ext), h in g.groupby(['method', 'rfit', 'ext'], dropna=False):
            t = h[h['kind'] == 'target'].iloc[0]
            e = h[h['kind'] == 'site_empty']['flux']
            i = h[h['kind'] == 'site_inj']
            rnd = h[h['kind'] == 'random']['flux']
            rec = (i['flux'].values - e.values) / i['f_in'].values      # injected flux recovered on top of the site
            raw = i['flux'].values / i['f_in'].values
            lab = f'{meth:9s} r={rfit:>4} ext={ext:>3}'
            lines.append(f'{lab}: target {t.flux:6.2f} ± {t.err:5.2f}'
                         f' | empty sites {np.median(e):+6.2f} (σ {1.4826 * np.median(np.abs(e - np.median(e))):5.2f})'
                         f' | random {np.median(rnd):+6.2f} (σ {1.4826 * np.median(np.abs(rnd - np.median(rnd))):5.2f})'
                         f' | injected recovered/in {np.median(raw):.2f} (σ {1.4826 * np.median(np.abs(raw - np.median(raw))):.2f})'
                         f', minus empty {np.median(rec):.2f}'
                         + (f' | target ext flux {t.ext_flux:6.2f}' if meth == 'psf_ext' else ''))
    txt = '\n'.join(lines)
    open(f'{OUTD}/summary.txt', 'w').write(txt + '\n')
    print(txt)


def plot(d, stacks):
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, AsinhStretch
    fig, axes = plt.subplots(len(BANDS), 4, figsize=(14, 3.4 * len(BANDS)), facecolor=S.SURF)
    yy, xx = np.mgrid[:2 * HALF + 1, :2 * HALF + 1] - HALF
    rr = np.hypot(xx, yy)
    for i, filt in enumerate(BANDS):
        img, err, psf = stacks[filt]
        g = d[(d['filter'] == filt)]
        t = g[(g['kind'] == 'target') & (g['method'] == 'psf_plane') & (g['rfit'] == 8)].iloc[0]
        # plane from the same fit, for display
        m = rr <= 8
        A = np.vstack([psf[m], np.ones(m.sum()), xx[m], yy[m]]).T
        coef, *_ = np.linalg.lstsq(A / err[m][:, None], img[m] / err[m], rcond=None)
        plane = coef[1] + coef[2] * xx + coef[3] * yy
        sub = img - plane
        res = sub - coef[0] * psf
        sig = np.nanmedian(err)
        for j, (im, title) in enumerate(((img, 'stack, cal frames'), (sub, 'minus local plane (r ≤ 8 pix fit)'),
                                         (res, 'minus plane and PSF'))):
            ax = axes[i, j]
            if j == 0:
                lo, hi = np.nanpercentile(im, [2, 99.5])
                nrm = ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(0.2))
            else:
                nrm = ImageNormalize(vmin=-4 * sig, vmax=10 * sig)
            ax.imshow(im, origin='lower', cmap='Greys_r', norm=nrm)
            ax.plot(HALF, HALF, '+', color=S.SERIES[1], ms=10, mew=1.5)
            ap = [r for r in S.aperture_setup(filt) if r['ee'] == S.EE_PRIMARY][0]
            ax.add_patch(plt.Circle((HALF, HALF), ap['r'], fill=False, ec=S.SERIES[1], lw=1))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9, color=S.INK)
        axes[i, 0].set_ylabel(filt, fontsize=10, color=S.INK)
        # radial profile of the plane-subtracted stack against the PSF at several fluxes
        ax = axes[i, 3]
        rb = np.arange(0, 13, 1.0)
        rc = 0.5 * (rb[1:] + rb[:-1])
        prof = np.array([np.nanmean(sub[(rr >= a) & (rr < b)]) for a, b in zip(rb[:-1], rb[1:])])
        perr = np.array([np.nanmean(err[(rr >= a) & (rr < b)]) / np.sqrt(((rr >= a) & (rr < b)).sum()) for a, b in zip(rb[:-1], rb[1:])])
        ax.errorbar(rc, prof, yerr=perr, fmt='o', color=S.INK, ms=4, label='target − plane')
        pp = np.array([np.mean(psf[(rr >= a) & (rr < b)]) for a, b in zip(rb[:-1], rb[1:])])
        f_ap = g[(g['kind'] == 'target') & (g['method'] == 'ap')]['flux'].iloc[0]
        for f, col, lab in ((t.flux, S.SERIES[0], f'PSF fit {t.flux:.1f} µJy'), (f_ap, S.SERIES[1], f'aperture {f_ap:.1f} µJy')):
            ax.plot(rc, f * pp, '-', color=col, lw=2, label=lab)
        ax.axhline(0, color=S.INK2, lw=0.6)
        ax.set_xlabel('r (pix)', fontsize=8, color=S.INK2)
        ax.set_ylabel('µJy / pix', fontsize=8, color=S.INK2)
        ax.tick_params(labelsize=7, colors=S.INK2)
        ax.legend(fontsize=7, frameon=False)
        if i == 0:
            ax.set_title('radial profile', fontsize=9, color=S.INK)
    fig.suptitle('Dither stack on the F560W position (no starbug map): local plane + stpsf PSF fit; circle = EE50 aperture',
                 fontsize=10, color=S.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f'{OUTD}/f1800w_recheck.png', dpi=130, facecolor=S.SURF)


if __name__ == '__main__':
    main()
