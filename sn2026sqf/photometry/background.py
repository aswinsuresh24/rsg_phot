'''
Independent check of the SN 2026sqf MIRI photometry: a simple parametric local background instead of the starbug2 map.

Per band, the four cal dithers are cut out around a position, shifted to a common sub-pixel centre and averaged with
inverse-variance weights (stack). On the stack, linear least squares within R_FIT pix of the centre, position fixed:
  plane   F PSF + b0 + bx x + by y
  quad    plane + x^2 + y^2 + xy (follows a curved ridge)
for each model and radius (the variants), at the target and at the injection sites of injection.py, both empty and
with the same stpsf source injected at the site (sub-pixel, so the stack shift blurs it like the target). Sites are
weighted by 1 / (distance from the target), as in injection.py. Per variant: recovery = weighted median over sites of
(injected - empty) / injected, and target = F / recovery. Estimate = median over variants; error = median fit
error (+) weighted empty-site scatter / recovery (+) half the range over variants.

The estimate is compared with the adopted MIRI photometry in ADOPTED (starbug fluxes, Monte Carlo injection errors).

python background.py   (after injection.py) -> OUT/background_fits.csv, OUT/miri_comparison.csv, background.png
'''
import numpy as np
import pandas as pd
import starbug as S
from injection import injected_flux, wmedian

HALF = 22                    # stack half-size (pix)
R_FIT = (6, 8, 11)           # pix
ADOPTED = f'{S.DATA}/sn2026sqf_hst_jwst_photometry.txt'


def load(filt):
    # per dither: image and error (uJy/pix, DQ-flagged pixels NaN), WCS
    from astropy.io import fits
    from astropy.wcs import WCS
    out = []
    for f in S.frames(filt):
        with fits.open(f) as h:
            sc = S.jy_scale(h)
            img = np.where(h['DQ'].data & 1, np.nan, h['SCI'].data * sc * 1e6)
            out.append(dict(file=f, img=img, err=h['ERR'].data * sc * 1e6, wcs=WCS(h['SCI'].header)))
    return out


def inject(fr, filt, sites, f_ujy):
    # uJy/pix image of stpsf sources of total flux f_ujy at sites [(ra, dec), ...] in one dither
    from astropy.io import fits
    from photutils.psf import FittableImageModel
    m = FittableImageModel(fits.getdata(f'{S.PSF_DIR}/{filt}.fits'), normalize=True)
    add = np.zeros(fr['img'].shape)
    yy, xx = np.mgrid[:add.shape[0], :add.shape[1]]
    for x, y in fr['wcs'].all_world2pix(np.array(sites), 0):
        m.flux, m.x_0, m.y_0 = f_ujy * S.psf_frac(filt), x, y
        sl = np.s_[max(int(y) - 50, 0):int(y) + 51, max(int(x) - 50, 0):int(x) + 51]
        add[sl] += m(xx[sl], yy[sl])
    return add


def stack(frames, ra, dec, adds=None):
    # inverse-variance stack of cutouts shifted so (ra, dec) sits on the centre pixel; adds: per-dither uJy/pix images
    from scipy.ndimage import shift
    num = den = 0
    for k, fr in enumerate(frames):
        x, y = [float(v) for v in fr['wcs'].all_world2pix(ra, dec, 0)]
        xi, yi = int(round(x)), int(round(y))
        sl = np.s_[yi - HALF - 3:yi + HALF + 4, xi - HALF - 3:xi + HALF + 4]
        im = fr['img'][sl] + (0 if adds is None else adds[k][sl])
        good = np.isfinite(im) & (fr['err'][sl] > 0)
        w = np.where(good, fr['err'][sl] ** -2., 0)
        num = num + shift(np.where(good, im, 0) * w, (yi - y, xi - x), order=1, mode='nearest')[3:-3, 3:-3]
        den = den + shift(w, (yi - y, xi - x), order=1, mode='nearest')[3:-3, 3:-3]
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den, den ** -0.5


def psf_image(filt):
    # stpsf PSF on the stack grid, centred, total flux 1
    from astropy.io import fits
    from photutils.psf import FittableImageModel
    m = FittableImageModel(fits.getdata(f'{S.PSF_DIR}/{filt}.fits'), normalize=True)
    m.flux, m.x_0, m.y_0 = S.psf_frac(filt), HALF, HALF
    yy, xx = np.mgrid[:2 * HALF + 1, :2 * HALF + 1]
    return m(xx, yy)


def fit(img, err, psf, rfit, quad):
    # F and its error (scaled by the reduced chi^2 when > 1) for F psf + plane [+ quadratic] within rfit
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    m = (np.hypot(xx, yy) <= rfit) & np.isfinite(img) & np.isfinite(err)
    x, y = xx[m], yy[m]
    A = np.vstack([psf[m], np.ones_like(x), x, y] + ([x ** 2, y ** 2, x * y] if quad else [])).T / err[m, None]
    b = img[m] / err[m]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    chi2 = np.sum((b - A @ coef) ** 2) / (len(b) - A.shape[1])
    return coef[0], np.sqrt(np.linalg.inv(A.T @ A)[0, 0] * max(chi2, 1))


def band(filt, target, sites, f_in):
    frames, psf = load(filt), psf_image(filt)
    variants = [(r, q) for r in R_FIT for q in (False, True)]
    adds = [inject(fr, filt, sites, f_in) for fr in frames]
    rows = []
    for k, pos in enumerate([target] + sites):
        empty, injected = stack(frames, *pos), stack(frames, *pos, adds) if k else None
        for r, q in variants:
            f, e = fit(*empty, psf, r, q)
            rows.append(dict(filter=filt, pos=k, rfit=r, model='quad' if q else 'plane', flux=f, err=e,
                             injected=fit(*injected, psf, r, q)[0] if k else np.nan))
    return rows


def summarise(fits_):
    # fits_ needs f_in and r_pix (site distance) per row
    out = []
    for filt, g in fits_.groupby('filter', sort=False):
        est = []
        for (r, mdl), v in g.groupby(['rfit', 'model']):
            t, s = v[v['pos'] == 0].iloc[0], v[v['pos'] > 0]
            w, e = 1 / s['r_pix'].values, s['flux'].values
            rec = wmedian(((s['injected'] - s['flux']) / s['f_in']).values, w)
            sig = 1.4826 * wmedian(np.abs(e - wmedian(e, w)), w)
            est.append(dict(flux=t['flux'] / rec, fit_err=t['err'] / rec, empty_sig=sig / rec, rec=rec))
        est = pd.DataFrame(est)
        f = est['flux'].median()
        err = np.sqrt(est['fit_err'].median() ** 2 + est['empty_sig'].median() ** 2 + (np.ptp(est['flux']) / 2) ** 2)
        out.append(dict(filter=filt, param_ujy=f, param_err_ujy=err, rec=est['rec'].median(), variant_range=np.ptp(est['flux'])))
    return pd.DataFrame(out).set_index('filter')


def compare(param):
    # adopted MIRI photometry (ADOPTED) against the parametric estimate, per band
    t = pd.read_csv(ADOPTED, comment='#')
    t = t[t['instrument'] == 'MIRI'].set_index('filter').loc[S.FILTERS]
    c = t[['wave_um', 'flux_ujy', 'err_lo_ujy', 'err_hi_ujy', 'detected', 'ul_ujy']].copy()
    c['param_ujy'], c['param_err_ujy'] = param['param_ujy'], param['param_err_ujy']
    c['rec'] = param['rec']
    d = c['param_ujy'] - c['flux_ujy']         # adopted error on the side of the difference
    c['param_vs_adopted_sigma'] = d / np.hypot(np.where(d > 0, c['err_hi_ujy'], c['err_lo_ujy']), c['param_err_ujy'])
    return c


def plot(c, fname):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    det, w = c['detected'].astype(bool), c['wave_um']
    ax.errorbar(w[det], c['flux_ujy'][det], [c['err_lo_ujy'][det], c['err_hi_ujy'][det]], fmt='o', color='k', ms=5, capsize=2,
                label='adopted (starbug, MC injection errors)')
    ax.errorbar(w[~det], c['ul_ujy'][~det], yerr=0.25 * c['ul_ujy'][~det], uplims=True, fmt='_', color='k', ms=8)
    ax.errorbar(w * 1.03, c['param_ujy'], c['param_err_ujy'], fmt='s', color='C0', ms=4, capsize=2, label='parametric background')
    ax.set(xscale='log', yscale='log', xlabel=r'$\lambda$ ($\mu$m)', ylabel=r'$F_\nu$ ($\mu$Jy)', ylim=(0.8, 120))
    ax.legend(fontsize=8)
    fig.savefig(fname, bbox_inches='tight', dpi=140)


if __name__ == '__main__':
    from multiprocessing import Pool
    ph = pd.read_csv(f'{S.OUT}/starbug_photometry.csv').set_index('filter')
    sites = pd.read_csv(f'{S.OUT}/injection_sites.csv')
    jobs = [(f, (t['ra'], t['dec']), [tuple(p) for p in sites.loc[sites['filter'] == f, ['ra', 'dec']].values], injected_flux(t))
            for f, t in ph.iterrows()]
    with Pool(S.NCORES) as p:
        fits_ = pd.DataFrame([r for rows in p.starmap(band, jobs) for r in rows])
    fits_['f_in'] = fits_['filter'].map({f: injected_flux(t) for f, t in ph.iterrows()})
    fits_ = fits_.merge(sites[['filter', 'site', 'r_pix']].rename(columns={'site': 'pos'}), on=['filter', 'pos'], how='left')
    fits_.to_csv(f'{S.OUT}/background_fits.csv', index=False, float_format='%.4f')
    c = compare(summarise(fits_))
    c.to_csv(f'{S.OUT}/miri_comparison.csv', float_format='%.3f')
    plot(c, f'{S.OUT}/background.png')
    pd.set_option('display.width', 250)
    print(c.round(2).to_string())
