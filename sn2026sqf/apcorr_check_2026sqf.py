'''
Which aperture correction fits the SN 2026sqf MIRI stage-2 (cal) frames: the CRDS table (jwst_miri_apcorr_0014, built
for resampled i2d mosaics) or one from the stpsf PSF (starbug_miri_2026sqf.aperture_setup)?

A. Curve-of-growth shape. dolphot point sources (type 1, band S/N > SNR_MIN, flag 0, crowding < 0.3,
   |sharpness| < 0.1), with no neighbour brighter than 10% of the star within r_EE70 + 2 FWHM and a local
   starbug-map level in the lower half of the candidates, measured in every frame on the cal frame with
   the CRDS sky annulus at the CRDS radii for EE 0.2-0.8. The measured flux(r_EE) / flux(r_EE70) is compared with
   each table's prediction, apcorr(r_EE70) / apcorr(r_EE); both include the PSF wings in the sky annulus. This
   does not depend on any absolute normalisation.
B. Absolute check. EE50 (and EE60) aperture fluxes of the isolated dolphot field stars from
   starbug_dolphot_compare_2026sqf.py with each correction applied, against dolphot, for low-background stars
   (local diffuse level x pi FWHM^2 < X_LOW x star flux).

Outputs, under data/dolphot/sn2026sqf/starbug/:
  sn2026sqf_miri_apcorr_check_stars.csv, sn2026sqf_miri_apcorr_check.csv, sn2026sqf_miri_apcorr_check.png

Run from the repo root after the two scripts above:  python apcorr_check_2026sqf.py [--plots-only]
'''
import os, sys, glob
from multiprocessing import Pool

import numpy as np

import starbug_miri_2026sqf as S

SNR_MIN, N_MAX = 20, 40
EE_REF = 0.7
X_LOW = 0.3
STARS_CSV = f'{S.OUT}/sn2026sqf_miri_apcorr_check_stars.csv'


def apcorr_rows(filt):
    '''CRDS rows (EE 0.2-0.8) with the matching stpsf correction for the same radius and annulus.'''
    t = S.crds_apcorr(filt)
    rin, rout = float(t['skyin'][0]), float(t['skyout'][0])
    r_ee, ee = S.encircled_energy(filt)
    ee_at = lambda r: np.interp(r, r_ee, ee)
    rows = []
    for row in t:
        r = float(row['radius'])
        ann = (r ** 2 / (rout ** 2 - rin ** 2)) * (ee_at(rout) - ee_at(rin))
        rows.append(dict(ee=round(float(row['eefraction']), 2), r=r, rin=rin, rout=rout,
                         apcorr_crds=float(row['apcorr']), apcorr_psf=1 / (ee_at(r) - ann)))
    return rows


def load_points():
    '''dolphot point sources: per-filter S/N, flux, sharpness, crowding, flag, plus RA/Dec.'''
    import starbug_dolphot_compare_2026sqf as C
    dol = C.load_dolphot()
    d = np.loadtxt(C.PHOT, usecols=range(C.N_BASE + C.N_PER_FILTER * len(S.FILTERS)))
    for k, f in enumerate(S.FILTERS):
        dol[f'{f}_sharp'] = d[:, C.N_BASE + C.N_PER_FILTER * k + 9]
    return dol


def run_frame(args):
    fname, dol = args
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    filt = fits.getheader(fname)['FILTER']
    fw = S.fwhm_pix(filt)
    bname = os.path.basename(fname).replace('.fits', '')
    bgd = fits.getdata(f'{S.OUT}/box{S.box_primary(filt)}_r{S.BGDR_PRIMARY}/{bname}-bgd.fits')
    rows_ap = apcorr_rows(filt)
    rout = rows_ap[0]['rout']
    with fits.open(fname) as h:
        img = h['SCI'].data * S.jy_scale(h)
        ny, nx = img.shape
        w = WCS(h['SCI'].header)
    x, y = w.all_world2pix(dol['ra'].values, dol['dec'].values, 0)
    d = dol.assign(x=x, y=y)
    d = d[(d['x'] > 400) & (d['x'] < nx - rout - 5) & (d['y'] > rout + 5) & (d['y'] < ny - rout - 5) & (d[f'{filt}_snr'] > 3)]
    fl = d[f'{filt}_ujy'].fillna(0).values
    dd = np.hypot(d['x'].values[:, None] - d['x'].values[None, :], d['y'].values[:, None] - d['y'].values[None, :])
    np.fill_diagonal(dd, np.inf)
    # no neighbour brighter than 10% of the star out to the EE70 aperture + 2 FWHM (the clipped annulus tolerates faint ones)
    r70 = [a['r'] for a in rows_ap if a['ee'] == EE_REF][0]
    iso = ~((dd < r70 + 2 * fw) & (fl[None, :] > 0.1 * fl[:, None])).any(axis=1)
    loc = np.array([bgd[int(round(b)), int(round(a))] for a, b in zip(d['x'], d['y'])]) - np.nanmedian(bgd)
    ok = (iso & (d['type'] == 1) & (d[f'{filt}_snr'] > SNR_MIN) & (d[f'{filt}_flag'] == 0)
          & (d[f'{filt}_crowd'] < 0.3) & (d[f'{filt}_sharp'].abs() < 0.1)).values
    if ok.any():
        ok &= loc < np.nanmedian(loc[ok])
    out = []
    for i in np.where(ok)[0][np.argsort(-d[f'{filt}_snr'].values[ok])][:N_MAX]:
        xi, yi = d['x'].values[i], d['y'].values[i]
        ann = CircularAnnulus((xi, yi), rows_ap[0]['rin'], rout).to_mask(method='center').get_values(img)
        _, sky, _ = sigma_clipped_stats(ann[np.isfinite(ann)], sigma=3)
        r = dict(file=bname, filter=filt, id=int(d['id'].values[i]), x=xi, y=yi, snr=d[f'{filt}_snr'].values[i], loc_bgd_mjysr=loc[i])
        for a in rows_ap:
            ap = CircularAperture((xi, yi), a['r'])
            r[f'f_ee{int(round(100 * a["ee"]))}'] = aperture_photometry(img, ap, method='exact')['aperture_sum'][0] - sky * ap.area
        out.append(r)
    return out


def summarise(stars):
    import pandas as pd
    rows = []
    ref = f'f_ee{int(round(100 * EE_REF))}'
    for filt in S.FILTERS:
        g = stars[stars['filter'] == filt]
        ap = {a['ee']: a for a in apcorr_rows(filt)}
        for ee, a in ap.items():
            ratio = g[f'f_ee{int(round(100 * ee))}'] / g[ref]
            med = np.nanmedian(ratio)
            rows.append(dict(filter=filt, ee=ee, r_pix=a['r'], n=len(g), measured=med,
                             measured_err=1.2533 * 1.4826 * np.nanmedian(np.abs(ratio - med)) / np.sqrt(max(len(g), 1)),
                             pred_psf=ap[EE_REF]['apcorr_psf'] / a['apcorr_psf'],
                             pred_crds=ap[EE_REF]['apcorr_crds'] / a['apcorr_crds'],
                             apcorr_psf=a['apcorr_psf'], apcorr_crds=a['apcorr_crds']))
    return pd.DataFrame(rows)


def dolphot_check():
    '''EE50 aperture with each correction / dolphot, for low-background isolated field stars.'''
    import pandas as pd
    comb = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_dolphot_compare_stars_combined.csv')
    comb = comb[(comb['id'] >= 0) & (comb['x_contam'] < X_LOW)]
    rows = []
    for filt in S.FILTERS:
        g = comb[comb['filter'] == filt]
        a = [r for r in S.aperture_setup(filt) if r['ee'] == S.EE_PRIMARY][0]
        k = a['apcorr_crds'] / a['apcorr_psf']
        row = dict(filter=filt, n_low_bgd=len(g))
        for lab, rat in (('stpsf', g['ap_ratio']), ('crds', g['ap_ratio'] * k), ('psf_forced', g['sb_ratio'])):
            row[f'{lab}_median'] = np.nanmedian(rat)
            row[f'{lab}_err'] = 1.2533 * 1.4826 * np.nanmedian(np.abs(rat - np.nanmedian(rat))) / np.sqrt(max(len(g), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summ, dchk, fname):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), facecolor=S.SURF)
    for ax, filt in zip(axes.flat, S.FILTERS):
        g = summ[summ['filter'] == filt].sort_values('ee')
        ax.plot(g['ee'], g['pred_crds'], '-', color=S.SERIES[1], lw=2, label='CRDS table (i2d)')
        ax.plot(g['ee'], g['pred_psf'], '-', color=S.SERIES[0], lw=2, label='stpsf (stage 2)')
        ax.errorbar(g['ee'], g['measured'], yerr=g['measured_err'], fmt='o', color=S.INK, ms=4, lw=1.2, label='measured stars')
        ax.axvline(S.EE_PRIMARY, color=S.INK2, lw=0.5, ls=':')
        ax.set_title(f'{filt} ({g["n"].iloc[0]} star-dither measurements)', fontsize=9, color=S.INK)
        ax.tick_params(labelsize=7, colors=S.INK2)
        ax.set_xlabel('EE fraction (CRDS radius)', fontsize=7.5, color=S.INK2)
        if filt in ('F560W', 'F1500W'):
            ax.set_ylabel(f'flux(r) / flux(r$_{{EE{int(100 * EE_REF)}}}$)', fontsize=8, color=S.INK2)
        if filt == S.FILTERS[0]:
            ax.legend(fontsize=7, frameon=False, loc='lower right')
    # absolute check against dolphot
    ax = axes.flat[-1]
    xs = np.arange(len(dchk))
    for k, (lab, col, name) in enumerate((('crds', S.SERIES[1], 'EE50 × CRDS'), ('stpsf', S.SERIES[0], 'EE50 × stpsf'),
                                          ('psf_forced', S.SERIES[6], 'forced PSF'))):
        m = dchk['n_low_bgd'] >= 3
        ax.errorbar(xs[m] + 0.15 * (k - 1), dchk[f'{lab}_median'][m], yerr=dchk[f'{lab}_err'][m], fmt='o', color=col, ms=4, lw=1.2, label=name)
    ax.axhline(1, color=S.INK2, lw=0.6)
    ax.set_xticks(xs, [f.replace('F', '').replace('W', '') for f in dchk['filter']], fontsize=7)
    ax.set_ylim(0.6, 1.6)
    ax.tick_params(labelsize=7, colors=S.INK2)
    ax.set_title(f'low-background field stars / dolphot', fontsize=9, color=S.INK)
    ax.set_xlabel('filter (µm × 100)', fontsize=7.5, color=S.INK2)
    ax.legend(fontsize=6.5, frameon=False, loc='upper right')
    fig.suptitle('Aperture correction on stage-2 cal frames: curve-of-growth shape of bright isolated low-background stars '
                 f'(plain CRDS annulus), and absolute fluxes vs dolphot (x < {X_LOW}, ≥ 3 stars)', fontsize=9.5, color=S.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fname, dpi=140, facecolor=S.SURF)
    plt.close(fig)


def main():
    import pandas as pd
    if '--plots-only' not in sys.argv:
        files = sorted(glob.glob(f'{S.JHAT_DIR}/*_mirimage_jhat.fits'))
        dol = load_points()
        with Pool(S.NCORES) as p:
            out = p.map(run_frame, [(f, dol) for f in files])
        pd.DataFrame([r for rows in out for r in rows]).to_csv(STARS_CSV, index=False)
    stars = pd.read_csv(STARS_CSV)
    summ = summarise(stars)
    dchk = dolphot_check()
    summ.to_csv(f'{S.OUT}/sn2026sqf_miri_apcorr_check.csv', index=False, float_format='%.4f')
    dchk.to_csv(f'{S.OUT}/sn2026sqf_miri_apcorr_check_dolphot.csv', index=False, float_format='%.4f')
    plot(summ, dchk, f'{S.OUT}/sn2026sqf_miri_apcorr_check.png')
    pd.set_option('display.width', 250)
    print(summ[summ['ee'].isin([0.3, 0.5, 0.6])].round(3).to_string())
    print(dchk.round(3).to_string())


if __name__ == '__main__':
    main()
