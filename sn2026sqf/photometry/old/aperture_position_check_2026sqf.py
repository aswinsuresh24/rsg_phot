'''
How sensitive is the adopted EE50 aperture photometry of SN 2026sqf (MIRI >= 10.5 um) to the aperture position?

The aperture sits at the F560W free-PSF position. In the long bands the free-PSF centroids moved 0.1-0.4" WSW
(starbug_miri_2026sqf.py), towards faint neighbours. For each band, every dither is rebuilt with the background map
tuned in starbug_pah_injection_2026sqf.py, and the EE50 aperture (stpsf apcorr, CRDS annulus on the subtracted
frame) is measured at:
  nominal     the F560W position (adopted)
  free_psf    the band's mean free-PSF centroid (fits within one HWHM); nominal if there are none
  nircam      the NIRCam dolphot counterpart, moved into the MIRI frame with the measured frame offset
  ring        OFFSETS_AS x 8 position angles around nominal
Dithers are combined with inverse-variance weights. The free_psf - nominal difference is the position systematic
used by build_photometry_table_2026sqf.py.

Outputs, under data/dolphot/sn2026sqf/starbug/position_check/: positions.csv, summary.csv
Run from the repo root after the PAH-injection script:  python sn2026sqf/aperture_position_check_2026sqf.py
'''
import os, glob
from multiprocessing import Pool

import numpy as np
import pandas as pd

import starbug_miri_2026sqf as S
import starbug_dolphot_compare_2026sqf as C
import starbug_pah_injection_2026sqf as P

OUTD = f'{S.OUT}/position_check'
AP_MIN_UM = 10.5
OFFSETS_AS = [0.05, 0.1, 0.2]
N_PA = 8


def positions(filt, frames, astro, ra_t, dec_t):
    '''Named (ra, dec) aperture centres for a band.'''
    cosd = np.cos(np.radians(dec_t))
    g = frames[(frames['filter'] == filt) & frames['primary']]
    g = g[S.psf_good(g)]
    pos = dict(nominal=(ra_t, dec_t))
    pos['free_psf'] = (g['ra_fit'].mean(), g['dec_fit'].mean()) if len(g) else (ra_t, dec_t)
    ra0, dec0 = S.load_target()
    nc = astro['nircam']
    pos['nircam'] = (ra0 + (nc['dra'] + astro['frame'][0]) / 3.6e6 / cosd, dec0 + (nc['ddec'] + astro['frame'][1]) / 3.6e6)
    for r in OFFSETS_AS:
        for k in range(N_PA):
            pa = 2 * np.pi * k / N_PA
            pos[f'ring_{r:.2f}_{int(np.degrees(pa))}'] = (ra_t + r * np.sin(pa) / 3600 / cosd, dec_t + r * np.cos(pa) / 3600)
    return pos


def run_frame(args):
    fname, pos, tuned, dol, ra_t, dec_t = args
    from astropy.io import fits
    filt = fits.getheader(fname)['FILTER']
    fw = S.fwhm_pix(filt)
    bname = os.path.basename(fname).replace('.fits', '')
    base, det, xt, yt, w, shape = P.frame_setup(fname, filt, ra_t, dec_t)
    ks = P.known_sources(fname, filt, dol, w, shape)
    tmp = f'{OUTD}/tmp'
    os.makedirs(tmp, exist_ok=True)
    opts = dict(base, BOX_SIZE=int(tuned['box']), BGD_R=int(tuned['bgdr']) * fw, OUTPUT=tmp + '/', QUIETMODE=1)
    sb, _ = S.subtract_background(fname, P.mask_list(det, ks, fw, w, tuned['mask']), opts)
    with fits.open(fname) as h:
        scale = S.jy_scale(h)
        err_jy = h['ERR'].data * scale
    ap_rows = S.aperture_setup(filt)
    t50 = S.tag_ee(S.EE_PRIMARY)
    rows = []
    for name, (ra, dec) in pos.items():
        x, y = [float(v) for v in w.all_world2pix(ra, dec, 0)]
        a = S.aperture_target(sb.image.data * scale, err_jy, x, y, ap_rows, 'ap', extra=False)
        rows.append(dict(file=bname, filter=filt, position=name, ra=ra, dec=dec,
                         offset_as=float(np.hypot((ra - ra_t) * np.cos(np.radians(dec_t)), dec - dec_t) * 3600),
                         ap_ujy=a[f'ap_{t50}_ujy'], ap_err_ujy=a[f'ap_{t50}_err_ujy']))
    for f in glob.glob(f'{tmp}/{bname}-*'):
        os.remove(f)
    return rows


def main():
    from astropy.io import fits
    os.makedirs(OUTD, exist_ok=True)
    frames = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_frames.csv')
    summ = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_injection_summary.csv')
    sel = {r['filter']: (r['box'], r['bgdr']) for _, r in summ[summ['primary']].iterrows()}
    frames['primary'] = [sel[f] == (b, r) for f, b, r in zip(frames['filter'], frames['box'], frames['bgdr'])]
    tuned = pd.read_csv(f'{P.OUTP}/tuned_photometry.txt', comment='#').set_index('filter')
    astro = S.astrometry_check()
    dol = C.load_dolphot()
    ra_t, dec_t = C.target_position()
    args = []
    for f in sorted(glob.glob(f'{S.JHAT_DIR}/*_mirimage_jhat.fits')):
        filt = fits.getheader(f)['FILTER']
        if tuned.loc[filt, 'wave_um'] < AP_MIN_UM:
            continue
        args.append((f, positions(filt, frames, astro, ra_t, dec_t), tuned.loc[filt], dol, ra_t, dec_t))
    with Pool(S.NCORES) as p:
        out = p.map(run_frame, args)
    d = pd.DataFrame([r for rows in out for r in rows])
    d.to_csv(f'{OUTD}/positions.csv', index=False)
    rows = []
    for (filt, name), g in d.groupby(['filter', 'position'], sort=False):
        f, e = S.wmean(g['ap_ujy'], g['ap_err_ujy'])
        rows.append(dict(filter=filt, position=name, offset_as=g['offset_as'].iloc[0], ap_ujy=f, ap_err_ujy=e))
    s = pd.DataFrame(rows)
    nom = s[s['position'] == 'nominal'].set_index('filter')['ap_ujy']
    s['delta_ujy'] = s['ap_ujy'] - s['filter'].map(nom)
    s['delta_frac'] = s['delta_ujy'] / s['filter'].map(nom)
    s.to_csv(f'{OUTD}/summary.csv', index=False, float_format='%.4f')
    shutil_tmp = f'{OUTD}/tmp'
    if os.path.isdir(shutil_tmp):
        os.rmdir(shutil_tmp)
    pd.set_option('display.width', 200)
    key = s[s['position'].isin(['nominal', 'free_psf', 'nircam'])]
    print(key.round(3).to_string())
    ring = s[s['position'].str.startswith('ring')].copy()
    ring['r'] = ring['position'].str.split('_').str[1].astype(float)
    print(ring.groupby(['filter', 'r'])['delta_frac'].agg(['mean', 'min', 'max']).round(3).to_string())


if __name__ == '__main__':
    main()
