'''
Adopted SN 2026sqf progenitor photometry: dolphot for NIRCam and MIRI blueward of 10 um, starbug EE50 aperture
photometry for MIRI redward of 10 um.

MIRI >= AP_MIN_UM (F1130W-F2550W): EE50 aperture flux with the stpsf aperture correction (sky from the CRDS
annulus on the background-subtracted frame), measured with the background map tuned on injections in the target's
PAH complex (starbug_pah_injection_2026sqf.py: tuned_photometry.txt, summary.csv). Errors:
  stat      dither-combined statistical error
  bgd       site-to-site scatter of injected sources beyond pixel noise x injected flux (background structure)
  map       std of the target flux over all background maps tried (BOX_SIZE x BGD_R x mask list)
  loss      injected sources come back low by the median bias b < 0, so the flux may be underestimated by
            F (1 / (1 + b) - 1); this one-sided term enters the upper error only (no correction is applied)
  err_lo = stat (+) bgd (+) map;  err_hi = err_lo (+) loss;  err = max(err_lo, err_hi) for symmetric likelihoods
MIRI < AP_MIN_UM (F560W-F1000W) and NIRCam: dolphot (TARGET_UJY_FILE), err_lo = err_hi = stat (+) bgd for MIRI,
where bgd is the same background-structure term measured for forced PSF photometry at the PAH-complex sites (it
applies to any PSF method in this complex); NIRCam errors are dolphot's.
Detection: flux / (stat (+) map) >= 3 (aperture) or the dolphot detection flag; otherwise ul = max(F, 0) + 3 err_hi.

Run from the repo root:  python build_photometry_table_2026sqf.py
Writes data/dolphot/sn2026sqf/sn2026sqf_photometry_adopted_ujy.txt
'''
import numpy as np
import pandas as pd

from starbug_miri_2026sqf import ROOT, OUT, TARGET_UJY_FILE

AP_MIN_UM = 10.5
PAH_DIR = f'{OUT}/pah_injection'
OUTFILE = f'{ROOT}/data/dolphot/sn2026sqf/sn2026sqf_photometry_adopted_ujy.txt'


def main():
    dol = pd.read_csv(TARGET_UJY_FILE, sep=r'\s+', na_values='---')
    tuned = pd.read_csv(f'{PAH_DIR}/tuned_photometry.txt', comment='#').set_index('filter')
    summ = pd.read_csv(f'{PAH_DIR}/summary.csv')
    rows = []
    for _, d in dol.iterrows():
        filt, wave = d['filter'], float(d['wave_um'])
        row = dict(instrument=d['instrument'], filter=filt, wave_um=wave)
        if d['instrument'] == 'MIRI' and wave >= AP_MIN_UM:
            t = tuned.loc[filt]
            f, stat, bgd = t['ap_ujy'], t['ap_err_ujy'], t['ap_bgd_ujy']
            sweep = summ.loc[summ['filter'] == filt, 'target_ap_ujy']
            mapsys = float(np.std(sweep))
            b = min(t['ap_inj_bias'], 0.0)
            loss = abs(f) * (1 / (1 + b) - 1) if b > -1 else np.inf
            err_lo = np.sqrt(stat ** 2 + bgd ** 2 + mapsys ** 2)
            err_hi = np.hypot(err_lo, loss)
            det = f / np.hypot(stat, mapsys) >= 3
            row.update(flux_ujy=f, err_lo_ujy=err_lo, err_hi_ujy=err_hi, err_ujy=max(err_lo, err_hi), detected=det,
                       stat_ujy=stat, bgd_ujy=bgd, map_ujy=mapsys, loss_ujy=loss, inj_bias=t['ap_inj_bias'],
                       n_sites=int(t['n_sites']), source=f'starbug EE50 aperture, stpsf apcorr ({t["mask"]}, box {int(t["box"])}, '
                                                         f'BGD_R {int(t["bgdr"])} FWHM)')
        else:
            f, stat = d['flux_ujy'], d['flux_err_ujy']
            bgd = 0.0
            if d['instrument'] == 'MIRI' and filt in tuned.index:
                bgd = tuned.loc[filt, 'psf_forced_bgd_ujy']
            err = np.hypot(stat, bgd) if np.isfinite(stat) else np.nan
            row.update(flux_ujy=f, err_lo_ujy=err, err_hi_ujy=err, err_ujy=err, detected=str(d['detected']) == 'True',
                       stat_ujy=stat, bgd_ujy=bgd, map_ujy=0.0, loss_ujy=0.0, inj_bias=np.nan, n_sites=np.nan, source='dolphot')
            if not row['detected']:
                row['ul_ujy'] = float(d['ul_ujy'])
        if row['detected']:
            row['ul_ujy'] = np.nan
        elif 'ul_ujy' not in row or not np.isfinite(row.get('ul_ujy', np.nan)):
            row['ul_ujy'] = max(row['flux_ujy'], 0) + 3 * row['err_hi_ujy']
        f = row['flux_ujy']
        row['mag_ab'] = -2.5 * np.log10(f * 1e-6 / 3631) if row['detected'] and f > 0 else np.nan
        row['mag_err'] = 1.0857 * row['err_ujy'] / f if row['detected'] and f > 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = ['instrument', 'filter', 'wave_um', 'flux_ujy', 'err_lo_ujy', 'err_hi_ujy', 'err_ujy', 'detected', 'ul_ujy', 'mag_ab',
            'mag_err', 'stat_ujy', 'bgd_ujy', 'map_ujy', 'loss_ujy', 'inj_bias', 'n_sites', 'source']
    with open(OUTFILE, 'w') as fp:
        fp.write('# SN 2026sqf progenitor candidate, adopted photometry (uJy, AB), made by build_photometry_table_2026sqf.py\n')
        fp.write(f'# NIRCam (2026-03-20) and MIRI < {AP_MIN_UM} um: dolphot ({TARGET_UJY_FILE.replace(ROOT + "/", "")}); MIRI bgd term added\n')
        fp.write(f'# MIRI >= {AP_MIN_UM} um (2024-04-12): starbug EE50 aperture x stpsf apcorr, background map tuned on injections in '
                 'the target PAH complex (data/dolphot/sn2026sqf/starbug/pah_injection/)\n')
        fp.write('# err_lo = stat (+) bgd (+) map; err_hi = err_lo (+) loss (one-sided: injected sources recovered low by inj_bias, '
                 'not corrected); err = max; ul = 3 sigma (err_hi) when not detected\n')
        fp.write('# not simultaneous: NIRCam 2026-03-20, MIRI 2024-04-12\n')
        out[cols].to_csv(fp, index=False, float_format='%.4f')
    pd.set_option('display.width', 250)
    print(out[cols[:-1]].round(3).to_string())


if __name__ == '__main__':
    main()
