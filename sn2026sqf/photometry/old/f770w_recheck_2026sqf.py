'''
Dither-stack PSF fits with local background models and injection calibration (f1800w_recheck_2026sqf.py and
f1800w_map_fit_2026sqf.py) for the SN 2026sqf short-wavelength MIRI bands: F770W (7.7 um PAH band), with F560W and
F1000W as controls. Writes to its own folder; no photometry table is changed.

The summary per band mirrors build_photometry_table_new_2026sqf.py: median target flux over all r = R_REF variants
and starbug maps, divided by the injection recovery; error = fit (+) empty-site scatter (+) half the range over
variants, maps and R_SPREAD radii (+) the recovery correction and its uncertainty. It is compared with the adopted
(dolphot) value and the earlier starbug values.

Outputs, under data/dolphot/sn2026sqf/starbug/f770w_recheck/: results.csv, summary.txt, stack_fit.png (map-free fits),
map_fit_results.csv, map_fit_summary.csv, map_fit.png, comparison.txt
Run from the repo root after the PAH-injection script:  python sn2026sqf/f770w_recheck_2026sqf.py
'''
import os, shutil

import numpy as np
import pandas as pd

import starbug_miri_2026sqf as S
import starbug_pah_injection_2026sqf as P
import f1800w_recheck_2026sqf as R
import f1800w_map_fit_2026sqf as M

OUTD = f'{S.OUT}/f770w_recheck'
BANDS = ['F560W', 'F770W', 'F1000W']
R_REF, R_SPREAD = 8, [8, 11]
ADOPTED = f'{S.ROOT}/data/dolphot/sn2026sqf/sn2026sqf_photometry_adopted_ujy.txt'


def compare():
    s = pd.read_csv(f'{OUTD}/map_fit_summary.csv')
    a = pd.read_csv(ADOPTED, comment='#').set_index('filter')
    sb = pd.read_csv(f'{S.OUT}/sn2026sqf_miri_starbug_photometry.txt', comment='#').set_index('filter')
    tuned = pd.read_csv(f'{P.OUTP}/tuned_photometry.txt', comment='#').set_index('filter')
    n_sites = pd.read_csv(f'{P.OUTP}/sites.csv').groupby('filter').size()
    lines = []
    for filt in BANDS:
        g = s[s['filter'] == filt]
        r = g[g['rfit'] == R_REF]
        rec = r['rec'].median()
        f = r['target'].median() / rec
        stat, null = r['target_err'].median(), r['empty_sig'].median()
        sp = g[g['rfit'].isin(R_SPREAD)]
        corr = sp['target'] / sp['rec']
        method = 0.5 * (corr.max() - corr.min())
        calib = np.hypot(abs(1 / rec - 1), r['rec_sig'].median() / np.sqrt(n_sites[filt])) * f
        err = np.sqrt(stat ** 2 + null ** 2 + method ** 2 + calib ** 2)
        lines.append(f'== {filt}')
        for grp, vs in (('map-free', ['nomap_plane', 'nomap_quad']), ('map subtracted', ['sub_const', 'sub_plane']),
                        ('map template', ['tmpl_const', 'tmpl_plane'])):
            h = r[r['variant'].isin(vs)]
            lines.append(f'  {grp:15s} median {h["target"].median():6.2f}  range {h["target"].min():6.2f}-{h["target"].max():6.2f}'
                         f'  empty σ {h["empty_sig"].median():.2f}  recovery {h["rec"].median():.2f} ± {h["rec_sig"].median():.2f}'
                         f'  alpha {h["alpha"].median():+.2f}')
        lines.append(f'  stack estimate {f:.2f} ± {err:.2f} (stat {stat:.2f}, empty-site {null:.2f}, method {method:.2f}, '
                     f'calib {calib:.2f}; recovery {rec:.2f})')
        lines.append(f'  adopted {a.loc[filt, "flux_ujy"]:.2f} ± {a.loc[filt, "err_ujy"]:.2f} ({a.loc[filt, "source"]}); '
                     f'starbug forced PSF (main run) {sb.loc[filt, "psf_forced_ujy"]:.2f}, EE50 aperture {sb.loc[filt, "ap_ujy"]:.2f}; '
                     f'PAH-tuned forced PSF {tuned.loc[filt, "psf_forced_ujy"]:.2f}, aperture {tuned.loc[filt, "ap_ujy"]:.2f}')
    txt = '\n'.join(lines)
    open(f'{OUTD}/comparison.txt', 'w').write(txt + '\n')
    print(txt)


def main():
    os.makedirs(OUTD, exist_ok=True)
    R.OUTD = M.OUTD = OUTD
    R.BANDS = M.BANDS = BANDS
    R.main()
    shutil.move(f'{OUTD}/f1800w_recheck.png', f'{OUTD}/stack_fit.png')
    M.main()
    compare()


if __name__ == '__main__':
    main()
