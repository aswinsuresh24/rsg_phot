'''
Blackbody fits to the observed SED of the SN 2026sqf progenitor candidate (obs.load_phot): sums of N = 1, 2, 3
blackbodies, each a sphere of temperature T and radius R at D = 19 Mpc, reddened by A_V = 0.125 (Gordon+23, R_V 3.1).

chi2: detections with sigma = err (+) 5% of the flux (as in the DUSTY fits); upper limits (3 sigma) one-sided,
penalised only when the model exceeds them. Bands are evaluated at their pivot wavelengths (no filter
convolution: for smooth blackbodies the difference is below the 5% floor). Global minimum by differential evolution.

python bb_fit.py   (synphot env, for dust_extinction)  -> printed table, plots/bb_fit.png
'''

import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from dust_extinction.parameter_averages import G23
from scipy.optimize import differential_evolution

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from obs import ROOT, load_phot

plt.style.use(ROOT / 'jwstrsg.mplstyle')

H, KB, C, SIGSB = 6.626e-27, 1.381e-16, 2.998e10, 5.6704e-5     # cgs
AU, LSUN, D = 1.496e13, 3.828e33, 19.0 * 3.0857e24               # cm, erg/s, cm
AV, RV = 0.01, 3.1
FLOOR, NSIG_UL = 0.05, 3
LOG_T, LOG_R = (np.log10(30), np.log10(6000)), (-3, 5)           # fit ranges: T (K), R (au)
NBB = (1, 2, 3)


def extinction(wave):
    # fraction transmitted; G23 covers 0.0912-32 um, extinction is negligible beyond
    ok = (wave > 0.0912) & (wave < 32)
    out = np.ones_like(wave)
    out[ok] = G23(Rv=RV).extinguish(wave[ok] * u.um, Av=AV)
    return out


def bb(wave, T, R):
    # uJy at D from a sphere of radius R (au) at temperature T (K); wave in um
    nu = C / (wave * 1e-4)
    bnu = 2 * H * nu**3 / C**2 / np.expm1(np.clip(H * nu / (KB * T), None, 700))
    return np.pi * bnu * (R * AU / D)**2 * 1e29


def components(theta):
    # theta = [log T1, log R1, log T2, log R2, ...] -> (T, R) pairs sorted hot to cold
    TR = 10**np.reshape(theta, (-1, 2))
    return TR[np.argsort(-TR[:, 0])]


def model(wave, theta, ext=None):
    ext = extinction(wave) if ext is None else ext
    return ext * sum(bb(wave, T, R) for T, R in components(theta))


def luminosity(T, R):
    return 4 * np.pi * (R * AU)**2 * SIGSB * T**4 / LSUN


hst_phot = load_phot(instruments=['HST'], name='hst')     # plotted only, not fitted
hst_det, hst_ul = ~hst_phot['uplim'], hst_phot['uplim']
hst_sig = np.hypot(hst_phot['fluxerr'], FLOOR * hst_phot['flux'])

phot = load_phot(instruments=['NIRCam', 'MIRI'], name='jwst')
det, ul = ~phot['uplim'], phot['uplim']
sig = np.hypot(phot['fluxerr'], FLOOR * phot['flux'])
EXT = extinction(phot['wave'])


def chi2(theta):
    m = model(phot['wave'], theta, EXT)
    over = np.clip(m[ul] - phot['flux'][ul], 0, None) / (phot['flux'][ul] / NSIG_UL)
    return np.sum(((phot['flux'][det] - m[det]) / sig[det])**2) + np.sum(over**2)


fits = {}
for n in NBB:
    res = differential_evolution(chi2, [LOG_T, LOG_R] * n, seed=1, tol=1e-10, maxiter=5000, polish=True)
    fits[n] = res.x
    dof, bic = det.sum() - 2 * n, res.fun + 2 * n * np.log(det.sum())
    print(f'\n{n} blackbody: chi2 {res.fun:.2f}  dof {dof}  BIC {bic:.1f}')
    for T, R in components(res.x):
        print(f'    T {T:7.0f} K   R {R:9.2f} au   L {luminosity(T, R):9.3g} Lsun')
    print(f'    total L {sum(luminosity(T, R) for T, R in components(res.x)):.3g} Lsun')


# --- plot: SED + fits (top), residuals in sigma (bottom)
textfontsize = 15
legendfontsize = 11
tickfontsize = 14
labelfontsize = 15
titlefontsize = 20
lam = np.geomspace(0.2, 30, 500)
fig, (ax, rx) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], figsize=(6, 5))
ax.errorbar(phot['wave'][det], phot['flux'][det], yerr=[phot['err_lo'][det], phot['err_hi'][det]],
            fmt='o', mfc='royalblue', mec='black', ms=6, zorder=5, label='NIRCam + MIRI photometry')
ax.errorbar(hst_phot['wave'][hst_det], hst_phot['flux'][hst_det], yerr=[hst_phot['err_lo'][hst_det], hst_phot['err_hi'][hst_det]],
            fmt='o', mfc='gray', mec='gray', ecolor='gray', ms=5, zorder=5, alpha=0.8, label='HST photometry')
ax.errorbar(phot['wave'][ul], phot['flux'][ul], yerr=0.3 * phot['flux'][ul], uplims=True,
            fmt='_', color='royalblue', ms=8, zorder=5, alpha=0.8)
ax.errorbar(hst_phot['wave'][hst_ul], hst_phot['flux'][hst_ul], yerr=0.3 * hst_phot['flux'][hst_ul], uplims=True,
            fmt='_', color='gray', ms=5, zorder=5, alpha=0.8)
for n, c in zip(NBB, ['C0', 'C1', 'C2']):
    lb_ = ' + '.join(f'{int(T)} K' for T, R in components(fits[n]))
    lb_ += f', $\\chi^2$ = {chi2(fits[n]):.1f}'
    ax.plot(lam, model(lam, fits[n]), color=c, label=lb_, lw=1.5)
    if n == NBB[-1]:
        for T, R in components(fits[n]):
            ax.plot(lam, extinction(lam) * bb(lam, T, R), color=c, ls=':', lw=1)
    rx.plot(phot['wave'][det], ((phot['flux'] - model(phot['wave'], fits[n])) / sig)[det], 'o-', color=c, ms=4)
rx.axhline(0, color='k', lw=0.8)
ax.set(xscale='log', yscale='log', xlim=(0.2, 30), ylim=(0.05, 200))
ax.set_ylabel(r'$F_\nu$ ($\mu$Jy)', fontsize=labelfontsize)
rx.set(xscale='log', xlim=(0.2, 30), ylim=(-5, 6))
rx.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize=labelfontsize)
rx.set_ylabel(r'$\Delta / \sigma$', fontsize=labelfontsize)
plt.subplots_adjust(hspace=0.09)
fig.align_ylabels([ax, rx])
ax.legend(fontsize=legendfontsize, loc='upper left', labelspacing=0.3, framealpha=0.9)
for a in (ax, rx):
    a.tick_params(axis='both', which='both', labelsize=tickfontsize)
fig.savefig(HERE.parent / 'plots' / 'bb_fit.png', bbox_inches='tight', dpi=400)