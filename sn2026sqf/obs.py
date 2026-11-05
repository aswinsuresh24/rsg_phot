'''
Observed SED of the SN 2026sqf progenitor candidate.

PHOT_FILE is the photometry table in use: pre-explosion HST (dolphot, several epochs, mostly
limits), NIRCam (dolphot, 2026-03-20) and MIRI (starbug aperture photometry with injection-recovery
errors, 2024-04-12; sn2026sqf/photometry/). Fluxes in uJy, wavelengths in um. load_phot corrects
for Milky Way extinction by default: E(B-V)_MW from the SFD map at the SN position (dustmaps),
Schlafly & Finkbeiner 2011 recalibration, Gordon+23 law, R_V 3.1. Host extinction is left to the
model. The table on disk is not changed.

    from obs import load_phot
    phot = load_phot()                          # everything but the narrow bands
    phot = load_phot(instruments=('NIRCam', 'MIRI'))
    phot['inst_filt'], phot['wave'], phot['flux'], phot['fluxerr'], phot['uplim'], phot['mjd']

The dict has the layout of rsg_phot.mcmc_nest.sn2026sqf_phot, so it can go straight to nest
(but nest's ext then must be the host part only, or load with ebv_mw=0). Filters repeat across
HST epochs, so select by instrument / mjd rather than by filter name alone.
'''
from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from dust_extinction.parameter_averages import G23

ROOT = Path(__file__).resolve().parents[1]
PHOT_DIR = ROOT / 'data/dolphot/sn2026sqf'
PHOT_FILE = PHOT_DIR / 'sn2026sqf_hst_jwst_photometry.txt'

# Pa alpha and Br alpha narrow bands: line emission, left out of continuum fits by default
NARROW = ('F187N', 'F405N')

SN_COORD = SkyCoord('10:38:47.953 +53:30:34.08', unit=(u.hourangle, u.deg))
RV_MW = 3.1


def sfd_ebv(coord=SN_COORD, sf11=True):
    # Milky Way E(B-V) from the SFD98 map (dustmaps data_dir), x0.86 for the Schlafly & Finkbeiner 2011 recalibration
    from dustmaps.sfd import SFDQuery
    return float(SFDQuery()(coord)) * (0.86 if sf11 else 1.)


def mw_transmission(wave, ebv, rv=RV_MW):
    # fraction of flux transmitted through the Milky Way at wave (um); G23 covers 0.0912-32 um
    return np.asarray(G23(Rv=rv).extinguish(np.asarray(wave) * u.um, Ebv=ebv))


def load_phot(path=PHOT_FILE, exclude=NARROW, instruments=None, name='hst_jwst', ebv_mw=None):
    '''
    Observed photometry from a SN 2026sqf photometry table (PHOT_FILE by default).

    Detections: flux +- err (err = larger side; err_lo / err_hi keep the asymmetric errors).
    Non-detections: uplim True, flux = the upper limit (ul_ujy), errors NaN. exclude drops filters
    (default: the narrow bands); instruments keeps only those (e.g. ('NIRCam', 'MIRI'); None: all).
    name labels the table in 'index' (sn2026sqf_<name>, used for output file names).
    Fluxes, errors and limits are divided by the MW transmission at each band's pivot
    wavelength; ebv_mw = None queries SFD (SF11) at the SN, 0 returns the table values as observed.

    Returns a dict of arrays in table order: inst_filt, instrument, mjd, wave (um), flux,
    fluxerr, err_lo, err_hi (uJy), uplim (bool), source (dolphot / starbug), mw_trans (the MW
    transmission divided out), plus 'index' (a name for outputs) and 'ebv_mw' (the E(B-V) used).
    '''
    t = pd.read_csv(path, comment='#')
    t = t[~t['filter'].isin(exclude) & (instruments is None or t['instrument'].isin(instruments))]
    det = t['detected'].astype(str).eq('True').values
    flux = np.where(det, t['flux_ujy'], t['ul_ujy'])
    err, lo, hi = (np.where(det, t[f'{k}_ujy'], np.nan) for k in ('err', 'err_lo', 'err_hi'))

    wave = t['wave_um'].values
    ebv_mw = sfd_ebv() if ebv_mw is None else ebv_mw
    trans = mw_transmission(wave, ebv_mw) if ebv_mw > 0 else np.ones_like(wave)
    return {'index': f'sn2026sqf_{name}', 'ebv_mw': ebv_mw,
            'inst_filt': t['filter'].values, 'instrument': t['instrument'].values, 'mjd': t['mjd'].values,
            'wave': wave, 'flux': flux / trans, 'fluxerr': err / trans, 'err_lo': lo / trans, 'err_hi': hi / trans,
            'uplim': ~det, 'source': t['source'].values, 'mw_trans': trans}


if __name__ == '__main__':
    p = load_phot()
    print(f'{PHOT_FILE.name} ({p["index"]})')
    print(f'MW-corrected, E(B-V)_MW = {p["ebv_mw"]:.4f} (SFD, SF11)')
    print(f'{"instrument":10s} {"filter":7s} {"mjd":>9s} {"wave":>7s} {"flux":>8s} {"err":>7s} {"A_MW":>6s}')
    for i, f, m, w, fl, e, ul, t in zip(p['instrument'], p['inst_filt'], p['mjd'], p['wave'], p['flux'], p['fluxerr'],
                                        p['uplim'], p['mw_trans']):
        print(f'{i:10s} {f:7s} {m:9.1f} {w:7.3f} {"<" if ul else " "}{fl:7.3f} {e:7.3f} {-2.5*np.log10(t):6.4f}')
