'''
starbug2 photometry of the SN 2026sqf progenitor candidate on the JHAT-aligned MIRI stage-2 frames
(GO 3295, 2024-04-12; 9 filters x 4 dithers).

Per frame (run_frame):
  1. source list: starbug2 detections, the measured positions (each replacing a detection within 1 FWHM), and the
     dolphot MIRI sources with band S/N > 3, so the background map masks all of them;
  2. starbug2 background map (BOX_SIZE, BGD_R from MAP) and subtraction;
  3. forced PSF photometry at the measured positions (stpsf PSFs: dated OPD, detector distortion);
  4. aperture photometry at the CRDS encircled-energy radii, sky from the CRDS annulus of the subtracted frame,
     aperture correction measured the same way on the detector-sampled stpsf PSF (apertures).
Dithers are combined with inverse-variance weights (error: max of formal and dither scatter / sqrt N); best_ee()
picks the aperture. The target position is the mean free-centroid PSF fit to the F560W dithers (locate).

MAP: one background setting per band. The earlier sweep (old/starbug_pah_injection_2026sqf.py: BOX_SIZE 5-40 pix x
BGD_R 2-5 FWHM x two mask lists, injections in the target's PAH complex) moved the target's aperture flux by less than
its statistical error in every band, so the setting whose injections came back least biased is fixed here.

starbug2 0.7.7 workarounds: bgd_subtraction() leaves self.background set and photometry() would subtract it again, so
it is zeroed; APPhot_Routine biases the sky high on subtracted frames (it keeps only sky pixels > 0) and uses CRDS
apcorr made for i2d mosaics, so the apertures are done here with photutils.

python starbug.py   (synphot env: starbug2 0.7.7, stpsf 2.2)  -> OUT/starbug_frames.csv, starbug_photometry.csv, starbug_cog.png
'''
import os, glob, shutil, tempfile, warnings
from functools import lru_cache
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = f'{ROOT}/data/dolphot/sn2026sqf'
OUT = f'{DATA}/miri_phot'
PSF_DIR = f'{DATA}/starbug/psf'                   # stpsf PSFs {filt}.fits, also STARBUG_DATDIR
CRDS_APCORR = os.path.expanduser('~/data/crds_cache/references/jwst/miri/jwst_miri_apcorr_0014.fits')
DOLPHOT, DOLPHOT_REF = f'{DATA}/photometry/miri_0_0.phot', f'{DATA}/coadd_0_0_f560w_i2d.fits'
TARGET = (159.699812, 53.509469)                  # NIRCam dolphot position; refined on F560W by locate()
OBS_DATE = '2024-04-12T00:00:00'

FILTERS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
MAP = {'F560W': (5, 3), 'F770W': (5, 2), 'F1000W': (25, 2), 'F1130W': (5, 4), 'F1280W': (10, 2),      # BOX_SIZE (pix),
       'F1500W': (5, 2), 'F1800W': (25, 5), 'F2100W': (40, 3), 'F2550W': (25, 2)}                    # BGD_R (FWHM)
EE_LIST = [0.3, 0.4, 0.5, 0.6, 0.7]
APS = [f'ap{int(100 * e)}' for e in EE_LIST]
R_FIT = 60                                        # pix: listed sources this close to a position are PSF-fitted with it
NCORES = 9

os.environ['STARBUG_DATDIR'] = PSF_DIR
os.environ['STPSF_PATH'] = os.path.expanduser('~/data/stpsf-dl/stpsf-data')    # stpsf 2.2 needs data >= 2.1
warnings.filterwarnings('ignore')


def frames(filt=None):
    from astropy.io import fits
    files = sorted(glob.glob(f'{DATA}/jhat/*_mirimage_jhat.fits'))
    return [f for f in files if filt is None or fits.getheader(f)['FILTER'] == filt]


def fwhm(filt):
    import starbug2
    return starbug2.filters[filt].pFWHM


def psf_frac(filt):
    # starbug normalises the PSF stamp to unit sum; PSF fluxes are divided by the stamp's share of the total flux
    from astropy.io import fits
    return fits.getheader(f'{PSF_DIR}/{filt}.fits')['PSFFRAC']


def jy_scale(hdul):
    # Jy/pix per MJy/sr, per pixel
    from starbug2.utils import get_MJysr2Jy_scalefactor
    return get_MJysr2Jy_scalefactor(hdul['SCI']) * hdul['AREA'].data


def make_psf(filt):
    # stpsf PSF at the target's detector position with the OPD closest to the observation; kept if it exists
    if os.path.exists(f'{PSF_DIR}/{filt}.fits'):
        return
    import stpsf
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(frames(filt)[0]) as h:
        x, y = WCS(h['SCI'].header).all_world2pix(*TARGET, 0)
    m = stpsf.MIRI()
    m.filter, m.detector_position = filt, (int(round(float(x))), int(round(float(y))))
    m.load_wss_opd_by_date(OBS_DATE, choice='closest', verbose=False, plot=False)
    psf = m.calc_psf(fov_pixels=101, oversample=4)
    psf['DET_DIST'].header['PSFFRAC'] = float(psf['DET_DIST'].data.sum())
    os.makedirs(PSF_DIR, exist_ok=True)
    fits.writeto(f'{PSF_DIR}/{filt}.fits', psf['DET_DIST'].data, psf['DET_DIST'].header)


@lru_cache
def apertures(filt):
    '''
    Per EE in EE_LIST: radius (pix, CRDS table) and the stpsf aperture correction, 1 / (aperture sum - annulus mean x area)
    of the detector-sampled PSF stamp on a total-flux scale, measured exactly as the data are (so it includes the PSF
    wings in the sky annulus and the detector broadening); plus the annulus (r_in, r_out).
    (The CRDS apcorr is made for drizzled i2d mosaics; the EE of the oversampled stpsf PSF lacks the detector effects and
    under-corrects small apertures: 18% at F560W EE30, 2-5% at EE50.)
    '''
    from astropy.io import fits
    from astropy.table import Table
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    t = Table.read(CRDS_APCORR)
    t = t[(t['filter'] == filt) & (t['subarray'] == 'FULL')]
    t.sort('eefraction')
    psf = fits.getdata(f'{PSF_DIR}/{filt}.fits')
    c = ((psf.shape[1] - 1) / 2, (psf.shape[0] - 1) / 2)
    ann = (float(t['skyin'][0]), float(t['skyout'][0]))
    sky = aperture_photometry(psf, CircularAnnulus(c, *ann), method='exact')['aperture_sum'][0] / (np.pi * (ann[1] ** 2 - ann[0] ** 2))
    r = np.interp(EE_LIST, t['eefraction'], t['radius'])
    corr = [1 / (aperture_photometry(psf, CircularAperture(c, ri), method='exact')['aperture_sum'][0] - sky * np.pi * ri ** 2) for ri in r]
    return r, np.array(corr), ann


def aperture(img, err, x, y, filt):
    # flux and error (uJy) per EE at (x, y); img, err in Jy/pix; sky = 3-sigma-clipped median in the annulus
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from astropy.stats import sigma_clipped_stats
    radii, corr, ann = apertures(filt)
    sky = CircularAnnulus((x, y), *ann).to_mask(method='center').get_values(img)
    sky = sky[np.isfinite(sky)]
    _, s, sd = sigma_clipped_stats(sky, sigma=3)
    out = []
    for r, c in zip(radii, corr):
        a = CircularAperture((x, y), r)
        p = aperture_photometry(img, a, error=err, method='exact')
        e = np.sqrt(p['aperture_sum_err'][0] ** 2 + a.area * sd ** 2 + (a.area * sd) ** 2 / len(sky))
        out += [1e6 * c * (p['aperture_sum'][0] - s * a.area), 1e6 * c * e]
    return out


@lru_cache
def dolphot_sources(filt):
    # RA, Dec, flux (uJy) of dolphot MIRI sources with band S/N > 3 (.phot: 12 global columns, then 13 per filter:
    # AB mag the 5th, S/N the 9th)
    from astropy.io import fits
    from astropy.wcs import WCS
    k = 12 + 13 * FILTERS.index(filt)
    d = np.loadtxt(DOLPHOT, usecols=(2, 3, k + 4, k + 8))
    d = d[d[:, 3] > 3]
    ra, dec = WCS(fits.getheader(DOLPHOT_REF, 'SCI')).all_pix2world(d[:, 0] - 0.5, d[:, 1] - 0.5, 0)
    return ra, dec, 3631e6 * 10 ** (-0.4 * d[:, 2])


def add_rows(det, x, y, w):
    # append sources at pixel positions (x, y) to a starbug detection table
    from astropy.table import vstack
    new = vstack([det[:1]] * len(x))
    for c in new.colnames:
        if new[c].dtype.kind == 'f':
            new[c] = np.nan
    new['RA'], new['DEC'] = w.all_pix2world(x, y, 0)
    new['xcentroid'], new['ycentroid'], new['flag'] = x, y, 0
    return vstack([det, new])


def run_frame(fname, positions, add=None, free=False, full=False):
    '''
    One frame: background map and subtraction, then PSF and aperture photometry at positions [(ra, dec), ...].
    add: image (MJy/sr) added to the frame first (injections). free: free-centroid PSF fit (locate) instead of forced.
    Returns one dict per position; full: also the background map (MJy/sr), the source list and the WCS.
    '''
    from astropy.io import fits
    from astropy.wcs import WCS
    from starbug2.starbug import StarbugBase
    filt = fits.getheader(fname)['FILTER']
    fw, (box, bgdr) = fwhm(filt), MAP[filt]
    tmp = tempfile.mkdtemp(dir=OUT)
    sb = StarbugBase(fname, options=dict(VERBOSE=0, QUIETMODE=1, FWHM=fw, SIGSRC=5.0, PSF_SIZE=max(11, int(4 * fw)) | 1,
                                         CALC_CROWD=0, BOX_SIZE=box, BGD_R=bgdr * fw, OUTPUT=tmp + '/',
                                         FORCE_POS=int(not free), MAX_XYDEV='1000p'))
    if add is not None:
        sb.image.data = sb.image.data + add
    w, ny, nx = WCS(sb.image.header), *sb.image.data.shape
    x, y = w.all_world2pix(*np.transpose(positions), 0)

    # source list: detections (minus those at the positions), positions, dolphot sources not already listed
    sb.detect()
    det = sb.detections
    sep = lambda xs, ys, ax, ay: np.hypot(np.subtract.outer(xs, ax), np.subtract.outer(ys, ay)).min(axis=1)
    det = det[sep(det['xcentroid'], det['ycentroid'], x, y) > fw]
    dx, dy = w.all_world2pix(*dolphot_sources(filt)[:2], 0)
    keep = ((dx > -20) & (dx < nx + 20) & (dy > -20) & (dy < ny + 20)
            & (sep(dx, dy, np.r_[det['xcentroid'], x], np.r_[det['ycentroid'], y]) > fw))
    det = add_rows(det, np.r_[x, dx[keep]], np.r_[y, dy[keep]], w)

    sb.detections = det
    raw = sb.image.data.copy()
    sb.bgd_estimate()
    sb.bgd_subtraction()
    sb.background = fits.ImageHDU(data=np.zeros_like(sb.image.data))
    sb.detections = det[sep(det['xcentroid'], det['ycentroid'], x, y) < R_FIT]
    sb.photometry()
    cat, pf = sb.psfcatalogue, psf_frac(filt)
    with fits.open(fname) as h:
        scale = jy_scale(h)
        err = h['ERR'].data * scale
    rows = []
    for k, (xk, yk) in enumerate(zip(x, y)):
        s = cat[np.argmin(np.hypot(cat['x_init'] - xk, cat['y_init'] - yk))]
        row = dict(file=os.path.basename(fname), filter=filt, pos=k, x=xk, y=yk, ra_fit=float(s['RA']),
                   dec_fit=float(s['DEC']), xydev=float(s['xydev']), psf_ujy=1e6 * s['flux'] / pf, psf_err_ujy=1e6 * s['eflux'] / pf)
        row.update(zip([f'{a}_{q}' for a in APS for q in ('ujy', 'err_ujy')], aperture(sb.image.data * scale, err, xk, yk, filt)))
        rows.append(row)
    shutil.rmtree(tmp)
    return (rows, dict(map=raw - sb.image.data, det=det, wcs=w)) if full else rows


def locate():
    # target position: mean free-centroid PSF fit over the F560W dithers that stayed within one HWHM
    with Pool(4) as p:
        rows = [r[0] for r in p.starmap(run_frame, [(f, [TARGET], None, True) for f in frames('F560W')])]
    ok = [r for r in rows if r['xydev'] <= fwhm('F560W') / 2]
    return float(np.mean([r['ra_fit'] for r in ok])), float(np.mean([r['dec_fit'] for r in ok]))


def wmean(f, e):
    f, e = np.asarray(f, float), np.asarray(e, float)
    w = 1 / e ** 2
    return np.sum(w * f) / np.sum(w), max(np.sqrt(1 / np.sum(w)), np.std(f, ddof=1) / np.sqrt(len(f)))


def combine(fr):
    # inverse-variance mean over dithers, per band and position
    out = []
    for (filt, pos), g in fr.groupby(['filter', 'pos'], sort=False):
        row = dict(filter=filt, pos=pos)
        for m in ['psf'] + APS:
            row[f'{m}_ujy'], row[f'{m}_err_ujy'] = wmean(g[f'{m}_ujy'], g[f'{m}_err_ujy'])
        out.append(row)
    return pd.DataFrame(out)


def best_ee(row):
    '''
    Aperture for one band (a row of combine()): the highest-S/N EE among those whose flux is within 1 sigma of the
    smallest aperture's. A point source has a flat (aperture-corrected) curve of growth; flux that appears only in
    larger apertures comes from neighbours or extended emission (the target's curve rises beyond EE50-60).
    '''
    f = np.array([row[f'{a}_ujy'] for a in APS])
    e = np.array([row[f'{a}_err_ujy'] for a in APS])
    return EE_LIST[int(np.argmax(np.where(np.abs(f - f[0]) < e, f / e, -np.inf)))]


def plot_cog(ph, fname):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, r in ph.iterrows():
        f = np.array([r[f'{a}_ujy'] for a in APS]) / r['ap_ujy']
        e = np.array([r[f'{a}_err_ujy'] for a in APS]) / r['ap_ujy']
        x = np.array(EE_LIST) + 0.004 * (k - 4)
        ax.errorbar(x, f, e, marker='o', ms=3, lw=1, capsize=0, label=r['filter'])
        ax.plot(r['ee'] + 0.004 * (k - 4), 1, 'k*', ms=8)
    ax.axhline(1, color='k', lw=0.6)
    ax.set(xlabel='encircled energy of the aperture', ylabel='flux / flux at the chosen aperture (*)', ylim=(0, 2.5))
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(fname, bbox_inches='tight', dpi=140)


if __name__ == '__main__':
    import starbug2
    os.makedirs(OUT, exist_ok=True)
    for f in FILTERS:
        make_psf(f)
    ra, dec = locate()
    print(f'F560W position {ra:.7f} {dec:.7f}')
    with Pool(NCORES) as p:
        fr = pd.DataFrame([r for rows in p.starmap(run_frame, [(f, [(ra, dec)]) for f in frames()]) for r in rows])
    fr.to_csv(f'{OUT}/starbug_frames.csv', index=False)
    ph = combine(fr).drop(columns='pos')
    ph.insert(1, 'wave_um', [starbug2.filters[f].wavelength for f in ph['filter']])
    ph.insert(2, 'ra', ra)
    ph.insert(3, 'dec', dec)
    ph['ee'] = ph.apply(best_ee, axis=1)
    ph['ap_ujy'] = [r[f'ap{int(100 * r["ee"])}_ujy'] for _, r in ph.iterrows()]
    ph['ap_err_ujy'] = [r[f'ap{int(100 * r["ee"])}_err_ujy'] for _, r in ph.iterrows()]
    ph.to_csv(f'{OUT}/starbug_photometry.csv', index=False, float_format='%.10g')
    plot_cog(ph, f'{OUT}/starbug_cog.png')
    print(ph[['filter', 'ee', 'ap_ujy', 'ap_err_ujy', 'psf_ujy', 'psf_err_ujy']].round(2).to_string())
