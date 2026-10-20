'''
StarbugII photometry of the SN 2026sqf progenitor candidate on the JHAT-aligned MIRI stage-2 frames
(GO 3295, 2024-04-12; 9 filters x 4 dithers of 27.75 s).

Pass 1 (locate): free-centroid PSF fits of the four F560W dithers at the TARGET_FILE position; the mean
of the fits within one HWHM is the target position used for every band in pass 2 (MIRI frame, so the
small MIRI - NIRCam frame offset drops out).

Pass 2, for each frame:
  1. starbug2 detection (blind). The target is put in the source list at the F560W position, replacing a
     detection within 1 FWHM if there is one (at F1130W and redward there usually is not; an unlisted
     target would be fitted as background).
  2. starbug2 diffuse-background map (BackGround_Estimate_Routine) with a fixed mask radius
     BGD_R = f * FWHM for every source, swept over BOX_SIZES x BGDR_FACTORS. The spread over the sweep is
     the background systematic. The primary map of each band is the one whose injected sources (step 5)
     come back least biased, |forced PSF ratio - 1| + |aperture ratio - 1|.
  3. starbug2 bgd_subtraction, then PSF photometry against stpsf DET_DIST PSFs (dated OPD, cruciform
     detector scattering, target detector position) in PSF_MODES:
       forced   FORCE_POS = 1 at the F560W position
       xyNN     free centroid, refit with the position forced when it moves more than MAX_XYDEV = N.N pix
                (starbug's own mechanism; 0.12" = 1.1 MIRI pix)
       free     free centroid, no refit; only fits within one HWHM are used when combining
     starbug normalises the PSF stamp to unit sum, so PSF fluxes are divided by the stamp's encircled
     fraction PSFFRAC to put them on a total-flux scale.
  4. Aperture photometry at the F560W position, done here rather than with starbug's APPhot_Routine (see
     "starbug2 0.7.7 workarounds"):
       - radii at encircled-energy fractions EE_LIST of the CRDS MIRI apcorr table, primary EE_PRIMARY;
       - ap:        sky from a sigma-clipped median in the CRDS annulus (skyin, skyout) of the
                    background-subtracted frame (a residual correction to the starbug map);
       - ap_rawann: the same aperture and annulus on the cal frame without the starbug map (plain
                    annulus photometry, for comparison);
       - ap_nosky:  the starbug map as the only sky;
       - aperture correction from the stpsf PSF, 1 / [EE(r) - (A_ap / A_ann)(EE(r_out) - EE(r_in))], which
         also accounts for PSF wings inside the sky annulus. The CRDS apcorr (column ap_crds) is built for
         resampled i2d mosaics, whose PSF is broadened by drizzling; on these stage-2 frames it
         over-corrects small apertures (F560W EE30: 3.34 vs 2.07) and gives a falling curve of growth;
       - the curve of growth across EE_LIST.
  5. Injection-recovery test of the whole chain: N_REAL realisations of N_INJ stpsf point sources at the
     dolphot flux (INJ_FLUX), 4-10 FWHM from the target, added to the cal frame before the background map
     and listed in the source list like the target. Recovered/injected flux per method and the injected
     curve of growth say whether the background map leaves point sources unbiased and flat at this
     local background level (stpsf is both injected and fitted, so PSF-model errors are not tested).
     The injected scatter beyond pixel noise, sqrt(scatter^2 - noise^2) x injected flux, is the
     background-structure error (*_bgd_ujy, additive). It is fixed on the sky, so it does not average down
     over dithers.
  6. PSF zero-point check: aperture/PSF flux ratio of bright, isolated stars (primary map, free fit).

Dithers are combined with inverse-variance weights. The statistical error is the larger of the formal
error and the dither scatter / sqrt(N); the total adds the background-structure and map-sweep errors. Bands with S/N < 3 also get a 3 sigma upper limit, max(flux, 0) + 3 err.

starbug2 0.7.7 workarounds:
  - bgd_subtraction() leaves self.background set, so the CLI chain -B -S -P subtracts the background
    twice. After subtracting, self.background is replaced with zeros. (Setting it to None makes
    photometry() subtract a global sigma-clipped median instead.)
  - APPhot_Routine selects sky pixels with `dat>0 & np.isfinite(dat)`, which evaluates to dat>0. On a
    background-subtracted frame that drops the negative half of the sky and biases the sky high.
  - APPhot_Routine applies CRDS apcorr values computed for a different sky annulus than its own.
  - bgd_subtraction() and GEN_RESIDUAL both write <name>-res.fits; they are renamed -bgdsub.fits and
    -res_{mode}.fits.

Outputs, under data/dolphot/sn2026sqf/starbug/:
  psf/                                      stpsf PSFs, also used as STARBUG_DATDIR
  box{N}_r{f}/                              -bgd, -bgdsub, -res_free, -res_forced (box closest to 2 FWHM, BGD_R 2 FWHM)
  cutouts/                                  101 px cutouts of cal, bgd, bgd-subtracted and residuals for every map
  sn2026sqf_miri_starbug_frames.csv         per-frame, per-map target photometry
  sn2026sqf_miri_starbug_injections.csv     per injected source
  sn2026sqf_miri_starbug_injection_summary.csv   recovery per band, map and method
  sn2026sqf_miri_starbug_zp.csv             per-frame aperture/PSF ratios of bright isolated stars
  sn2026sqf_miri_starbug_photometry.txt     combined per-band photometry (uJy and AB mag)
  plots: *_residuals.png, *_identity.png, *_psfmodes.png, *_injection.png, *_cog.png, *_sed.png

Run from the repo root:  python starbug_miri_2026sqf.py [--plots-only]
(synphot conda env: starbug2 0.7.7, stpsf 2.2.0; stpsf data >= 2.1 under STPSF_DATA)
'''
import os, sys, glob, shutil, warnings, zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
JHAT_DIR = f'{ROOT}/data/dolphot/sn2026sqf/jhat'
TARGET_FILE = f'{ROOT}/data/dolphot/sn2026sqf/sn2026sqf_pipeline5as_photometry.txt'
TARGET_UJY_FILE = f'{ROOT}/data/dolphot/sn2026sqf/sn2026sqf_pipeline5as_photometry_ujy.txt'
OLD_STARBUG_FILE = f'{ROOT}/data/dolphot/sn2026sqf/sn2026sqf_pipeline5as_photometry_starbug.txt'
NIRCAM_IMAGE = f'{ROOT}/data/dolphot/sn2026sqf/f150w2_i2d.fits'
DOLPHOT_CATS = f'{ROOT}/data/dolphot/sn2026sqf/proc/sn2026sqf_0_0_*.csv'   # NIRCam dolphot catalogues
NEIGHBOUR_MAG = 26.0     # F430M limit for neighbours marked in the identity plot
OUT = f'{ROOT}/data/dolphot/sn2026sqf/starbug'
PSF_DIR = f'{OUT}/psf'
CRDS_APCORR = os.path.expanduser('~/data/crds_cache/references/jwst/miri/jwst_miri_apcorr_0014.fits')
STPSF_DATA = os.path.expanduser('~/data/stpsf-dl/stpsf-data')    # stpsf 2.2 needs data >= 2.1
OBS_DATE = '2024-04-12T00:00:00'

FILTERS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
BOX_SIZES = [2, 5, 10, 15, 25]
BGDR_FACTORS = [2, 3]    # BGD_R in units of FWHM
BGDR_PRIMARY = 2
EE_LIST = [0.3, 0.4, 0.5, 0.6, 0.7]
EE_PRIMARY = 0.5
# name: (FORCE_POS, MAX_XYDEV)
PSF_MODES = {'forced': (1, '1000p'), 'xy05': (0, '0.5p'), 'xy10': (0, '1p'), 'xy15': (0, '1.5p'),
             'xy20': (0, '2p'), 'free': (0, '1000p')}
INJ_MODES = ['forced', 'xy10', 'free']
N_INJ, N_REAL = 8, 2
INJ_FLUX_DEFAULT = 25.0  # uJy, for bands without a dolphot flux
PSF_FOV = 101          # stpsf stamp (pix); fit region is set separately by fit_size()
R_FIT = 60             # fit every listed source within this many pix of the target
N_ZP = 25              # bright isolated stars per frame for the PSF zero-point check
CUT_HALF = 50          # cutouts saved for every map
NCORES = 9

os.environ['STARBUG_DATDIR'] = PSF_DIR      # read by starbug2 at import
os.environ['STPSF_PATH'] = STPSF_DATA

warnings.filterwarnings('ignore')


def load_target():
    from astropy.io import ascii
    t = ascii.read(TARGET_FILE)
    return float(t['ra'][0]), float(t['dec'][0])


def injection_flux(filt):
    from astropy.io import ascii
    t = ascii.read(TARGET_UJY_FILE)
    t = t[t['filter'] == filt]
    try:
        return float(t['flux_ujy'][0])
    except (ValueError, IndexError):
        return INJ_FLUX_DEFAULT


def fwhm_pix(filt):
    import starbug2
    return starbug2.filters[filt].pFWHM


def box_primary(filt):
    '''Default map (full-frame products kept); the photometry uses the injection-selected map.'''
    return min(BOX_SIZES, key=lambda b: abs(b - 2 * fwhm_pix(filt)))


def fit_size(filt):
    n = max(11, int(np.ceil(4 * fwhm_pix(filt))))
    return n if n % 2 else n + 1


def tag_ee(e):
    return f'ee{int(round(100 * e))}'


# ---------------------------------------------------------------------------------------------
# PSFs and aperture corrections
# ---------------------------------------------------------------------------------------------
def make_psf(args):
    filt, detpos = args
    fname = f'{PSF_DIR}/{filt}.fits'          # starbug's default name for MIRIMAGE
    if os.path.exists(fname):
        return fname
    import stpsf
    from astropy.io import fits
    m = stpsf.MIRI()
    m.filter = filt
    m.detector_position = detpos
    try:
        m.load_wss_opd_by_date(OBS_DATE, choice='closest', verbose=False, plot=False)
        opd = 'by_date'
    except Exception as e:
        print(f'{filt}: dated OPD failed ({e}); using default OPD')
        opd = 'default'
    psf = m.calc_psf(fov_pixels=PSF_FOV, oversample=4)
    det = psf['DET_DIST']
    hdr = det.header
    hdr['PSFFRAC'] = (float(det.data.sum()), 'stamp sum of a unit-total PSF')
    hdr['OPDSRC'] = opd
    hdr['DETPOS'] = str(detpos)
    fits.PrimaryHDU(data=det.data, header=hdr).writeto(fname, overwrite=True)
    # oversampled, for encircled energy
    over = psf['OVERDIST']
    fits.PrimaryHDU(data=over.data, header=over.header).writeto(f'{PSF_DIR}/{filt}_over4.fits', overwrite=True)
    return fname


def encircled_energy(filt):
    '''EE(r) of the unit-total stpsf PSF, r in detector pixels.'''
    from astropy.io import fits
    from photutils.aperture import CircularAperture, aperture_photometry
    over = fits.getdata(f'{PSF_DIR}/{filt}_over4.fits')
    c = (over.shape[1] - 1) / 2, (over.shape[0] - 1) / 2
    radii = np.arange(0.25, 20.01, 0.25)
    sums = [aperture_photometry(over, CircularAperture(c, 4 * r), method='exact')['aperture_sum'][0] for r in radii]
    return radii, np.array(sums)


def crds_apcorr(filt):
    from astropy.table import Table
    t = Table.read(CRDS_APCORR)
    t = t[(t['filter'] == filt) & (t['subarray'] == 'FULL')]
    t.sort('eefraction')
    return t


def aperture_setup(filt):
    '''Per EE fraction: radius, CRDS apcorr, stpsf apcorr (with and without sky-annulus PSF wings).'''
    t = crds_apcorr(filt)
    rin, rout = float(t['skyin'][0]), float(t['skyout'][0])
    r_ee, ee = encircled_energy(filt)
    ee_at = lambda r: np.interp(r, r_ee, ee)
    rows = []
    for f in EE_LIST:
        r = float(np.interp(f, t['eefraction'], t['radius']))
        ann_frac = (r ** 2 / (rout ** 2 - rin ** 2)) * (ee_at(rout) - ee_at(rin))
        rows.append(dict(ee=f, r=r, rin=rin, rout=rout,
                         apcorr_crds=float(np.interp(f, t['eefraction'], t['apcorr'])),
                         apcorr_psf=1 / (ee_at(r) - ann_frac), apcorr_psf_nosky=1 / ee_at(r)))
    return rows


# ---------------------------------------------------------------------------------------------
# Per-frame starbug run
# ---------------------------------------------------------------------------------------------
def jy_scale(hdul):
    '''Jy per pixel per (MJy/sr), per pixel.'''
    from starbug2.utils import get_MJysr2Jy_scalefactor
    return get_MJysr2Jy_scalefactor(hdul['SCI']) * hdul['AREA'].data


def aperture_target(img_jy, err_jy, x, y, ap_rows, prefix='ap', extra=True):
    '''
    Aperture photometry with the CRDS annulus as sky, on whatever image is passed (background-subtracted for
    prefix 'ap', the cal frame for 'ap_rawann'). extra adds the CRDS-apcorr and map-only (nosky) columns.
    '''
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from astropy.stats import sigma_clipped_stats
    out = {}
    rin, rout = ap_rows[0]['rin'], ap_rows[0]['rout']
    sky_pix = CircularAnnulus((x, y), rin, rout).to_mask(method='center').get_values(img_jy)
    sky_pix = sky_pix[np.isfinite(sky_pix)]
    _, sky, sky_std = sigma_clipped_stats(sky_pix, sigma=3)
    out[f'{prefix}_sky_ujy_pix'] = sky * 1e6
    for row in ap_rows:
        ap = CircularAperture((x, y), row['r'])
        ph = aperture_photometry(img_jy, ap, error=err_jy, method='exact')
        raw, eraw = float(ph['aperture_sum'][0]), float(ph['aperture_sum_err'][0])
        net = raw - sky * ap.area
        enet = np.sqrt(eraw ** 2 + ap.area * sky_std ** 2 + (ap.area * sky_std) ** 2 / len(sky_pix))
        tag = tag_ee(row['ee'])
        out[f'{prefix}_{tag}_ujy'] = 1e6 * net * row['apcorr_psf']
        out[f'{prefix}_{tag}_err_ujy'] = 1e6 * enet * row['apcorr_psf']
        if extra:
            out[f'{prefix}_{tag}_crds_ujy'] = 1e6 * net * row['apcorr_crds']
            out[f'{prefix}_{tag}_nosky_ujy'] = 1e6 * raw * row['apcorr_psf_nosky']
    return out


def zp_ratios(img_jy, psfcat, zp_xy, ap_rows, psffrac, filt):
    '''Aperture (EE_PRIMARY, stpsf and CRDS apcorr) / PSF flux for the bright isolated stars at zp_xy.'''
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from astropy.stats import sigma_clipped_stats
    row = [r for r in ap_rows if r['ee'] == EE_PRIMARY][0]
    out = []
    for x0, y0 in zp_xy:
        s = psfcat[int(np.argmin(np.hypot(psfcat['x_init'] - x0, psfcat['y_init'] - y0)))]
        x, y = s['x_fit'], s['y_fit']
        ann = CircularAnnulus((x, y), row['rin'], row['rout']).to_mask(method='center').get_values(img_jy)
        _, sky, _ = sigma_clipped_stats(ann[np.isfinite(ann)], sigma=3)
        ap = CircularAperture((x, y), row['r'])
        net = aperture_photometry(img_jy, ap, method='exact')['aperture_sum'][0] - sky * ap.area
        psf = s['flux'] / psffrac
        if psf > 0 and s['flux'] / s['eflux'] > 20:
            out.append(dict(filter=filt, x=x, y=y, psf_ujy=1e6 * psf, ap_ujy=1e6 * net * row['apcorr_psf'],
                            ratio=net * row['apcorr_psf'] / psf, ratio_crds=net * row['apcorr_crds'] / psf))
    return out


def add_sources(det, xs, ys, wcs, flag_col):
    '''Append known positions (target, injections) to a starbug detection table.'''
    from astropy.table import vstack
    new = det[:len(xs)].copy() if len(det) >= len(xs) else vstack([det[:1]] * len(xs))
    for c in new.colnames:
        if new[c].dtype.kind == 'f':
            new[c] = np.nan
        elif new[c].dtype.kind == 'b':
            new[c] = False
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    new['xcentroid'], new['ycentroid'], new['RA'], new['DEC'] = xs, ys, ra, dec
    new['flag'] = 0
    new[flag_col] = True
    return vstack([det, new])


def prepare(fname, ra_t, dec_t, base_opts):
    '''Blind detection plus the target at (ra_t, dec_t), replacing a detection within 1 FWHM.'''
    from astropy.wcs import WCS
    from starbug2.starbug import StarbugBase
    det_dir = f'{OUT}/det'
    os.makedirs(det_dir, exist_ok=True)
    sb = StarbugBase(fname, options=dict(base_opts, OUTPUT=det_dir + '/'))
    sb.detect()
    det = sb.detections
    for c in ('is_target', 'is_inj', 'zp_star'):
        det[c] = False
    w = WCS(sb.image.header)
    xt, yt = [float(v) for v in w.all_world2pix(ra_t, dec_t, 0)]
    d = np.hypot(det['xcentroid'] - xt, det['ycentroid'] - yt)
    info = dict(target_detected=bool(len(det) and d.min() < base_opts['FWHM']), n_detect=len(det))
    info['det_sep_pix'] = float(d.min()) if info['target_detected'] else np.nan
    if info['target_detected']:
        det.remove_row(int(np.argmin(d)))
    det = add_sources(det, [xt], [yt], w, 'is_target')
    return det, xt, yt, w, info


def psf_fit(sb, fitlist, mode, residual=None):
    '''starbug photometry() on the already background-subtracted image in one of PSF_MODES.'''
    force, maxdev = PSF_MODES[mode]
    sb.options.update(FORCE_POS=force, MAX_XYDEV=maxdev, GEN_RESIDUAL=int(residual is not None), QUIETMODE=1)
    sb.detections = fitlist.copy()
    sb.photometry()
    if residual:
        shutil.move(f'{os.path.dirname(residual)}/{sb.bname}-res.fits', residual)
    return sb.psfcatalogue


def psf_vals(cat, x, y, psffrac, prefix):
    import starbug2
    s = cat[int(np.argmin(np.hypot(cat['x_init'] - x, cat['y_init'] - y)))]
    return {f'{prefix}_ujy': 1e6 * s['flux'] / psffrac, f'{prefix}_err_ujy': 1e6 * s['eflux'] / psffrac,
            f'{prefix}_xydev': float(s['xydev']), f'{prefix}_refit': bool(int(s['flag']) & starbug2.SRC_FIX),
            f'{prefix}_x': float(s['x_fit']), f'{prefix}_y': float(s['y_fit']),
            f'{prefix}_ra': float(s['RA']), f'{prefix}_dec': float(s['DEC'])}


def subtract_background(fname, det, opts, add_mjysr=None):
    '''Load a frame (plus injected sources), build the starbug map, subtract it; returns sb and the cal image.'''
    from astropy.io import fits
    from starbug2.starbug import StarbugBase
    sb = StarbugBase(fname, options=opts)
    if add_mjysr is not None:
        sb.image.data = sb.image.data + add_mjysr
    raw = sb.image.data.copy()
    sb.detections = det.copy()
    sb.bgd_estimate()
    sb.bgd_subtraction()
    odir = opts['OUTPUT'].rstrip('/')
    shutil.move(f'{odir}/{sb.bname}-res.fits', f'{odir}/{sb.bname}-bgdsub.fits')
    sb.background = fits.ImageHDU(data=np.zeros_like(sb.image.data))
    return sb, raw


def injection_setup(fname, det, xt, yt, filt, w, psffrac, ap_rows, rng):
    '''N_REAL sets of N_INJ positions 4-10 FWHM from the target, clear of detections and of each other.'''
    from astropy.io import fits
    from photutils.psf import FittableImageModel
    fw = fwhm_pix(filt)
    r_lo, r_hi = 4 * fw, max(45, 10 * fw)
    sep_min = max(5 * fw, ap_rows[0]['rout'] + 3 * fw)
    f_in = injection_flux(filt)
    with fits.open(fname) as h:
        shape, scale = h['SCI'].data.shape, jy_scale(h)
    model = FittableImageModel(fits.getdata(f'{PSF_DIR}/{filt}.fits'), normalize=True)
    half = PSF_FOV // 2
    sets = []
    for real in range(N_REAL):
        pts = []
        for _ in range(5000):
            r, th = rng.uniform(r_lo, r_hi), rng.uniform(0, 2 * np.pi)
            x, y = xt + r * np.cos(th), yt + r * np.sin(th)
            if not (400 < x < shape[1] - 40 and 40 < y < shape[0] - 40):
                continue
            if np.min(np.hypot(det['xcentroid'] - x, det['ycentroid'] - y)) < 3 * fw:
                continue
            if pts and np.min([np.hypot(x - a, y - b) for a, b in pts]) < sep_min:
                continue
            pts.append((x, y))
            if len(pts) == N_INJ:
                break
        add = np.zeros(shape, np.float32)
        for x, y in pts:
            model.flux, model.x_0, model.y_0 = f_in * 1e-6 * psffrac, x, y
            x0, y0 = int(round(x)), int(round(y))
            sl = np.s_[max(y0 - half, 0):min(y0 + half + 1, shape[0]), max(x0 - half, 0):min(x0 + half + 1, shape[1])]
            yy, xx = np.mgrid[sl]
            add[sl] += model(xx, yy) / scale[sl]            # Jy/pix -> MJy/sr
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        det_i = add_sources(det, xs, ys, w, 'is_inj')
        near = np.hypot(det_i['xcentroid'] - xt, det_i['ycentroid'] - yt) < r_hi + 2 * fit_size(filt)
        sets.append(dict(pts=pts, add=add, det=det_i, fitlist=det_i[near | det_i['is_inj']], f_in=f_in))
    return sets


def run_frame(args):
    fname, ra_t, dec_t, locate = args
    from astropy.io import fits
    filt = fits.getheader(fname)['FILTER']
    fw = fwhm_pix(filt)
    bname = os.path.basename(fname).replace('.fits', '')
    psffrac = fits.getheader(f'{PSF_DIR}/{filt}.fits')['PSFFRAC']
    ap_rows = aperture_setup(filt)
    base_opts = dict(VERBOSE=0, FWHM=fw, SIGSRC=5.0, PSF_SIZE=fit_size(filt), CALC_CROWD=0, APCORR_FILE=CRDS_APCORR)

    det, xt, yt, w, info = prepare(fname, ra_t, dec_t, base_opts)
    near = np.hypot(det['xcentroid'] - xt, det['ycentroid'] - yt) < R_FIT
    fitlist = det[near]

    if locate:   # pass 1: free fit on the primary map only
        odir = f'{OUT}/locate'
        os.makedirs(odir, exist_ok=True)
        opts = dict(base_opts, BOX_SIZE=box_primary(filt), BGD_R=BGDR_PRIMARY * fw, OUTPUT=odir + '/')
        sb, _ = subtract_background(fname, det, opts)
        v = psf_vals(psf_fit(sb, fitlist, 'free'), xt, yt, psffrac, 'psf_free')
        return [dict(file=bname, filter=filt, xydev_pix=v['psf_free_xydev'], ra_fit=v['psf_free_ra'],
                     dec_fit=v['psf_free_dec'], psf_ujy=v['psf_free_ujy'])], [], []

    # bright, isolated, away from edges and the coronagraph/Lyot columns, for the PSF zero-point check
    dd = np.hypot(det['xcentroid'][:, None] - det['xcentroid'][None, :], det['ycentroid'][:, None] - det['ycentroid'][None, :])
    np.fill_diagonal(dd, np.inf)
    snr = np.asarray(det['flux'] / det['eflux'], dtype=float)
    ok = ((dd.min(axis=1) > 6 * fw) & (det['xcentroid'] > 400) & (det['xcentroid'] < 1000)
          & (det['ycentroid'] > 30) & (det['ycentroid'] < 994) & (snr > 20) & ~det['is_target'])
    zp_idx = np.where(ok)[0][np.argsort(-snr[ok])][:N_ZP]
    zp_xy = list(zip(det['xcentroid'][zp_idx], det['ycentroid'][zp_idx]))
    fitlist_zp = det[near | np.isin(np.arange(len(det)), zp_idx)]

    rng = np.random.default_rng(zlib.crc32(bname.encode()))
    inj = injection_setup(fname, det, xt, yt, filt, w, psffrac, ap_rows, rng)
    with fits.open(fname) as h:
        scale = jy_scale(h)
        err_jy = h['ERR'].data * scale
    tmp = f'{OUT}/inj_tmp'
    os.makedirs(tmp, exist_ok=True)

    results, zps, injections, cutouts = [], [], [], dict(origin=np.array([int(round(xt)) - CUT_HALF, int(round(yt)) - CUT_HALF]))
    for box in BOX_SIZES:
        for bf in BGDR_FACTORS:
            primary = box == box_primary(filt) and bf == BGDR_PRIMARY
            odir = f'{OUT}/box{box}_r{bf}'
            os.makedirs(odir, exist_ok=True)
            opts = dict(base_opts, BOX_SIZE=box, BGD_R=bf * fw, OUTPUT=odir + '/', QUIETMODE=0)
            sb, raw = subtract_background(fname, det, opts)
            res = dict(file=bname, filter=filt, box=box, bgdr=bf, primary=primary, x_t=xt, y_t=yt, psffrac=psffrac,
                       bgd_at_target_mjysr=float(raw[int(round(yt)), int(round(xt))] - sb.image.data[int(round(yt)), int(round(xt))]))
            res.update(info)
            for mode in PSF_MODES:
                resid = f'{odir}/{bname}-res_{mode}.fits' if mode in ('free', 'forced') else None
                cat = psf_fit(sb, fitlist_zp if (primary and mode == 'free') else fitlist, mode, resid)
                res.update(psf_vals(cat, xt, yt, psffrac, f'psf_{mode}'))
                if primary and mode == 'free':
                    zps = zp_ratios(sb.image.data * scale, cat, zp_xy, ap_rows, psffrac, filt)
                    for z in zps:
                        z['file'] = bname
            # free-fit summary columns used by the astrometry and identity checks
            res.update(xydev_pix=res['psf_free_xydev'], ra_fit=res['psf_free_ra'], dec_fit=res['psf_free_dec'])
            res.update(aperture_target(sb.image.data * scale, err_jy, xt, yt, ap_rows, 'ap'))
            res.update(aperture_target(raw * scale, err_jy, xt, yt, ap_rows, 'ap_rawann', extra=False))
            results.append(res)
            yc, xc = int(round(yt)), int(round(xt))
            sl = np.s_[yc - CUT_HALF:yc + CUT_HALF + 1, xc - CUT_HALF:xc + CUT_HALF + 1]
            cutouts['raw'] = raw[sl]
            for suffix in ('bgd', 'bgdsub', 'res_free', 'res_forced'):
                cutouts[f'{box}_{bf}_{suffix}'] = fits.getdata(f'{odir}/{bname}-{suffix}.fits')[sl]
                if not primary:
                    os.remove(f'{odir}/{bname}-{suffix}.fits')

            # injection-recovery through the same map settings
            for real, s in enumerate(inj):
                sbi, raw_i = subtract_background(fname, s['det'], dict(opts, OUTPUT=tmp + '/', QUIETMODE=1), s['add'])
                cats = {m: psf_fit(sbi, s['fitlist'], m) for m in INJ_MODES}
                for k, (x, y) in enumerate(s['pts']):
                    r = dict(file=bname, filter=filt, box=box, bgdr=bf, primary=primary, real=real, k=k, x=x, y=y,
                             f_in=s['f_in'], r_target_pix=float(np.hypot(x - xt, y - yt)))
                    for m in INJ_MODES:
                        v = psf_vals(cats[m], x, y, psffrac, f'psf_{m}')
                        r.update({kk: v[kk] for kk in (f'psf_{m}_ujy', f'psf_{m}_err_ujy', f'psf_{m}_xydev', f'psf_{m}_refit')})
                    r.update(aperture_target(sbi.image.data * scale, err_jy, x, y, ap_rows, 'ap'))
                    r.update(aperture_target(raw_i * scale, err_jy, x, y, ap_rows, 'ap_rawann', extra=False))
                    injections.append(r)
    for f in glob.glob(f'{tmp}/{bname}-*'):
        os.remove(f)
    os.makedirs(f'{OUT}/cutouts', exist_ok=True)
    np.savez_compressed(f'{OUT}/cutouts/{bname}.npz', **cutouts)
    return results, zps, injections


# ---------------------------------------------------------------------------------------------
# Combine dithers
# ---------------------------------------------------------------------------------------------
METHODS = ['psf_forced', 'psf_xy05', 'psf_xy10', 'psf_xy15', 'psf_xy20', 'psf_free', 'ap', 'ap_rawann']
METHOD_LABEL = {'psf_forced': 'PSF forced', 'psf_xy05': 'PSF MAX_XYDEV 0.5 px', 'psf_xy10': 'PSF MAX_XYDEV 1 px',
                'psf_xy15': 'PSF MAX_XYDEV 1.5 px', 'psf_xy20': 'PSF MAX_XYDEV 2 px', 'psf_free': 'PSF free',
                'ap': f'aperture EE{int(100 * EE_PRIMARY)}, bgd-subtracted', 'ap_rawann': f'aperture EE{int(100 * EE_PRIMARY)}, plain annulus'}


def wmean(f, e):
    f, e = np.asarray(f, float), np.asarray(e, float)
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    f, e = f[good], e[good]
    if not len(f):
        return np.nan, np.nan
    w = 1 / e ** 2
    m = np.sum(w * f) / np.sum(w)
    err = max(np.sqrt(1 / np.sum(w)), np.std(f, ddof=1) / np.sqrt(len(f)) if len(f) > 1 else 0)
    return m, err


def psf_good(g):
    '''Free-centroid fits that stayed within one HWHM of the target position.'''
    return g['xydev_pix'] <= 0.5 * g['filter'].map(fwhm_pix)


def value_cols(m):
    if m.startswith('psf'):
        return f'{m}_ujy', f'{m}_err_ujy'
    t = tag_ee(EE_PRIMARY)
    return f'{m}_{t}_ujy', f'{m}_{t}_err_ujy'


def method_mean(g, m):
    fc, ec = value_cols(m)
    ok = psf_good(g) if m == 'psf_free' else np.ones(len(g), bool)
    return wmean(g[fc][ok], g[ec][ok])


def astrometry_check():
    '''
    Ties the MIRI frames to the NIRCam dolphot catalogue (DOLPHOT_CATS):
      frame offset   median MIRI - NIRCam position of the bright isolated (zero-point) stars
      nircam         the dolphot source nearest the catalogue position (the NIRCam counterpart)
      neighbours     dolphot sources within 1.2" with F430M < NEIGHBOUR_MAG
    Offsets are (dRA cos Dec, dDec) in mas relative to the TARGET_FILE position.
    '''
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra0, dec0 = load_target()
    off = lambda ra, dec: ((np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3.6e6, (np.asarray(dec) - dec0) * 3.6e6)
    cat = pd.concat([pd.read_csv(f, usecols=['RA', 'Dec', 'F200W_mag', 'F430M_mag', 'F430M_err'])
                     for f in sorted(glob.glob(DOLPHOT_CATS))])
    cat['dra'], cat['ddec'] = off(cat['RA'], cat['Dec'])
    cat['sep'] = np.hypot(cat['dra'], cat['ddec'])
    near = cat[cat['sep'] < 1200].sort_values('sep')
    nircam = near.iloc[0]
    neighbours = near[(near['F430M_mag'] < NEIGHBOUR_MAG) & (near.index != near.index[0])]
    # frame offset from the zero-point stars
    zp = pd.read_csv(f'{OUT}/sn2026sqf_miri_starbug_zp.csv')
    ref = cat[(cat['F430M_mag'] < 24.5) & (cat['F430M_err'] < 0.1)]
    rc = SkyCoord(ref['RA'].values, ref['Dec'].values, unit='deg')
    d = []
    for f, g in zp.groupby('file'):
        ra, dec = WCS(fits.getheader(f'{JHAT_DIR}/{f}.fits', 'SCI')).all_pix2world(g['x'], g['y'], 0)
        i, sep, _ = SkyCoord(ra, dec, unit='deg').match_to_catalog_sky(rc)
        m = sep < 0.15 * u.arcsec
        d += list(zip((ra[m] - ref['RA'].values[i[m]]) * np.cos(np.radians(dec0)) * 3.6e6,
                      (dec[m] - ref['Dec'].values[i[m]]) * 3.6e6))
    d = np.array(d)
    frame = np.median(d, axis=0)
    frame_err = 1.2533 * np.std(d, axis=0) / np.sqrt(len(d))
    return dict(frame=frame, frame_err=frame_err, n_frame=len(d), nircam=nircam, neighbours=neighbours)


def select_maps(summ):
    '''Per band, the (box, bgdr) whose injections are least biased in forced PSF and aperture photometry.'''
    sel = {}
    for filt, s in summ.groupby('filter'):
        cost = (s['psf_forced_ratio'] - 1).abs() + (s['ap_ratio'] - 1).abs()
        best = s.loc[cost.idxmin()]
        sel[filt] = (int(best['box']), int(best['bgdr']))
    return sel


def combine(frames, astro, summ):
    import pandas as pd
    rows = []
    for filt in FILTERS:
        fr = frames[frames['filter'] == filt]
        g = fr[fr['primary']]
        b0, bf0 = int(g['box'].iloc[0]), int(g['bgdr'].iloc[0])
        si = summ[(summ['filter'] == filt) & (summ['box'] == b0) & (summ['bgdr'] == bf0)].iloc[0]
        row = dict(filter=filt, box=b0, bgdr=bf0, n_frames=len(g), n_target_detected=int(g['target_detected'].sum()),
                   psf_free_n_used=int(psf_good(g).sum()))
        for m in METHODS:
            per_map = [method_mean(gg, m)[0] for _, gg in fr.groupby(['box', 'bgdr'])]
            f, e = method_mean(g, m)
            row[f'{m}_ujy'], row[f'{m}_err_ujy'], row[f'{m}_sys_ujy'] = f, e, np.nanstd(per_map)
            # background-structure error from the injections (PSF MAX_XYDEV modes use the forced value)
            mi = m if m in ('psf_forced', 'psf_xy10', 'psf_free', 'ap', 'ap_rawann') else 'psf_forced'
            row[f'{m}_inj_ratio'] = si[f'{mi}_ratio']
            row[f'{m}_bgd_ujy'] = si[f'{mi}_bgdfrac'] * si['f_in']     # additive, so scaled by the injected flux
            if m.startswith('psf_xy'):
                row[f'{m}_n_refit'] = int(g[f'{m}_refit'].sum())
            et = np.sqrt(e ** 2 + row[f'{m}_sys_ujy'] ** 2 + row[f'{m}_bgd_ujy'] ** 2)
            row[f'{m}_tot_err_ujy'] = et
            row[f'{m}_snr'] = f / et
            row[f'{m}_mag'] = -2.5 * np.log10(f * 1e-6 / 3631) if f > 0 else np.nan
            row[f'{m}_mag_err'] = 1.0857 * et / f if f > 0 else np.nan
            # detection on statistical (+) map-sweep error; the background-structure term is flux scatter of a real source
            row[f'{m}_det_snr'] = f / np.hypot(e, row[f'{m}_sys_ujy'])
            row[f'{m}_ul_ujy'] = max(f, 0) + 3 * et if row[f'{m}_det_snr'] < 3 else np.nan
        row['psf_free_reliable'] = row['psf_free_n_used'] >= 2
        t = tag_ee(EE_PRIMARY)
        row['ap_crds_ujy'] = wmean(g[f'ap_{t}_crds_ujy'], g[f'ap_{t}_err_ujy'])[0]
        row['ap_nosky_ujy'] = wmean(g[f'ap_{t}_nosky_ujy'], g[f'ap_{t}_err_ujy'])[0]
        for p in ('ap', 'ap_rawann'):
            for e in EE_LIST:
                row[f'{p}_{tag_ee(e)}_ujy'] = wmean(g[f'{p}_{tag_ee(e)}_ujy'], g[f'{p}_{t}_err_ujy'])[0]
        # free-PSF centroid relative to the NIRCam counterpart, MIRI frame offset removed
        ok = psf_good(g)
        dra = (g['ra_fit'][ok] - load_target()[0]) * np.cos(np.radians(load_target()[1])) * 3.6e6 - astro['frame'][0] - astro['nircam']['dra']
        ddec = (g['dec_fit'][ok] - load_target()[1]) * 3.6e6 - astro['frame'][1] - astro['nircam']['ddec']
        row.update(psf_free_dra_nircam_mas=float(np.mean(dra)) if ok.any() else np.nan,
                   psf_free_ddec_nircam_mas=float(np.mean(ddec)) if ok.any() else np.nan,
                   psf_free_pos_err_mas=float(np.sqrt((np.var(dra, ddof=1) + np.var(ddec, ddof=1)) / 2 / ok.sum())) if ok.sum() > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def injection_summary(inj):
    '''Recovered / injected flux per band, map and method: median, robust scatter, N; plus the injected curve of growth.'''
    import pandas as pd
    rows = []
    for (filt, box, bf), g in inj.groupby(['filter', 'box', 'bgdr']):
        row = dict(filter=filt, box=box, bgdr=bf, primary=bool(g['primary'].iloc[0]), n=len(g), f_in=g['f_in'].iloc[0])
        for m in ['psf_forced', 'psf_xy10', 'psf_free', 'ap', 'ap_rawann']:
            fc, _ = value_cols(m)
            ratio = g[fc] / g['f_in']
            if m == 'psf_free':
                good = g['psf_free_xydev'] <= 0.5 * fwhm_pix(filt)
                row['psf_free_frac_wandered'] = 1 - good.mean()
                ratio = ratio[good]
            med = np.nanmedian(ratio)
            row[f'{m}_ratio'] = med
            row[f'{m}_scatter'] = 1.4826 * np.nanmedian(np.abs(ratio - med))
            row[f'{m}_ratio_err'] = 1.2533 * row[f'{m}_scatter'] / np.sqrt(max(np.isfinite(ratio).sum(), 1))
            # scatter beyond the pixel noise: background structure, as a fraction of the flux
            noise = np.nanmedian(g[value_cols(m)[1]] / g['f_in'])
            row[f'{m}_noise'] = noise
            row[f'{m}_bgdfrac'] = np.sqrt(max(row[f'{m}_scatter'] ** 2 - noise ** 2, 0))
        for p in ('ap', 'ap_rawann'):
            for e in EE_LIST:
                r = g[f'{p}_{tag_ee(e)}_ujy'] / g['f_in']
                row[f'{p}_cog_{tag_ee(e)}'] = np.nanmedian(r)
                row[f'{p}_cog_{tag_ee(e)}_lo'], row[f'{p}_cog_{tag_ee(e)}_hi'] = np.nanpercentile(r, [16, 84])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
BLUE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
METHOD_COLOR = {'psf_forced': SERIES[0], 'ap': SERIES[1], 'ap_rawann': SERIES[2], 'psf_free': SERIES[6], 'psf_xy10': SERIES[3]}


def plot_residuals(frames, fname):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from astropy.io import fits
    from astropy.visualization import ImageNormalize, AsinhStretch
    half = 22
    fig, axes = plt.subplots(len(FILTERS), 6, figsize=(15, 2.45 * len(FILTERS)), facecolor=SURF)
    for i, filt in enumerate(FILTERS):
        r = frames[(frames['filter'] == filt) & frames['primary']].sort_values('file').iloc[0]
        npz = np.load(f'{OUT}/cutouts/{r["file"]}.npz')
        x0, y0 = npz['origin']
        xc, yc = x0 + CUT_HALF, y0 + CUT_HALF
        csl = np.s_[CUT_HALF - half:CUT_HALF + half + 1, CUT_HALF - half:CUT_HALF + half + 1]
        pre = f'{int(r["box"])}_{int(r["bgdr"])}_'
        cut = dict(raw=npz['raw'][csl])
        for k, suffix in (('bgd', 'bgd'), ('sub', 'bgdsub'), ('free', 'res_free'), ('forced', 'res_forced')):
            cut[k] = npz[pre + suffix][csl]
        yy, xx = np.mgrid[yc - half:yc + half + 1, xc - half:xc + half + 1]
        rr = np.hypot(xx - r['x_t'], yy - r['y_t'])
        outer = cut['sub'][rr > 4 * fwhm_pix(filt)]
        sig = 1.4826 * np.nanmedian(np.abs(outer - np.nanmedian(outer)))
        lo, hi = np.nanpercentile(cut['raw'], [1, 99.7])
        nrm_raw = ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(0.1))
        nrm_sub = ImageNormalize(vmin=-4 * sig, vmax=12 * sig)
        panels = [('raw', 'cal (MJy/sr)', nrm_raw), ('bgd', 'starbug bgd', nrm_raw), ('sub', 'bgd-subtracted', nrm_sub),
                  ('free', 'residual, free PSF', nrm_sub), ('forced', 'residual, forced PSF', nrm_sub)]
        for j, (k, title, nrm) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(cut[k], origin='lower', norm=nrm, cmap='Greys_r',
                      extent=(xc - half - .5, xc + half + .5, yc - half - .5, yc + half + .5))
            ax.plot(r['x_t'], r['y_t'], '+', color='#eb6834', ms=10, mew=1.5)
            if k in ('sub', 'free'):
                ax.plot(r['psf_free_x'], r['psf_free_y'], 'x', color='#1baf7a', ms=7, mew=1.5)
            if k == 'sub':
                a0 = [a for a in aperture_setup(filt) if a['ee'] == EE_PRIMARY][0]
                ax.add_patch(Circle((r['x_t'], r['y_t']), a0['r'], fill=False, ec='#eb6834', lw=1))
                for rad in (a0['rin'], min(a0['rout'], half)):
                    ax.add_patch(Circle((r['x_t'], r['y_t']), rad, fill=False, ec='#eb6834', lw=0.7, ls='--'))
            ax.set_xlim(xc - half - .5, xc + half + .5); ax.set_ylim(yc - half - .5, yc + half + .5)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9, color=INK)
        axes[i, 0].set_ylabel(f'{filt}\nbox {int(r["box"])}, BGD_R {int(r["bgdr"])} FWHM', fontsize=9, color=INK)
        # radial profile about the F560W position, in units of the local pixel noise
        ax = axes[i, 5]
        rb = np.arange(0, 3 * fwhm_pix(filt) + 1, max(1, fwhm_pix(filt) / 3))
        for k, col, lab in (('sub', SERIES[0], 'bgd-sub'), ('free', SERIES[6], 'res. free'), ('forced', SERIES[1], 'res. forced')):
            prof = [np.nanmean(cut[k][(rr >= a) & (rr < b)]) / sig for a, b in zip(rb[:-1], rb[1:])]
            ax.plot(0.5 * (rb[1:] + rb[:-1]), prof, '-o', color=col, ms=3, lw=2, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_xlabel('r from F560W position (pix)', fontsize=7, color=INK2)
        ax.set_ylabel('mean / pixel σ', fontsize=7, color=INK2)
        ax.text(0.97, 0.95, f'forced {r["psf_forced_ujy"]:.2f}±{r["psf_forced_err_ujy"]:.2f} µJy\n'
                            f'free {r["psf_free_ujy"]:.2f} µJy, xydev {r["psf_free_xydev"]:.2f} px',
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color=INK)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc='center right')
            ax.set_title('radial profile', fontsize=9, color=INK)
    fig.suptitle('SN 2026sqf MIRI (dither 1 per band, injection-selected map): + F560W position, × free PSF centroid; '
                 f'solid circle EE{int(100 * EE_PRIMARY)} aperture, dashed CRDS sky annulus', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fname, dpi=120, facecolor=SURF)
    plt.close(fig)


def plot_identity(frames, comb, astro, fname):
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.visualization import ImageNormalize, AsinhStretch
    ra_t, dec_t = load_target()
    c0 = SkyCoord(ra_t, dec_t, unit='deg')
    nc, nb = astro['nircam'], astro['neighbours']
    fig = plt.figure(figsize=(13, 4.6), facecolor=SURF)

    def miri_offsets(g):
        '''free-PSF centroids (good fits) relative to the NIRCam counterpart, MIRI frame offset removed (mas)'''
        g = g[psf_good(g)]
        dra = (g['ra_fit'] - ra_t) * np.cos(np.radians(dec_t)) * 3.6e6 - astro['frame'][0] - nc['dra']
        ddec = (g['dec_fit'] - dec_t) * 3.6e6 - astro['frame'][1] - nc['ddec']
        return dra.values, ddec.values

    # (a) NIRCam F150W2 with the counterpart, bright neighbours and the MIRI PSF size
    with fits.open(NIRCAM_IMAGE, memmap=True) as h:
        cut = Cutout2D(h['SCI'].data, c0, 3.2 * u.arcsec, wcs=WCS(h['SCI'].header))
    # plain axes in cutout pixels (WCSAxes is broken by the matplotlib/astropy pair in this env)
    ax = fig.add_subplot(1, 3, 1)
    lo, hi = np.nanpercentile(cut.data, [5, 99.8])
    ax.imshow(cut.data, origin='lower', cmap='Greys_r', norm=ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(0.05)))
    pscale = cut.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    to_pix = lambda dra, ddec: cut.wcs.world_to_pixel(SkyCoord(ra_t + np.asarray(dra) / 3.6e6 / np.cos(np.radians(dec_t)),
                                                               dec_t + np.asarray(ddec) / 3.6e6, unit='deg'))
    px, py = to_pix(nc['dra'], nc['ddec'])
    for filt, col in (('F560W', SERIES[0]), ('F1800W', SERIES[2])):
        ax.add_patch(plt.Circle((px, py), 0.5 * fwhm_pix(filt) * 0.11 / pscale, fill=False, ec=col, lw=1.2, ls='--',
                                label=f'{filt} HWHM'))
    ax.scatter([px], [py], marker='+', s=160, color='#eb6834', lw=2, zorder=4, label=f'NIRCam counterpart (F430M {nc["F430M_mag"]:.2f})')
    qx, qy = to_pix(nb['dra'], nb['ddec'])
    ax.scatter(qx, qy, s=60, facecolor='none', edgecolor='#eda100', lw=1.3, zorder=3, label=f'neighbours, F430M < {NEIGHBOUR_MAG:g}')
    for x, y, m in zip(qx, qy, nb['F430M_mag']):
        ax.text(x + 3, y + 3, f'{m:.1f}', color='#eda100', fontsize=7)
    for k, filt in enumerate(FILTERS):
        dra, ddec = miri_offsets(frames[(frames['filter'] == filt) & frames['primary']])
        if len(dra):
            mx, my = to_pix(np.mean(dra) + nc['dra'], np.mean(ddec) + nc['ddec'])
            ax.scatter([mx], [my], s=18, color=BLUE_RAMP[k], edgecolor=SURF, lw=0.5, zorder=5)
    # compass, 0.4" arms
    ny, nx = cut.data.shape
    o = np.array([0.85 * nx, 0.82 * ny])
    oc = cut.wcs.pixel_to_world(*o)
    for dpa, lab in ((0, 'N'), (90, 'E')):
        tip = np.array(cut.wcs.world_to_pixel(oc.directional_offset_by(dpa * u.deg, 0.4 * u.arcsec)))
        ax.annotate('', xy=tip, xytext=o, arrowprops=dict(arrowstyle='->', color=SURF, lw=1.2))
        ax.text(*(o + 1.3 * (tip - o)), lab, color=SURF, fontsize=8, ha='center', va='center')
    ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f'NIRCam F150W2, 3.2″ ({pscale:.3f}″/pix); dots: MIRI band-mean centroids', fontsize=7.5, color=INK2)
    ax.set_title('NIRCam counterpart and neighbours', fontsize=9, color=INK)
    ax.legend(fontsize=6.5, loc='lower left', frameon=True, framealpha=0.85)

    # (b) MIRI free-PSF centroid - NIRCam counterpart
    ax = fig.add_subplot(1, 3, 2)
    for k, filt in enumerate(FILTERS):
        dra, ddec = miri_offsets(frames[(frames['filter'] == filt) & frames['primary']])
        if not len(dra):
            continue
        ax.scatter(dra / 1e3, ddec / 1e3, s=12, color=BLUE_RAMP[k], alpha=0.6, lw=0)
        ax.scatter([np.mean(dra) / 1e3], [np.mean(ddec) / 1e3], s=40, color=BLUE_RAMP[k], edgecolor=INK, lw=0.6, zorder=4)
        ax.text(np.mean(dra) / 1e3, np.mean(ddec) / 1e3, f'  {filt} ({len(dra)})', fontsize=6.5, color=INK2, va='center')
    for filt, col in (('F560W', SERIES[0]), ('F1800W', SERIES[2])):
        ax.add_patch(plt.Circle((0, 0), 0.5 * fwhm_pix(filt) * 0.11, fill=False, ec=col, lw=1, ls='--'))
    near = nb[nb['sep'] < 700]
    ax.scatter((near['dra'] - nc['dra']) / 1e3, (near['ddec'] - nc['ddec']) / 1e3, s=60, facecolor='none',
               edgecolor='#eda100', lw=1.3)
    ax.axhline(0, color=INK2, lw=0.5); ax.axvline(0, color=INK2, lw=0.5)
    ax.set_aspect('equal'); lim = 0.75
    ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('ΔRA (″, E left)', fontsize=8, color=INK2); ax.set_ylabel('ΔDec (″)', fontsize=8, color=INK2)
    ax.set_title(f'MIRI free-PSF centroid − NIRCam counterpart\n(frame offset {astro["frame"][0]:+.0f}, {astro["frame"][1]:+.0f} mas removed; '
                 f'fits within HWHM)', fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7, colors=INK2)
    # (c) curve of growth, normalised to EE_PRIMARY
    ax = fig.add_subplot(1, 3, 3)
    for k, filt in enumerate(FILTERS):
        r = comb[comb['filter'] == filt].iloc[0]
        ref = r[f'ap_{tag_ee(EE_PRIMARY)}_ujy']
        y = [r[f'ap_{tag_ee(e)}_ujy'] / ref for e in EE_LIST]
        ax.plot(EE_LIST, y, '-o', color=BLUE_RAMP[k], lw=2, ms=4)
        ax.text(EE_LIST[-1] + 0.01, min(y[-1], 2.45), filt, fontsize=6.5, color=INK2, va='center')
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_xlim(0.27, 0.78)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel('aperture EE fraction', fontsize=8, color=INK2)
    ax.set_ylabel(f'apcorr flux / flux(EE{int(100 * EE_PRIMARY)})', fontsize=8, color=INK2)
    ax.set_title('target curve of growth (flat = point source)', fontsize=9, color=INK)
    ax.tick_params(labelsize=7, colors=INK2)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, facecolor=SURF)
    plt.close(fig)


def dolphot_ujy(filt):
    from astropy.io import ascii
    t = ascii.read(TARGET_UJY_FILE)
    t = t[t['filter'] == filt]
    try:
        return float(t['flux_ujy'][0]), float(t['flux_err_ujy'][0]), str(t['detected'][0]) == 'True'
    except (ValueError, IndexError):
        return np.nan, np.nan, False


def plot_psfmodes(comb, fname):
    '''Target flux per PSF mode (forced -> free) against the aperture value and dolphot.'''
    import matplotlib.pyplot as plt
    modes = ['psf_forced', 'psf_xy05', 'psf_xy10', 'psf_xy15', 'psf_xy20', 'psf_free']
    labels = ['forced', '0.5', '1.0', '1.5', '2.0', 'free']
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), facecolor=SURF)
    for ax, filt in zip(axes.flat, FILTERS):
        r = comb[comb['filter'] == filt].iloc[0]
        a, ea = r['ap_ujy'], r['ap_tot_err_ujy']
        ax.axhspan(a - ea, a + ea, color=SERIES[1], alpha=0.18, lw=0)
        ax.axhline(a, color=SERIES[1], lw=1.5, label=METHOD_LABEL['ap'])
        fd, ed, detd = dolphot_ujy(filt)
        if np.isfinite(fd):
            ax.axhline(fd, color=INK2, lw=1.2, ls='--', label='dolphot' + ('' if detd else ' (not detected)'))
        for k, m in enumerate(modes):
            f, e = r[f'{m}_ujy'], r[f'{m}_tot_err_ujy']
            hollow = m == 'psf_free' and not r['psf_free_reliable']
            ax.errorbar([k], [f], yerr=[e], fmt='D', color=SERIES[0], mfc=SURF if hollow else SERIES[0], ms=6, lw=1.5)
            if m.startswith('psf_xy'):
                ax.text(k, f + e, f'{int(r[f"{m}_n_refit"])}', fontsize=7, color=INK2, ha='center', va='bottom')
        ax.set_xticks(range(len(modes)), labels, fontsize=7.5)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_title(filt, fontsize=9, color=INK)
        if filt == FILTERS[0]:
            ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    for ax in axes[-1]:
        ax.set_xlabel('PSF mode: MAX_XYDEV (pix)', fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel('F$_ν$ (µJy)', fontsize=8, color=INK2)
    fig.suptitle('Target PSF flux vs centroid freedom (number above a point: dithers refit as forced; '
                 'hollow free: < 2 fits within HWHM)', fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, dpi=140, facecolor=SURF)
    plt.close(fig)


def plot_injection(summ, fname):
    '''Recovered / injected flux vs background box size, per band and method.'''
    import matplotlib.pyplot as plt
    methods = ['psf_forced', 'psf_free', 'ap', 'ap_rawann']
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), facecolor=SURF)
    for ax, filt in zip(axes.flat, FILTERS):
        s = summ[summ['filter'] == filt]
        ax.axhline(1, color=INK2, lw=0.6)
        ax.axvline(box_primary(filt), color=INK2, lw=0.6, ls=':')
        for k, m in enumerate(methods):
            for bf, ls, mfc in ((BGDR_PRIMARY, '-', None), (3 if BGDR_PRIMARY == 2 else 2, '--', SURF)):
                g = s[s['bgdr'] == bf].sort_values('box')
                if m == 'ap_rawann' and bf != BGDR_PRIMARY:
                    continue      # independent of the map
                x = g['box'] * (1 + 0.03 * (k - 1.5))
                ax.errorbar(x, g[f'{m}_ratio'], yerr=g[f'{m}_ratio_err'], fmt='o', ls=ls, color=METHOD_COLOR[m],
                            mfc=mfc or METHOD_COLOR[m], ms=4, lw=1.5,
                            label=(METHOD_LABEL[m] + ('' if bf == BGDR_PRIMARY else f', BGD_R {bf} FWHM')) if filt == FILTERS[0] else None)
        sel = s[s['primary']]
        ax.plot(sel['box'], sel['psf_forced_ratio'], '*', color=INK, ms=11, mfc='none', mew=1.2,
                label='selected map' if filt == FILTERS[0] else None)
        ax.set_xscale('log')
        ax.set_xticks(BOX_SIZES, [str(b) for b in BOX_SIZES])
        ax.set_xticks([], minor=True)
        ax.tick_params(labelsize=7, colors=INK2)
        f_in = s['f_in'].iloc[0]
        sc = sel.iloc[0]
        ax.set_title(f'{filt}: {f_in:.1f} µJy injected; σ/F: PSF {sc["psf_forced_scatter"]:.2f}, ap {sc["ap_scatter"]:.2f} '
                     f'(noise {sc["psf_forced_noise"]:.2f})',
                     fontsize=8, color=INK)
        ax.set_ylim(0.5, 1.5)
    for ax in axes[-1]:
        ax.set_xlabel('starbug BOX_SIZE (pix)', fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel('recovered / injected (median)', fontsize=8, color=INK2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc='lower center', ncol=4)
    fig.suptitle(f'Injection-recovery: {N_REAL}×{N_INJ} stpsf sources per frame, 4–10 FWHM from the target '
                 '(error bar: error of the median; dotted: default box)', fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(fname, dpi=140, facecolor=SURF)
    plt.close(fig)


def plot_cog(comb, summ, fname):
    '''Target curve of growth against injected point sources through the same map, both normalised at EE_PRIMARY.'''
    import matplotlib.pyplot as plt
    t0 = tag_ee(EE_PRIMARY)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), facecolor=SURF)
    for ax, filt in zip(axes.flat, FILTERS):
        s = summ[(summ['filter'] == filt) & summ['primary']].iloc[0]
        r = comb[comb['filter'] == filt].iloc[0]
        for p, col, lab in (('ap', SERIES[0], 'bgd-subtracted'), ('ap_rawann', SERIES[2], 'plain annulus')):
            ref = s[f'{p}_cog_{t0}']
            y = np.array([s[f'{p}_cog_{tag_ee(e)}'] for e in EE_LIST]) / ref
            lo = np.array([s[f'{p}_cog_{tag_ee(e)}_lo'] for e in EE_LIST]) / ref
            hi = np.array([s[f'{p}_cog_{tag_ee(e)}_hi'] for e in EE_LIST]) / ref
            ax.fill_between(EE_LIST, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(EE_LIST, y, '--', color=col, lw=1.5, label=f'injected, {lab}')
            yt = np.array([r[f'{p}_{tag_ee(e)}_ujy'] for e in EE_LIST]) / r[f'{p}_{t0}_ujy']
            ax.plot(EE_LIST, yt, '-o', color=col if p == 'ap_rawann' else SERIES[1], lw=2, ms=4, label=f'target, {lab}')
        ax.axhline(1, color=INK2, lw=0.6)
        ax.set_ylim(0, 2.5)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_title(filt, fontsize=9, color=INK)
        if filt == FILTERS[0]:
            ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    for ax in axes[-1]:
        ax.set_xlabel('aperture EE fraction', fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(f'flux / flux(EE{int(100 * EE_PRIMARY)})', fontsize=8, color=INK2)
    fig.suptitle('Curve of growth: target vs injected point sources at the same local background '
                 '(primary map; band: 16–84% of injections)', fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, dpi=140, facecolor=SURF)
    plt.close(fig)


def plot_sed(comb, fname):
    import matplotlib.pyplot as plt
    import pandas as pd
    import starbug2
    old = pd.read_csv(OLD_STARBUG_FILE)
    wl = {f: starbug2.filters[f].wavelength for f in FILTERS}
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    mjy = lambda m: 3631e6 * 10 ** (-0.4 * m)
    dol = np.array([dolphot_ujy(f) for f in FILTERS], dtype=object)
    det = dol[:, 2].astype(bool)
    x0 = np.array([wl[f] for f in FILTERS])
    ax.errorbar(x0[det], dol[det, 0].astype(float), yerr=dol[det, 1].astype(float), fmt='o', mfc='none', mec=INK2,
                ecolor=INK2, ms=6, label='dolphot (published pipeline)')
    ax.errorbar(old['wave_um'], mjy(old['psf_forced_mag']), fmt='s', mfc='none', mec='#e87ba4', ms=5, label='starbug, old (no bgd subtraction)')
    series = (('psf_forced', 'D', 'PSF forced at F560W position'), ('psf_free', 'd', 'PSF free'), ('ap', 'o', METHOD_LABEL['ap']))
    for k, (key, mk, lab) in enumerate(series):
        col = METHOD_COLOR[key]
        x = x0 * (1 + 0.012 * (k - 1))
        f = comb[f'{key}_ujy'].values
        e = comb[f'{key}_tot_err_ujy'].values
        ul = comb[f'{key}_ul_ujy'].notna().values
        rel = comb['psf_free_reliable'].values if key == 'psf_free' else np.ones(len(f), bool)
        ax.errorbar(x[~ul & rel], f[~ul & rel], yerr=e[~ul & rel], fmt=mk, color=col, ms=6, lw=1.5, label=lab, zorder=3)
        ax.errorbar(x[ul & rel], comb[f'{key}_ul_ujy'].values[ul & rel], yerr=0.25 * comb[f'{key}_ul_ujy'].values[ul & rel],
                    uplims=True, fmt='_', color=col, ms=8, lw=1.5, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([5, 7, 10, 15, 20, 25]); ax.set_xticklabels(['5', '7', '10', '15', '20', '25'])
    ax.set_xticks([], minor=True)
    ax.set_xlabel('wavelength (µm)', color=INK2); ax.set_ylabel('F$_ν$ (µJy)', color=INK2)
    ax.tick_params(colors=INK2)
    ax.set_title('SN 2026sqf progenitor, MIRI 2024-04-12 (errors incl. map-sweep and background-structure terms; bars = 3σ limits;\n'
                 'free PSF shown only where ≥ 2 dithers stay within HWHM)', fontsize=8.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(fname, dpi=140, facecolor=SURF)
    plt.close(fig)


def write_table(comb, fname):
    import starbug2
    cols = ['filter', 'wave_um', 'box', 'bgdr', 'n_frames', 'n_target_detected']
    for m in METHODS:
        cols += [f'{m}_ujy', f'{m}_err_ujy', f'{m}_sys_ujy', f'{m}_bgd_ujy', f'{m}_tot_err_ujy', f'{m}_inj_ratio',
                 f'{m}_snr', f'{m}_det_snr', f'{m}_mag', f'{m}_mag_err', f'{m}_ul_ujy']
        if m.startswith('psf_xy'):
            cols.append(f'{m}_n_refit')
    cols += ['psf_free_n_used', 'psf_free_reliable', 'psf_free_dra_nircam_mas', 'psf_free_ddec_nircam_mas', 'psf_free_pos_err_mas',
             'ap_crds_ujy', 'ap_nosky_ujy'] + [f'{p}_{tag_ee(e)}_ujy' for p in ('ap', 'ap_rawann') for e in EE_LIST]
    comb = comb.copy()
    comb['wave_um'] = [starbug2.filters[f].wavelength for f in comb['filter']]
    with open(fname, 'w') as fp:
        fp.write('# SN 2026sqf MIRI (GO 3295, 2024-04-12) starbug2 photometry on JHAT stage-2 frames, made by starbug_miri_2026sqf.py\n')
        fp.write('# position: mean F560W free-PSF centroid (MIRI frame); PSF: stpsf DET_DIST, dated OPD, total-flux normalised\n')
        fp.write(f'# background: starbug2 map; box/bgdr columns (BGD_R = bgdr x FWHM) = map with the least injection bias; '
                 f'*_sys_ujy = std over BOX_SIZE {BOX_SIZES} x BGD_R {BGDR_FACTORS} FWHM\n')
        fp.write('# psf_forced: FORCE_POS=1; psf_xyNN: starbug MAX_XYDEV = N.N pix refit (n_refit: dithers forced); '
                 'psf_free: no refit, fits within one HWHM only\n')
        fp.write(f'# ap: EE{int(100 * EE_PRIMARY)} radius, CRDS sky annulus on the bgd-subtracted frame, stpsf apcorr incl. annulus wings; '
                 'ap_rawann: same on the cal frame (no starbug map); ap_crds: CRDS apcorr (i2d-based); ap_nosky: starbug map as the only sky\n')
        fp.write('# psf_free_d*_nircam: centroid - NIRCam dolphot counterpart, MIRI frame offset removed\n')
        fp.write('# err = max(formal, dither scatter/sqrt(N)); sys = std over maps; bgd = injection scatter beyond pixel noise x injected flux '
                 '(background structure, does not average over dithers); tot_err = err (+) sys (+) bgd; snr/mag errors use tot_err; '
                 'det_snr = f / (err (+) sys); ul = max(f,0)+3 tot_err when det_snr<3\n')
        fp.write('# inj_ratio: median recovered/injected flux of injected point sources through the same map (not applied)\n')
        comb[cols].to_csv(fp, index=False, float_format='%.4f')


def main():
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    os.makedirs(PSF_DIR, exist_ok=True)
    shutil.copy(CRDS_APCORR, f'{PSF_DIR}/apcorr_miri.fits')
    files = sorted(glob.glob(f'{JHAT_DIR}/*_mirimage_jhat.fits'))
    frames_csv = f'{OUT}/sn2026sqf_miri_starbug_frames.csv'
    inj_csv = f'{OUT}/sn2026sqf_miri_starbug_injections.csv'
    if '--plots-only' not in sys.argv:
        ra_t, dec_t = load_target()
        detpos = {}
        for f in files:
            with fits.open(f) as h:
                filt = h[0].header['FILTER']
                if filt not in detpos:
                    x, y = WCS(h['SCI'].header).all_world2pix(ra_t, dec_t, 0)
                    detpos[filt] = (int(round(float(x))), int(round(float(y))))
        with Pool(NCORES) as p:
            print('PSFs:', p.map(make_psf, [(f, detpos[f]) for f in FILTERS]))
        # pass 1: F560W position
        f560 = [f for f in files if fits.getheader(f)['FILTER'] == 'F560W']
        with Pool(NCORES) as p:
            loc = pd.DataFrame([r for res, _, _ in p.map(run_frame, [(f, ra_t, dec_t, True) for f in f560]) for r in res])
        good = loc[psf_good(loc)]
        ra_p, dec_p = good['ra_fit'].mean(), good['dec_fit'].mean()
        print(f'F560W position from {len(good)} dithers: {ra_p:.7f} {dec_p:.7f} '
              f'({(ra_p - ra_t) * np.cos(np.radians(dec_t)) * 3.6e6:+.0f}, {(dec_p - dec_t) * 3.6e6:+.0f} mas from the catalogue)')
        # pass 2
        for d in glob.glob(f'{OUT}/box*'):
            shutil.rmtree(d)
        with Pool(NCORES) as p:
            out = p.map(run_frame, [(f, ra_p, dec_p, False) for f in files])
        pd.DataFrame([r for res, _, _ in out for r in res]).to_csv(frames_csv, index=False)
        pd.DataFrame([z for _, zp, _ in out for z in zp]).to_csv(f'{OUT}/sn2026sqf_miri_starbug_zp.csv', index=False)
        pd.DataFrame([z for _, _, inj in out for z in inj]).to_csv(inj_csv, index=False)
    frames = pd.read_csv(frames_csv)
    inj = pd.read_csv(inj_csv)
    astro = astrometry_check()
    print(f'MIRI - NIRCam frame offset: {astro["frame"].round(1)} +- {astro["frame_err"].round(1)} mas from {astro["n_frame"]} stars')
    summ = injection_summary(inj)
    sel = select_maps(summ)
    print('selected maps (box, BGD_R/FWHM):', sel)
    frames['primary'] = [sel[f] == (b, r) for f, b, r in zip(frames['filter'], frames['box'], frames['bgdr'])]
    summ['primary'] = [sel[f] == (b, r) for f, b, r in zip(summ['filter'], summ['box'], summ['bgdr'])]
    summ.to_csv(f'{OUT}/sn2026sqf_miri_starbug_injection_summary.csv', index=False, float_format='%.4f')
    comb = combine(frames, astro, summ)
    write_table(comb, f'{OUT}/sn2026sqf_miri_starbug_photometry.txt')
    plot_residuals(frames, f'{OUT}/sn2026sqf_miri_starbug_residuals.png')
    plot_identity(frames, comb, astro, f'{OUT}/sn2026sqf_miri_starbug_identity.png')
    plot_psfmodes(comb, f'{OUT}/sn2026sqf_miri_starbug_psfmodes.png')
    plot_injection(summ, f'{OUT}/sn2026sqf_miri_starbug_injection.png')
    plot_cog(comb, summ, f'{OUT}/sn2026sqf_miri_starbug_cog.png')
    plot_sed(comb, f'{OUT}/sn2026sqf_miri_starbug_sed.png')
    pd.set_option('display.width', 250)
    print(comb[['filter', 'box'] + [f'{m}_ujy' for m in METHODS] + ['ap_err_ujy', 'ap_sys_ujy', 'psf_forced_err_ujy', 'psf_forced_sys_ujy']].round(2).to_string())
    print(comb[['filter', 'box', 'bgdr'] + [f'{m}_{k}' for m in ('psf_forced', 'ap') for k in ('err_ujy', 'sys_ujy', 'bgd_ujy', 'tot_err_ujy', 'inj_ratio')]].round(2).to_string())
    print(summ[summ['primary']][['filter', 'box', 'bgdr', 'n', 'f_in'] + [c for c in summ.columns if c.endswith(('_ratio', '_scatter', '_noise'))]
                                + ['psf_free_frac_wandered']].round(3).to_string())


if __name__ == '__main__':
    main()
