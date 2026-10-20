'''
Quick RADMC-3D mockup of the best 4000 K torus mix, compared with its 1D DUSTY analogues.

Geometry: rho(r, theta) ∝ g(mu) (R_in / r)^s for R_in < r < Y R_in, normalised so the radial V-band
extinction optical depth along mu = cos theta is tau(mu), with PROFILE (env) either
  kochanek  tau(mu) = tau_p + (tau_e - tau_p)(1 - |mu|)^eta       (Kochanek 2024, MNRAS 529, 1958)
  wedge     tau(mu) = tau_e for |mu| < sin(theta_disk), else tau_p  (sharp-edged equatorial wedge of
            half-opening theta_disk, Uno et al. 2023, ApJ 944, 204, with a thin polar component added)
  disk      flared disk rho = rho_0 (r/R_in)^-(p+beta) exp(-z^2 / 2H^2), H = h0 R_in (r/R_in)^beta (surface
            density ∝ r^-p; env H0, BETA, P), plus a spherical (R_in / r)^s envelope of radial tau_p; the
            disk's midplane radial tau is tau_e - tau_p, so the equatorial total is tau_e Everything else is ported from the DUSTY
setup of the best 4000 K torus mix (T_in = 1200 K, s = 1, Y = 1000; s and Y set by env S, Y): the COMARCS input
spectrum, 80% amC (Zubko ACAR) + 20% SiC (Pegourie) by number, MRN q = 3 between 0.05 and
1 micron, isotropic scattering, and R_in from DUSTY at T_in = 1200 K along the polar tau.

Compared:
  RADMC 2D       SED at several inclinations, and its 4 pi average
  DUSTY sectors  int_0^1 F_DUSTY(tau(mu)) dmu, i.e. independent 1D sightlines (each at its own R_in)
  DUSTY mix      (1 - f) F(tau_e) + f F(tau_p), the literal two-component analogue (f = F_THIN; eta = 1 -> half
                 of the sky has tau above the midpoint)
  RADMC 1D       spherical RADMC at tau_p and tau_e against DUSTY (validates the opacity/star port)

Optional old wind (WIND_TAU > 0, wedge profile only): a spherical rho ∝ r^-2 shell of radial V-band
optical depth WIND_TAU from WIND_IN to WIND_OUT (units of R_in) is added to the torus density. Its 1D
analogue heats a DUSTY wind shell with the emergent spectrum of each torus sector (back-heating of the
torus by the wind is neglected), with T_in iterated so the wind starts at the same radius as in RADMC.

Run from the repo root:  python radmc_torus_test.py   (needs radmc3d on PATH and DUSTY_PATH set)
'''
import os, sys, time, subprocess, pickle, hashlib
import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii
from radmc3dPy.miescat import bhmie
from rsg_phot.dust import dusty_likelihood

ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.environ.get('PROFILE', 'kochanek')
TAG = '' if (os.environ.get('TAU_E'), os.environ.get('TAU_P')) == (None, None) else (
    f"_te{os.environ['TAU_E']}_tp{os.environ['TAU_P']}" + ''.join(f'_{k.lower()}{os.environ[k]}' for k in ('ETA', 'S', 'Y', 'TIN') if k in os.environ)
    + (f"_wedge{os.environ['THDISK']}" if PROFILE == 'wedge' else '')
    + (f"_disk{os.environ.get('H0', '0.15')}b{os.environ.get('BETA', '1.2')}p{os.environ.get('P', '1')}" if PROFILE == 'disk' else '') + ('_' + os.environ['DUST'] if 'DUST' in os.environ else '')
    + ''.join(f"_{k.lower().replace('_', '')}{os.environ[k]}" for k in ('WIND_TAU', 'WIND_IN', 'WIND_OUT') if k in os.environ))
OUT = os.path.join(ROOT, 'data', 'radmc_torus_test' + TAG)
PLOT = os.path.join(ROOT, 'plots', f'radmc_torus_test{TAG}.png')

# model (best 4000 K torus mix except tau)
TEFF, TIN = 4000., float(os.environ.get('TIN', 1200.))
S, Y = float(os.environ.get('S', 1.)), float(os.environ.get('Y', 1000.))
# optional old wind around the torus: rho ∝ r^-2 from WIND_IN to WIND_OUT (units of R_in), radial tau_V = WIND_TAU
WIND_TAU = float(os.environ.get('WIND_TAU', 0.))
WIND_IN, WIND_OUT, WIND_S = float(os.environ.get('WIND_IN', Y)), float(os.environ.get('WIND_OUT', 1e4)), 2.
# tau_e / tau_p / eta / weight of the thin component in the two-point mix; override with env TAU_E, TAU_P, ETA, F_THIN
TAU_E, TAU_P, ETA = float(os.environ.get('TAU_E', 10.)), float(os.environ.get('TAU_P', 1.)), float(os.environ.get('ETA', 1.))
F_THIN = float(os.environ.get('F_THIN', 0.5))
THDISK = float(os.environ.get('THDISK', 30.))   # wedge half-opening angle from the equator (deg)
MU_EDGE = np.sin(np.radians(THDISK))
H0, BETA, P_SIG = float(os.environ.get('H0', 0.15)), float(os.environ.get('BETA', 1.2)), float(os.environ.get('P', 1.))
LOGL = float(os.environ.get('LOGL', 4.944))   # best-fit log L of the matching 1D mix (for the plot only)
# dust mixture by number (DUST env): amC + 20% SiC (default) or pure amC (Zubko ACAR)
DUST = os.environ.get('DUST', 'amC_SiC20')
NK = {'amC_SiC20': {'amC-zb2.nk': 0.8, 'SiC-peg.nk': 0.2}, 'amC': {'amC-zb2.nk': 1.0}}[DUST]
AMIN, AMAX, Q, NA = 0.05, 1.0, 3.0, 50
# RADMC settings
NR, NTH = 200, 90
NPHOT, NPHOT_SPEC = 1_000_000, 10_000
THREADS = int(os.environ.get('THREADS', 10))
TLIMIT = int(os.environ.get('TLIMIT', 600))    # seconds per radmc3d call
# scattering costs ~0.6 s per wavelength (vs ~0.04 s without), so SEDs are computed without
# scattering on the full grid and corrected by the scattered/unscattered ratio on this subset
# (albedo 0.5 at V, 0.18 at 10 um, 0.04 at 20 um)
LAM_SCAT = np.logspace(np.log10(0.4), np.log10(14.), 24)
def disk_tau(m, n=4000):
    # radial V-band tau of the disk part along mu = cos(theta) (the envelope adds tau_p)
    r = np.logspace(0, np.log10(Y), n)
    h = H0*r**BETA
    f = r**-(P_SIG + BETA)*np.exp(-0.5*(r*m/h)**2)
    return (TAU_E - TAU_P)*np.trapezoid(f, r)/np.trapezoid(r**-(P_SIG + BETA), r)
def taufun(m):
    if PROFILE == 'disk':
        return TAU_P + disk_tau(m)
    if PROFILE == 'wedge':
        return TAU_E if abs(m) < MU_EDGE else TAU_P
    return TAU_P + (TAU_E - TAU_P)*(1 - abs(m))**ETA
def gauss(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5*(b - a)*x + 0.5*(b + a), 0.5*(b - a)*w
# viewing-angle quadrature for the 4 pi average over mu in [0, 1]; the wedge SED changes steeply
# across the wedge edge, so integrate the two sides separately
if PROFILE in ('wedge', 'disk'):
    # for the disk, split at about twice the outer H/r, where the disk sightlines end
    MU_SPLIT = MU_EDGE if PROFILE == 'wedge' else min(0.9, 2*H0*Y**(BETA - 1))
    (m1, w1), (m2, w2) = gauss(0., MU_SPLIT, 4), gauss(MU_SPLIT, 1., 4)
    MU_GL, W_GL = np.concatenate([m1, m2]), np.concatenate([w1, w2])
else:
    MU_GL, W_GL = gauss(0., 1., 8)
# plotted with the quadrature inclinations (30-50 deg: Uno+23 viewing angle); exactly 90 deg
# crashes the ray tracer with midplane mirror symmetry
INCL_SHOW = [0., 30., 50., 89.9] if PROFILE in ('wedge', 'disk') else [0., 89.9]

LSUN, PC, SIGSB, C_UM = 3.828e33, 3.0857e18, 5.6704e-5, 2.99792458e14
DUSTY_L = 1e4                    # DUSTY reference luminosity (Lsun); all runs at this L

def lam_grid():
    # dense through the SiC / C2H2 region
    return np.unique(np.concatenate([np.logspace(-1, 3, 150), np.linspace(7., 16., 46)]))

def builder(cls=dusty_likelihood, work='dusty_work'):
    return cls(dist=19*u.Mpc, modeldir=os.path.join(ROOT, 'data/marcs/comarcs'), logg=0.0, logz=-0.5,
                            mass=2.0, c_o=1.4, density='powerlaw', shell_thickness=Y,
                            comp={'sic': 0.2} if DUST == 'amC_SiC20' else None,
                            nk_comp={'amC-zb2.nk': 0.8} if DUST == 'amC_SiC20' else {'amC-zb2.nk': 1.0}, outdir=os.path.join(ROOT, 'data/dusty_grid'),
                            workdir=os.path.join(OUT, work))

def run_dusty(B, taus):
    p = {'logl': 4., 'temp': TEFF, 'dust_temp': TIN, 'shell_thickness': Y, 'density_power': S}
    out = {}
    for tau, (spec, res) in zip(np.sort(taus), B.run_dusty_sequence(p, taus)):
        if spec is None:
            raise RuntimeError(f'DUSTY failed at tau {tau}: {res.get("failed")}')
        wv, i = np.unique(np.asarray(spec['lambda']), return_index=True)
        out[round(float(tau), 6)] = {'lam': wv, 'ftot': np.asarray(spec['fTot'])[i], 'r1': res['r1(cm)']}
    return out

class _Nested(dusty_likelihood):
    # DUSTY run whose input spectrum is a file written by wind_dusty
    input_file = None
    def get_input_spec(self, temp, workdir):
        return self.input_file or super().get_input_spec(temp, workdir)

def wind_dusty(Bw, inner, r_target, tag):
    # DUSTY wind shell (rho ∝ r^-2, radial tau_V = WIND_TAU, Y = WIND_OUT / WIND_IN) heated by the emergent
    # spectrum of one torus sector; T_in is iterated (r1 ∝ T_in^-2.5) until r1 is within 1% of r_target
    fn = os.path.join(Bw.get_workdir(), f'torus_{tag}.dat')
    with open(fn, 'w') as f:
        f.write(f'torus sector {tag}\n  lambda    L_lambda\n (micron)  (arbitrary)\n')
        np.savetxt(f, np.c_[inner['lam'], np.clip(inner['ftot']/inner['lam'], 1e-60, None)])
    Bw.input_file = fn
    t = 300.
    p = {'logl': 4., 'temp': TEFF, 'shell_thickness': WIND_OUT/WIND_IN, 'density_power': WIND_S}
    for _ in range(10):
        spec, res = Bw.run_dusty_sequence(dict(p, dust_temp=t), [WIND_TAU, 1.001*WIND_TAU])[0]
        if spec is None:
            raise RuntimeError(f'DUSTY wind failed ({tag}, T_in {t:.1f}): {res.get("failed")}')
        r1 = res['r1(cm)']
        if abs(r1/r_target - 1) < 0.01:
            break
        t *= (r1/r_target)**0.4
    wv, i = np.unique(np.asarray(spec['lambda']), return_index=True)
    return {'lam': wv, 'ftot': np.asarray(spec['fTot'])[i], 'tin': t, 'r1': r1}

def read_nk(name):
    t = np.loadtxt(os.path.join(os.environ['DUSTY_PATH'], 'Lib_nk', name), skiprows=7)
    return t[:, 0], t[:, 1], t[:, 2]

def opacities(lam):
    # DUSTY's synthetic grain: per-material size-averaged cross sections (MRN, NA log-spaced
    # sizes), mixed by number fraction. The grain mass is arbitrary since tau sets the density.
    a = np.logspace(np.log10(AMIN), np.log10(AMAX), NA)
    nsd = a**-Q
    norm = np.trapezoid(nsd, a)
    sig_abs, sig_sca = np.zeros(lam.size), np.zeros(lam.size)
    for name, x in NK.items():
        wl, n, k = read_nk(name)
        nl, kl = np.interp(lam, wl, n), np.interp(lam, wl, k)
        qa, qs = np.zeros((NA, lam.size)), np.zeros((NA, lam.size))
        for j in range(lam.size):
            for i in range(NA):
                r = bhmie(np.array(2*np.pi*a[i]/lam[j]), nl[j] + 1j*kl[j], np.array([0., 180.]))
                qa[i, j], qs[i, j] = r[3], r[4]
        area = np.pi*(a*1e-4)**2
        sig_abs += x*np.trapezoid(nsd[:, None]*qa*area[:, None], a, axis=0)/norm
        sig_sca += x*np.trapezoid(nsd[:, None]*qs*area[:, None], a, axis=0)/norm
    mgrain = 1e-12
    return sig_abs/mgrain, sig_sca/mgrain

def star_fnu_1pc(B, lam):
    # COMARCS input spectrum DUSTY uses (L_lambda, arbitrary), as F_nu at 1 pc for L = DUSTY_L,
    # normalised on the RADMC grid so the injected luminosity is exact
    t = np.loadtxt(B.get_input_spec(TEFF, B.get_workdir()), skiprows=3)
    lfl = np.exp(np.interp(np.log(lam), np.log(t[:, 0]), np.log(t[:, 0]*t[:, 1])))
    lnu = lfl/(C_UM/lam)
    nu = C_UM/lam
    lnu *= DUSTY_L*LSUN/np.trapezoid(lnu[::-1], nu[::-1])
    return lnu/(4*np.pi*PC**2)

def write_model(d, lam, kabs, ksca, fstar, rin, taufun, ntheta, wind=True):
    os.makedirs(d, exist_ok=True)
    w = lambda fn, s: open(os.path.join(d, fn), 'w').write(s)
    # log-spaced in r - R_in from 1e-4 R_in, so the innermost cells stay optically thin at high tau
    # (and for steep r^-s) while the outer cells grow by ~11% each
    ri = rin*(1 + np.concatenate([[0.], np.logspace(-4, np.log10(Y - 1), NR)]))
    wind = wind and WIND_TAU > 0
    if wind:
        # extend out to the wind's outer edge, refined just outside its inner edge
        rw = rin*WIND_IN
        ext = np.logspace(np.log10(ri[-1]), np.log10(rin*WIND_OUT), NR//2 + 1)[1:]
        ri = np.unique(np.concatenate([ri, ext, [rw], rw*(1 + np.logspace(-4, np.log10(0.3), 40))]))
        ri = ri[np.concatenate([[True], np.diff(ri)/ri[1:] > 1e-7])]
    nr = len(ri) - 1
    rc = np.sqrt(ri[1:]*ri[:-1])
    if ntheta == 1:
        ti = np.array([0., np.pi])
        mu = np.array([0.])          # tau is isotropic in 1D; taufun ignores mu
    else:
        ti = np.linspace(0., np.pi/2, ntheta + 1)       # mirror symmetry about the equator
        mu = np.cos(0.5*(ti[1:] + ti[:-1]))
    kext_v = np.exp(np.interp(np.log(0.55), np.log(lam), np.log(kabs + ksca)))
    # rho(r, theta) ∝ tau(mu) r^-s, each theta row renormalised to the exact radial tau on this grid
    shape = np.where(rc < rin*Y, (rc/rin)**-S, 0.)
    if PROFILE == 'disk' and ntheta > 1:
        # spherical envelope of radial tau_p plus the flared disk, whose midplane radial tau is tau_e - tau_p
        env = TAU_P*shape/(kext_v*np.sum(shape*np.diff(ri)))
        x = rc/rin
        mid = np.where(rc < rin*Y, x**-(P_SIG + BETA), 0.)
        mid *= (TAU_E - TAU_P)/(kext_v*np.sum(mid*np.diff(ri)))
        z_over_h = (x[:, None]*mu[None, :])/(H0*x[:, None]**BETA)
        rho = env[:, None] + mid[:, None]*np.exp(-0.5*z_over_h**2)
    else:
        rho = np.array([taufun(m)*shape/(kext_v*np.sum(shape*np.diff(ri))) for m in mu]).T    # (nr, ntheta)
    if wind:
        ws = np.where((rc > rin*WIND_IN) & (rc < rin*WIND_OUT), (rc/rin)**-WIND_S, 0.)
        rho += (WIND_TAU*ws/(kext_v*np.sum(ws*np.diff(ri))))[:, None]
    w('wavelength_micron.inp', f'{lam.size}\n' + ''.join(f'{x:13.6e}\n' for x in lam))
    w('camera_wavelength_micron.inp', f'{LAM_SCAT.size}\n' + ''.join(f'{x:13.6e}\n' for x in LAM_SCAT))
    rstar = np.sqrt(DUSTY_L*LSUN/(4*np.pi*SIGSB*TEFF**4))
    w('stars.inp', f'2\n1 {lam.size}\n\n{rstar:13.6e} 1.989e33 0 0 0\n\n' + ''.join(f'{x:13.6e}\n' for x in lam)
      + '\n' + ''.join(f'{x:13.6e}\n' for x in fstar))
    w('amr_grid.inp', '1\n0\n100\n0\n' + ('1 0 0\n' if ntheta == 1 else '1 1 0\n') + f'{nr} {ntheta} 1\n'
      + ''.join(f'{x:21.14e}\n' for x in ri) + ''.join(f'{x:21.14e}\n' for x in ti) + f'0\n{2*np.pi:21.14e}\n')
    w('dust_density.inp', f'1\n{rho.size}\n1\n' + ''.join(f'{x:13.6e}\n' for x in rho.ravel(order='F')))
    w('dustopac.inp', '2\n1\n============\n1\n0\nmix\n------------\n')
    w('dustkappa_mix.inp', f'2\n{lam.size}\n' + ''.join(f'{l:13.6e} {a:13.6e} {s:13.6e}\n' for l, a, s in zip(lam, kabs, ksca)))
    # camera_min_drr caps the image sub-pixeling at dr/r = 0.03 (about the resolution of a plain
    # log grid); without it the 1e-4 R_in inner cells make each SED ~30x slower
    w('radmc3d.inp', f'nphot = {NPHOT}\nnphot_spec = {NPHOT_SPEC}\nscattering_mode_max = 1\n'
      'modified_random_walk = 1\nistar_sphere = 0\ncamera_min_drr = 0.03\n')
    return rc, mu

def radmc(d, *args):
    t0 = time.time()
    r = subprocess.run(['perl', '-e', f'alarm {TLIMIT}; exec @ARGV', 'radmc3d', *args], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'radmc3d {" ".join(args)} failed or hit the {TLIMIT} s limit in {d}:\n{r.stdout[-1500:]}')
    return time.time() - t0

def read_sed(d):
    s = np.loadtxt(os.path.join(d, 'spectrum.out'), skiprows=3)
    lam, fnu = s[:, 0], s[:, 1]
    return lam, fnu*(C_UM/lam)*4*np.pi*PC**2/(DUSTY_L*LSUN)        # lambda F_lambda / F_bol, as DUSTY fTot

def sed(d, incl):
    # full-grid SED without scattering, times the scattering correction from the LAM_SCAT subset
    # (interpolated in log lambda; held at its end values outside the subset)
    t = radmc(d, 'sed', 'incl', f'{incl:.4f}', 'noscat')
    lam, f0 = read_sed(d)
    t += radmc(d, 'spectrum', 'incl', f'{incl:.4f}', 'loadlambda')
    ls, fs = read_sed(d)
    corr = fs/np.exp(np.interp(np.log(ls), np.log(lam), np.log(f0)))
    return lam, f0*np.interp(np.log(lam), np.log(ls), corr), t, (ls, corr)

def read_temp(d, nth):
    t = np.loadtxt(os.path.join(d, 'dust_temperature.dat'), skiprows=3)
    return t.reshape((-1, nth), order='F')

def bol(lam, ftot):
    return np.trapezoid(ftot/lam, lam)

def main():
    os.makedirs(OUT, exist_ok=True)
    B = builder()
    lam = lam_grid()
    timing = {}

    t0 = time.time()
    extra = {'wedge': [], 'disk': list(np.logspace(np.log10(TAU_P), np.log10(TAU_E), 20))}.get(PROFILE, list(TAU_P + (TAU_E - TAU_P)*(1 - MU_GL)**ETA))
    taus = np.unique(np.round([TAU_P, TAU_E] + extra, 6))
    D = run_dusty(B, taus)
    timing['DUSTY (all taus)'] = time.time() - t0
    rin = D[TAU_P]['r1']
    print(f'DUSTY: {len(D)} taus; R_in(tau_p) = {rin:.3e} cm, R_in(tau_e) = {D[TAU_E]["r1"]:.3e} cm at L = 1e4 Lsun')
    W = {}
    if WIND_TAU > 0:
        if PROFILE != 'wedge':
            raise ValueError('the wind is only set up for PROFILE=wedge')
        t0 = time.time()
        Bw = builder(_Nested, 'dusty_wind')
        for tau in (TAU_P, TAU_E):
            W[tau] = wind_dusty(Bw, D[tau], rin*WIND_IN, f'tau{tau:g}')
            print(f'DUSTY wind behind the tau {tau:g} sector: T_in {W[tau]["tin"]:.0f} K at r1 = {W[tau]["r1"]:.3e} cm', flush=True)
        timing['DUSTY wind'] = time.time() - t0

    t0 = time.time()
    kabs, ksca = opacities(lam)
    timing['Mie opacities'] = time.time() - t0
    fstar = star_fnu_1pc(B, lam)

    res = {'lam_grid': lam, 'kabs': kabs, 'ksca': ksca, 'dusty': D, 'dusty_wind': W, 'rin': rin, 'radmc1d': {}, 'radmc2d': {}}
    # 2D torus
    d = os.path.join(OUT, 'radmc2d')
    rc, mu = write_model(d, lam, kabs, ksca, fstar, rin, taufun, NTH)
    # cache the thermal solution and per-inclination SEDs, keyed on the input files
    key = hashlib.md5(b''.join(open(os.path.join(d, f), 'rb').read() for f in sorted(os.listdir(d)) if f.endswith('.inp'))).hexdigest()
    cfile = os.path.join(d, 'cache.pkl')
    cache = pickle.load(open(cfile, 'rb')) if os.path.exists(cfile) else {}
    if cache.get('key') != key or not os.path.exists(os.path.join(d, 'dust_temperature.dat')):
        cache = {'key': key, 'tt': radmc(d, 'mctherm', 'setthreads', str(THREADS)), 'sed': {}}
        pickle.dump(cache, open(cfile, 'wb'))
    tt = cache['tt']
    T2 = read_temp(d, NTH)
    incl = sorted(set(np.degrees(np.arccos(MU_GL))) | set(INCL_SHOW))
    ts, seds, scat = 0., {}, {}
    for i in incl:
        if i not in cache['sed']:
            cache['sed'][i] = sed(d, i)
            pickle.dump(cache, open(cfile, 'wb'))
        l2, seds[i], t, scat[i] = cache['sed'][i]
        ts += t
        print(f'  incl {i:5.1f}: {t:.0f} s', flush=True)
    avg = sum(w*seds[np.degrees(np.arccos(m))] for m, w in zip(MU_GL, W_GL))
    timing['RADMC 2D: mctherm / all seds'] = (tt, ts)
    print(f'RADMC 2D: mctherm {tt:.0f} s, {len(incl)} seds {ts:.0f} s; 4pi-average L_out/L_in = {bol(l2, avg):.3f}')
    print(f'  T at R_in: pole {T2[0, 0]:.0f} K, equator {T2[0, -1]:.0f} K')
    res['radmc2d'] = {'lam': l2, 'incl': incl, 'sed': seds, 'avg': avg, 'T': T2, 'r': rc, 'mu': mu, 'scat': scat}

    # 1D spherical validation at tau_p and tau_e (each at DUSTY's own R_in for that tau)
    for tau in (TAU_P, TAU_E):
        d = os.path.join(OUT, f'radmc1d_tau{tau:g}')
        rc, _ = write_model(d, lam, kabs, ksca, fstar, D[tau]['r1'], lambda m, t=tau: t, 1, wind=False)
        tt = radmc(d, 'mctherm', 'setthreads', str(THREADS))
        l, f, ts, sc = sed(d, 0.)
        res['radmc1d'][tau] = {'lam': l, 'ftot': f, 'T': read_temp(d, 1)[:, 0], 'r': rc, 'scat': sc}
        timing[f'RADMC 1D tau={tau:g}: mctherm / sed'] = (tt, ts)
        print(f'RADMC 1D tau {tau:g}: mctherm {tt:.0f} s, sed {ts:.0f} s; T(R_in) = {res["radmc1d"][tau]["T"][0]:.0f} K '
              f'(DUSTY {TIN:.0f} K); L_out/L_in = {bol(l, f):.3f}')

    if W:
        # 1D check of the nested DUSTY wind: the tau_p sector plus wind, spherical, in RADMC
        d = os.path.join(OUT, 'radmc1d_taup_wind')
        rc, _ = write_model(d, lam, kabs, ksca, fstar, rin, lambda m: TAU_P, 1)
        tt = radmc(d, 'mctherm', 'setthreads', str(THREADS))
        l, f, ts, sc = sed(d, 0.)
        res['radmc1d_wind'] = {'lam': l, 'ftot': f, 'T': read_temp(d, 1)[:, 0], 'r': rc}
        timing['RADMC 1D tau_p + wind: mctherm / sed'] = (tt, ts)
        dd = np.exp(np.interp(np.log(l), np.log(W[TAU_P]['lam']), np.log(np.clip(W[TAU_P]['ftot'], 1e-300, None))))
        print(f'RADMC 1D tau_p + wind: mctherm {tt:.0f} s, sed {ts:.0f} s; L_out/L_in = {bol(l, f):.3f}; RADMC/DUSTY at 2/5/10/15/20/30 um: '
              + ' '.join(f'{f[np.argmin(abs(l - x))]/dd[np.argmin(abs(l - x))]:.2f}' for x in (2, 5, 10, 15, 20, 30)))

    # 1D analogues on the DUSTY grid
    lamd = D[TAU_P]['lam']
    if W:
        # each sector's torus output reprocessed by the wind
        Fe, Fp = np.interp(lamd, W[TAU_E]['lam'], W[TAU_E]['ftot']), np.interp(lamd, W[TAU_P]['lam'], W[TAU_P]['ftot'])
        sect = MU_EDGE*Fe + (1 - MU_EDGE)*Fp
    elif PROFILE == 'disk':
        # independent sightlines over a dense mu grid: each at tau(mu), log-flux interpolated in log tau
        T_ = np.array(sorted(D)); lt = np.log(T_)
        LF = np.log(np.clip(np.array([np.interp(lamd, D[t]['lam'], D[t]['ftot']) for t in T_]), 1e-300, None))
        mug = np.linspace(0, 1, 1001); x = np.clip(np.log([taufun(m) for m in mug]), lt[0], lt[-1])
        j = np.clip(np.searchsorted(lt, x) - 1, 0, len(lt) - 2); w = ((x - lt[j])/(lt[j+1] - lt[j]))[:, None]
        sect = np.trapezoid(np.exp((1 - w)*LF[j] + w*LF[j+1]), mug, axis=0)
    elif PROFILE == 'wedge':
        # exact: a fraction sin(theta_disk) of the sky at tau_e, the rest at tau_p
        sect = MU_EDGE*D[TAU_E]['ftot'] + (1 - MU_EDGE)*np.interp(lamd, D[TAU_P]['lam'], D[TAU_P]['ftot'])
    else:
        sect = sum(w*np.interp(lamd, D[t]['lam'], D[t]['ftot'])
                   for t, w in zip(np.round(TAU_P + (TAU_E - TAU_P)*(1 - MU_GL)**ETA, 6), W_GL))
    # two-point analogue; for the disk, the thin weight is the sky fraction below the geometric midpoint of tau_p and tau_e
    ft = float(np.mean([taufun(m) < np.sqrt(TAU_P*TAU_E) for m in np.linspace(0, 1, 1001)])) if PROFILE == 'disk' else F_THIN
    mix = (1 - ft)*Fe + ft*Fp if W else (1 - ft)*D[TAU_E]['ftot'] + ft*np.interp(lamd, D[TAU_P]['lam'], D[TAU_P]['ftot'])
    res['f_thin'] = ft
    res['dusty_sectors'], res['dusty_mix'], res['dusty_lam'] = sect, mix, lamd
    res['timing'] = timing
    pickle.dump(res, open(os.path.join(OUT, 'results.pkl'), 'wb'))
    plot(res)
    print('timing:', timing)

def photometry(B, BP, lam, ftot):
    # spec_to_sed renormalises to F_bol = L / 4 pi d^2, which is right for DUSTY's 4 pi output but
    # not for one viewing angle (an observer can see more or less than that), so undo it
    from scipy.integrate import simpson
    sed = B.spec_to_sed({'lambda': lam, 'fTot': ftot}, LOGL)
    k = simpson(ftot/lam, x=lam)
    sed['flam'] *= k; sed['fnu'] *= k
    p = B.model_phot(None, BP, sed=sed)
    return np.array([p[k] for k in BP]), sed['fnu'].value

def plot(res):
    from rsg_phot.rsg import rsg_phot
    B = builder()
    T = ascii.read(os.path.join(ROOT, 'data/dolphot/sn2026sqf/sn2026sqf_pipeline5as_photometry_ujy.txt'), fill_values=[('---', '0')])
    filt, inst = [str(f) for f in T['filter']], [str(i).lower() for i in T['instrument']]
    wave = np.asarray(T['wave_um'], float)
    det = np.asarray(T['detected']).astype(str) == 'True'
    flux = np.ma.filled(np.ma.asarray(T['flux_ujy'], float), np.nan)
    ferr = np.ma.filled(np.ma.asarray(T['flux_err_ujy'], float), np.nan)
    ul = np.ma.filled(np.ma.asarray(T['ul_ujy'], float), np.nan)
    rp = rsg_phot(verbose=False)
    pandeia = lambda f, i: f if i == 'nircam' else f'F{int(f[1:-1]):04d}W'
    BP = {f: rp.get_jwst_filters(f'{i},{pandeia(f, i)}') for f, i in zip(filt, inst)}
    band = ~np.isin(filt, ['F187N', 'F405N'])

    INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e4e3df'
    BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
    RAMP = ['#b9d4f5', '#86b6ef', '#5598e7', '#2a78d6', '#0d366b']
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8})
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[3, 1.4], hspace=0.06, wspace=0.18)
    ax, rx, tx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    R2, D = res['radmc2d'], res['dusty']
    lamd = res['dusty_lam']
    curves = {}
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list('incl', [RAMP[0], RAMP[-1]])(np.linspace(0, 1, len(R2['incl'])))
    for i, c in zip(R2['incl'], ramp):
        _, fnu = photometry(B, BP, R2['lam'], R2['sed'][i])
        lab = {0.: 'RADMC 2D, i = 0° (pole-on)', 89.9: 'RADMC 2D, i = 89.9° (edge-on); light → dark = pole → equator'}.get(i)
        ax.plot(R2['lam'], fnu, color=c, lw=1.0, label=lab)
    for key, lam, f, c, ls, lab in [('avg', R2['lam'], R2['avg'], INK, '-', 'RADMC 2D, 4π average'),
                                    ('sect', lamd, res['dusty_sectors'], ORANGE, '--', 'DUSTY independent sightlines ∫F(τ(μ))dμ'),
                                    ('mix', lamd, res['dusty_mix'], AQUA, ':', f'DUSTY two-component mix {1-res.get("f_thin", F_THIN):.2f}F(τ$_e$)+{res.get("f_thin", F_THIN):.2f}F(τ$_p$)')]:
        ph, fnu = photometry(B, BP, lam, f)
        curves[key] = (ph, lam, f)
        ax.plot(lam, fnu, color=c, ls=ls, lw=2.2 if key == 'avg' else 1.8, label=lab)
    obs = det & band
    ax.errorbar(wave[obs], flux[obs], yerr=ferr[obs], fmt='o', ms=5, color=INK2, mec='white', mew=0.8, elinewidth=1, zorder=10,
                label='observed (context only, not fitted)')
    lim = ~det & np.isfinite(ul)
    ax.errorbar(wave[lim], ul[lim], yerr=0.35*ul[lim], uplims=True, fmt='v', ms=5, color=INK2, mfc='white', mec=INK2, elinewidth=1, zorder=10)
    ax.set_yscale('log'); ax.set_ylim(0.01, 300); ax.set_ylabel('$F_\\nu$ (µJy) at 19 Mpc, log L = %.2f' % LOGL)
    ax.legend(loc='upper left', frameon=False, fontsize=7.8)

    # ratios to the independent-sightline DUSTY model
    ref = lambda lam: np.exp(np.interp(np.log(lam), np.log(lamd), np.log(res['dusty_sectors'])))
    rx.plot(R2['lam'], R2['avg']/ref(R2['lam']), color=INK, lw=1.8, label='RADMC 2D 4π average / DUSTY sightlines')
    rx.plot(lamd, res['dusty_mix']/res['dusty_sectors'], color=AQUA, ls=':', lw=1.8, label='DUSTY two-component / DUSTY sightlines')
    for tau, c in [(TAU_P, '#b4b2a9'), (TAU_E, INK2)]:
        r1 = res['radmc1d'][tau]
        dd = np.exp(np.interp(np.log(r1['lam']), np.log(D[tau]['lam']), np.log(D[tau]['ftot'])))
        rx.plot(r1['lam'], r1['ftot']/dd, color=c, lw=1, label=f'validation: RADMC 1D / DUSTY at τ = {tau:g}')
    if 'radmc1d_wind' in res:
        r1, Wp = res['radmc1d_wind'], res['dusty_wind'][TAU_P]
        dd = np.exp(np.interp(np.log(r1['lam']), np.log(Wp['lam']), np.log(np.clip(Wp['ftot'], 1e-300, None))))
        rx.plot(r1['lam'], r1['ftot']/dd, color=ORANGE, lw=1, label='validation: RADMC 1D / DUSTY nested, τ$_p$ + wind')
    rx.axhline(1, color=INK2, lw=0.8)
    rx.set_yscale('log'); rx.set_ylim(0.3, 3); rx.set_yticks([0.5, 1, 2]); rx.set_yticklabels(['0.5', '1', '2'])
    rx.set_ylabel('ratio'); rx.set_xlabel('Wavelength (µm)')
    rx.legend(loc='lower left', frameon=False, fontsize=7.2, ncol=2)
    for a in (ax, rx):
        a.set_xscale('log'); a.set_xlim(0.5, 40); a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for sp in ['top', 'right']: a.spines[sp].set_visible(False)
    ax.tick_params(labelbottom=False)
    for a in (ax, rx):
        a.set_xticks([0.5, 1, 2, 5, 10, 20, 40]); a.set_xticklabels(['0.5', '1', '2', '5', '10', '20', '40'])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # temperature profiles
    rin = res['rin']
    T2, r2, mu = R2['T'], R2['r'], R2['mu']
    for j, c, lab in [(0, RAMP[1], 'RADMC 2D, pole'), (len(mu)//2, RAMP[3], f'RADMC 2D, θ = {np.degrees(np.arccos(mu[len(mu)//2])):.0f}°'),
                      (-1, RAMP[4], 'RADMC 2D, equator')]:
        tx.plot(r2/rin, T2[:, j], color=c, lw=1.8, label=lab)
    for tau, c in [(TAU_P, '#b4b2a9'), (TAU_E, INK2)]:
        r1 = res['radmc1d'][tau]
        tx.plot(r1['r']/rin, r1['T'], color=c, lw=1, ls='--', label=f'RADMC 1D τ = {tau:g} (R$_{{in}}$ from DUSTY)')
    tx.axhline(TIN, color=ORANGE, lw=0.8, ls=':'); tx.text(1.05, TIN*1.04, f'DUSTY T$_{{in}}$ = {TIN:.0f} K', color=INK2, fontsize=8)
    tx.set_xscale('log'); tx.set_yscale('log'); tx.set_xlim(1, r2.max()/rin); tx.set_ylim(15, 2000)
    tx.set_xlabel('r / R$_{in}$ (R$_{in}$ = DUSTY r$_1$ at τ$_p$)'); tx.set_ylabel('dust temperature (K)')
    tx.grid(True, color=GRID, lw=0.6); tx.set_axisbelow(True)
    for sp in ['top', 'right']: tx.spines[sp].set_visible(False)
    tx.legend(loc='lower left', frameon=False, fontsize=7.8)

    tm = res['timing']
    fig.suptitle(f'RADMC-3D torus mockup vs 1D DUSTY: T$_{{eff}}$ = {TEFF:.0f} K, T$_{{in}}$ = {TIN:.0f} K, ρ ∝ r$^{{-{S:g}}}$, Y = {Y:g}, '
                 f'τ$_e$ = {TAU_E:g}, τ$_p$ = {TAU_P:g}, ' + {'wedge': f'wedge θ$_{{disk}}$ = {THDISK:g}°',
                 'disk': f'flared disk H/r = {H0:g}(r/R$_{{in}}$)$^{{{BETA - 1:.2g}}}$, Σ ∝ r$^{{-{P_SIG:g}}}$ + r$^{{-{S:g}}}$ envelope'}.get(PROFILE, f'η = {ETA:g}')
                 + ('; amC + 20% SiC' if DUST == 'amC_SiC20' else '; pure amC')
                 + (f'\n+ r$^{{-2}}$ wind, τ$_w$ = {WIND_TAU:g} from {WIND_IN:g} to {WIND_OUT:g} R$_{{in}}$' if WIND_TAU > 0 else ''), fontsize=11.5, y=0.97)
    fig.text(0.5, 0.915, f'RADMC 2D: mctherm {tm["RADMC 2D: mctherm / all seds"][0]:.0f} s ({NPHOT:.0e} photons, {THREADS} threads), '
             f'{len(R2["incl"])} SEDs {tm["RADMC 2D: mctherm / all seds"][1]:.0f} s', ha='center', fontsize=8.5, color=INK2)
    fig.savefig(PLOT, dpi=160, facecolor='white', bbox_inches='tight')
    print('saved', PLOT)

    # band-level summary
    print('\nband   ' + '  '.join(f'{k:>8s}' for k in curves) + '   (uJy at log L = %.2f)' % LOGL)
    for k, f in enumerate(BP):
        print(f'{f:6s} ' + '  '.join(f'{curves[c][0][k]:8.3f}' for c in curves))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot(pickle.load(open(os.path.join(OUT, 'results.pkl'), 'rb')))
    else:
        main()
