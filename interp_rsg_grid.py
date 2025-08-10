import glob, os
import pickle
import synphot 
import numpy as np
from astropy.io import fits, ascii
from scipy import interpolate
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from astropy.io.misc.hdf5 import read_table_hdf5
from synphot import SpectralElement
from synphot.models import Empirical1D
import astropy.units as u
import astropy.constants as const
import traceback

def get_jwst_filters(filtnam):

    inst, filt = filtnam.split(',')
    filt = filt.lower()
    inst = inst.lower()

    bpfile = os.path.join(bandpass_dir, inst, f'{filt.lower()}.txt')

    table = ascii.read(bpfile)
    wave = table['Microns']*1e4 * u.Angstrom
    trans = table['Throughput']

    bp = SpectralElement(Empirical1D, points=wave, lookup_table=trans)    
        
    return bp

def extinction_law(wave, Av, Rv, deredden=False):

    # Inverse wavelength dependent quantities
    x=1./(wave*1.0e-4)
    y=x-1.82

    # Variables proportional to extinction
    a=np.zeros(len(wave))
    b=np.zeros(len(wave))
    Fa=np.zeros(len(wave))
    Fb=np.zeros(len(wave))

    # Masks to apply piecewise Cardelli et al. function
    mask1 = np.where(x < 1.1)
    mask2 = np.where((x > 1.1) & (x < 3.3))
    mask3 = np.where(x > 3.3)
    mask4 = np.where(x > 5.9)

    x=1./(wave[mask1]*1.0e-4)
    y=x-1.82

    a[mask1]=0.574 * x**1.61
    b[mask1]=-0.527 * x**1.61

    x=1./(wave[mask2]*1.0e-4)
    y=x-1.82

    a[mask2]=1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 +\
        0.72085 * y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[mask2]=1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 -\
        5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7

    x=1./(wave[mask4]*1.0e-4)
    y=x-1.82

    Fa[mask4]=-0.04473 * (x-5.9)**2 - 0.009779 * (x-5.9)**3
    Fb[mask4]=0.2130 * (x-5.9)**2 + 0.1207 * (x-5.9)**3

    x=1./(wave[mask3]*1.0e-4)
    y=x-1.82

    a[mask3]=1.752 - 0.316 * x - 0.104/((x-4.67)**2 + 0.341) + Fa[mask3]
    b[mask3]=-3.090 + 1.825 * x + 1.206/((x-4.62)**2 + 0.263) + Fb[mask3]

    Alam = Av*(a+b/Rv)
    elam = 10**(-0.4 * Alam)
    if deredden: elam = 1./elam

    return(elam)

comp = 'sil'
bandpass_dir = os.environ['BANDPASS']
nrc_filts = ['F070W','F090W','F115W','F140M','F150W','F162M',
                    'F164N','F150W2','F182M','F187N','F200W','F210M',
                    'F212N','F250M','F277W','F300M','F322W2','F323N',
                    'F335M','F356W','F360M','F405N','F410M','F430M',
                    'F444W','F460M','F466N','F470N','F480M']

temps = np.array([2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0, 4200.0, 4500.0, 4800.0, 5000.0])
dust_temps = np.array([500.0, 800.0, 1000.0, 1200.0, 1500.0])
taus = np.array([0.01, 0.325, 0.641, 0.956, 1.27, 1.59, 1.9, 2.22, 2.53, 2.85, 3.16, 3.48, 3.79,
                 4.11, 4.42, 4.74, 5.05, 5.37, 5.68, 6.0])
lums = np.logspace(3, 6, 16)
rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
av = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 3.0, 5.0])

full_grid = np.zeros((1497, len(temps), len(dust_temps), len(taus), len(lums), len(rv), len(av)))
mags = {}
for flt in nrc_filts:
    mags[flt] = np.zeros((len(temps), len(dust_temps), len(taus), len(lums), len(rv), len(av)))


subdir = 'data/r2/dusty_sil_grid/dusty_out'
flux_scale = ((1 * u.L_sun) / (4 * np.pi * (10*u.Mpc)**2)).to(u.erg/u.s/u.cm**2)
for i, te in enumerate(temps):
    for j, td in enumerate(dust_temps):
        sd = f'rsg_{te}_{td}'
        specfile = os.path.join(subdir, sd, sd+'.hdf5')
        spectable = read_table_hdf5(specfile)
        wv = spectable['lambda'].to(u.Angstrom)
        _, mask = np.unique(wv, return_index=True)
        wv = wv[mask]
        energy = (const.h*const.c/wv).to(u.erg)
        print(specfile)

        for k, col in enumerate(spectable.columns[1:]):
            flux = spectable[col][mask]
            normalize = simpson(flux, x=wv.to(u.um).value)
            flux = flux_scale * flux/normalize / u.micron
            flux = flux.to(u.erg/u.s/u.cm**2/u.Angstrom)

            for m, Rv_ in enumerate(rv):
                for n, Av_ in enumerate(av):
                    host_ext = extinction_law(wv.value, Av_, Rv_)
                    flux_ext = flux*host_ext
                    sp = synphot.SourceSpectrum(Empirical1D, points=wv, lookup_table=(flux_ext/energy).value)

                    for flt in nrc_filts:
                        bp = get_jwst_filters('NIRCAM,' + flt)
                        kwargs = {'force': 'taper', 'binset': wv}
                        obs = synphot.Observation(sp, bp, **kwargs)
                        mag = obs.effstim('abmag')
                        
                        for l, lum in enumerate(lums):
                            logl = np.log10(lum)
                            scale_mag = mag.value-2.5*logl
                            full_grid[:, i, j, k, l, m, n] = lum*flux_ext
                            
                            mags[flt][i, j, k ,l, m, n] = scale_mag

models = {}
params = (temps, dust_temps, taus, lums, rv, av)

for flt in nrc_filts:
    models[flt] = interpolate.RegularGridInterpolator(params, mags[flt], method='linear', bounds_error=True)

pfile = os.path.join('data', 'interpolate', f'rsg_{comp}_s2_ext.pkl')
pickle.dump(models, open(pfile, 'wb'))

fullparams = (wv.value, temps, dust_temps, taus, lums, rv, av)
fullmodel = interpolate.RegularGridInterpolator(fullparams, 
                                                full_grid, method='linear', bounds_error=True)

pfile = os.path.join('data', 'interpolate', f'rsg_{comp}_s2_fullgrid.pkl')
pickle.dump(fullmodel, open(pfile, 'wb'))