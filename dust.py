import pandeia.engine
import stsynphot
import synphot
from synphot import SpectralElement
from synphot.models import Empirical1D
import os, glob, sys
import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.io import ascii
from scipy import interpolate
from scipy.integrate import simpson

dustdir = 'data/dust/'

# RSG models for range of temperatures
temp=[2600,2800,3000,3200,3300,3400,3500,3600,3700,3800,3900,4000,
            4250,4500,5000,6000,7000,8000]
temp = np.array(temp)

# This is the total integrated flux of a source with Lum = 1 Lsun in units of
# erg/s/cm2.  Used to renormalize the RSG spectrum.  This is equivalent to
# 3.839e33/(4 * pi * (10 * 3.08568025e18)^2) for 1 Lsol at 10 pc.
FLUX_SCALE = 3.22398177e-7

# Dust-to-gas mass ratio assumption
DUST_TO_GAS = 0.01

data10 = np.loadtxt(dustdir+'data10.dat', unpack=True, dtype=float)

kappa = np.loadtxt(dustdir+'dust01_trans.dat')
kappa = kappa/np.max(kappa)
wavelength = np.loadtxt(dustdir+'wavelength.dat')

def rebin(a, newshape):
    newarray = np.zeros(newshape)
    curr = 0
    for i in np.arange(newshape):
        s = slice(curr, curr+int(a.shape[0]/newshape), 1)
        newarray[i] = np.mean(a[s])
        curr += int(a.shape[0]/newshape)
    return(newarray)

wavelength = rebin(wavelength, 7748)
rsg_10 = interpolate.RegularGridInterpolator((wavelength, temp), data10.T, method='cubic', bounds_error=False)

# to interpolate
# xnew, ynew = np.meshgrid(wavelength_array, temp_array, indexing='ij', sparse=True)
# znew = rsg_10((xnew, ynew))

def get_avg2(x, p):
    l=x*1.0e-4
    t=p/10.0
    tgra1 = (0.500446 + 1.795729*t - 1.877658*t*t + 0.852820*t**3 - 0.141635*t**4)
    tgra2 = (4.318269 - 14.236698*t + 13.804110*t*t - 5.991369*t**3+0.959539*t**4)/l
    tgra3 = (-5.114167 + 32.462564*t - 26.895305*t*t + 10.197398*t**3 - 1.414338*t**4)/l**2
    tgra4 = (3.384105 - 21.107633*t + 12.229167*t*t - 2.172318*t**3 - 0.149866*t**4)/l**3
    tgra5 = (-1.059677 + 5.553703*t - 1.527415*t*t - 0.881450*t**3+0.391763*t**4)/l**4
    tgra6 = (0.121772 - 0.518968*t - 0.048566*t*t + 0.248002*t**3 - 0.077054*t**4)/l**5
    agraphite2 = (tgra1 + tgra2 + tgra3 + tgra4 + tgra5 + tgra6)*t*l**(-1.375229)

    return(agraphite2)

def get_dust(x, p, model='g2'):

    if model=='g2': 
        dust = get_avg2(x, p)
    else:
        raise NotImplementedError('Only model "g2" is currently available')

    dust[np.where(dust < 0)] = 0

    return(dust)

def get_rsg(scale, temp, model='10'):

    wavelength_grid, temp_grid = np.meshgrid(wavelength, temp, indexing='ij', sparse=True)
    if model == '10': 
        flux = rsg_10((wavelength_grid, temp_grid))
        flux = flux.flatten()
    else: 
        raise NotImplementedError('Only model "10" is currently available')

    normalize = simpson(flux, x=wavelength)
    # Renormalize the RSG spectrum so it's in units of erg/s/cm2/angstrom for
    # synphot to interpret
    flux = scale * FLUX_SCALE * flux/normalize

    return(flux)

def get_ext_bb(p, rsg_model='10', dust_model='g2', sptype='all', masked=True):

    w = wavelength
    flux = get_rsg(p[1], p[2], model=rsg_model)

    if masked and len(w)!=5157: #5157?
        mask = (wavelength > 3500.0) & (wavelength < 1.0e5) #why upper limit?
        kappa = np.loadtxt(dustdir+'dust01_trans.dat')
        w = w[mask]
        flux = flux[mask]
        kappa = kappa[mask]

    if sptype=='i' or sptype=='intrinsic':
        sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=flux)
        return(sp)

    a_dust = get_dust(w, p[0], model=dust_model)
    obsflux = flux * 10**(-0.4*a_dust)
    bb_scale = simpson(flux-obsflux, x=w)

    if sptype=='s' or sptype=='star':
        sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=obsflux)
        return(sp)

    bb = kappa * (w*1.0e-5)**-5 * 1.0/(np.exp(143843215.0/(w*p[3]))-1.0)
    # We also need to renormalize the bb flux.  The fraction of total luminosity
    # in the bb part of the spectrum is "scale".  So here we can simply multiply
    # by p[1] * FLUX_SCALE
    normalize_bb = simpson(bb, x=w)
    bb = bb_scale * bb / normalize_bb

    if sptype=='b' or sptype=='bb' or sptype=='dust' or sptype=='blackbody':
        sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=bb)
    elif sptype=='ss' or sptype=='scaled_star':
        sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=obsflux)
    else:
        sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=obsflux + bb)
        mask = sp(sp.waveset) < 0

    if sptype=='all':
        Lsol = simpson(sp(sp.waveset), x=sp.waveset)*4*np.pi*(10*3.08568025e18)**2/(3.839e33)
        try:
            assert np.abs((Lsol-p[1])/p[1]) < 3.0e-2
        except AssertionError:
            print(p)
            print(Lsol)
            print(np.abs((Lsol-p[1])/p[1]))
            print(rsg_model, dust_model, masked)
            sys.exit()

    return(sp)