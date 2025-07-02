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
 
class dustgen(object):
    def __init__(self, dist=10*u.Mpc, interp_method='cubic'):

        self.dustdir = 'data/dust/'
        self.dist = dist

        # RSG models for range of temperatures
        self.temp = np.array([2600,2800,3000,3200,3300,3400,3500,3600,3700,
                              3800,3900,4000,4250,4500,5000,6000,7000,8000]) * u.K

        # This is the total integrated flux of a source with Lum = 1 Lsun in units of
        # erg/s/cm2.  Used to renormalize the RSG spectrum.  This is equivalent to
        # 3.839e33/(4 * pi * (10 * 3.08568025e18)^2) for 1 Lsol at 10 pc.
        # self.FLUX_SCALE = 3.22398177e-7
        self.FLUX_SCALE = ((1 * u.L_sun) / (4 * np.pi * self.dist**2)).to(u.erg/u.s/u.cm**2)

        # Dust-to-gas mass ratio assumption
        self.DUST_TO_GAS = 0.01

        self.data10 = np.loadtxt(self.dustdir+'data10.dat', unpack=True, dtype=float)

        self.kappa = np.loadtxt(self.dustdir+'dust01_trans.dat')
        self.kappa = self.kappa/np.max(self.kappa)
        self.wavelength = np.loadtxt(self.dustdir+'wavelength.dat')
        self.wavelength = self.rebin(self.wavelength, 7748) * u.Angstrom
        self.rsg_10 = interpolate.RegularGridInterpolator((self.wavelength.value, self.temp.value), self.data10.T, method=interp_method, bounds_error=False)

        # to interpolate
        # xnew, ynew = np.meshgrid(wavelength_array, temp_array, indexing='ij', sparse=True)
        # znew = rsg_10((xnew, ynew))

    def rebin(self, a, newshape):
        newarray = np.zeros(newshape)
        curr = 0
        for i in np.arange(newshape):
            s = slice(curr, curr+int(a.shape[0]/newshape), 1)
            newarray[i] = np.mean(a[s])
            curr += int(a.shape[0]/newshape)
        return newarray
    
    def get_avg2(self, x, p):
        l=x*1.0e-4
        t=p[0]/10.0
        tgra1 = (0.500446 + 1.795729*t - 1.877658*t*t + 0.852820*t**3 - 0.141635*t**4)
        tgra2 = (4.318269 - 14.236698*t + 13.804110*t*t - 5.991369*t**3+0.959539*t**4)/l
        tgra3 = (-5.114167 + 32.462564*t - 26.895305*t*t + 10.197398*t**3 - 1.414338*t**4)/l**2
        tgra4 = (3.384105 - 21.107633*t + 12.229167*t*t - 2.172318*t**3 - 0.149866*t**4)/l**3
        tgra5 = (-1.059677 + 5.553703*t - 1.527415*t*t - 0.881450*t**3+0.391763*t**4)/l**4
        tgra6 = (0.121772 - 0.518968*t - 0.048566*t*t + 0.248002*t**3 - 0.077054*t**4)/l**5
        agraphite2 = (tgra1 + tgra2 + tgra3 + tgra4 + tgra5 + tgra6)*t*l**(-1.375229)

        return agraphite2
    
    def get_avs2(self, x, p):
        l=x*1.0e-4
        t=p[0]/10.0
        tsil1 = (0.437549 - 0.446323*t + 0.648423*t*t - 0.321970*t**3+0.055555*t**4)
        tsil2 = (-0.486741 + 4.034854*t - 5.530127*t*t + 2.711095*t**3 - 0.469112*t**4)/l
        tsil3 = (1.166512 - 12.015845*t + 14.917191*t*t - 7.058630*t**3+1.204954*t**4)/l**2
        tsil4 = (-0.655682 + 13.849514*t - 14.342367*t*t + 6.165157*t**3 - 0.984406*t**4)/l**3
        tsil5 = (0.169689 - 4.956815*t + 4.137525*t*t - 1.419899*t**3+0.176166*t**4)/l**4
        tsil6 = (-0.016829 + 0.582619*t - 0.381166*t*t + 0.083593*t**3 - 0.002153*t**4)/l**5
        asilicate2 = (tsil1 + tsil2 + tsil3 + tsil4 + tsil5 + tsil6)*t*l**(-0.642318)

        return(asilicate2)
    
    def get_avs10(self, x, p):
        l=x*1.0e-4
        t=p[0]/10.0
        tsil1 = (0.197398 - 0.293417*t + 0.192686*t*t - 0.041375*t**3+0.000902*t**4)
        tsil2 = (0.093593 + 2.491030*t - 1.453387*t*t + 0.239280*t**3+0.013273*t**4)/l
        tsil3 = (0.357331 - 6.883382*t + 3.239407*t*t - 0.198182*t**3 - 0.121949*t**4)/l**2
        tsil4 = (0.022567 + 7.169214*t - 1.884278*t*t - 0.728088*t**3+0.309664*t**4)/l**3
        tsil5 = (-0.065599 - 2.173130*t - 0.305525*t*t + 0.839291*t**3 - 0.223389*t**4)/l**4
        tsil6 = (0.012188 + 0.216324*t + 0.133118*t*t - 0.153598*t**3+0.036469*t**4)/l**5
        asilicate10 = (tsil1 + tsil2 + tsil3 + tsil4 + tsil5 + tsil6)*t*l**(-0.323043)

        return(asilicate10)
    
    def get_avg10(self, x, p):
        l=x*1.0e-4
        t=p[0]/10.0
        tgra1 = (0.760499 + 0.879164*t - 0.350748*t*t - 0.039612*t**3+0.034161*t**4)
        tgra2 = (4.061343 - 7.166933*t + 2.791544*t*t + 0.214647*t**3 - 0.233685*t**4)/l
        tgra3 = (-5.133851 + 16.344656*t - 4.283100*t*t - 1.764900*t**3+0.780217*t**4)/l**2
        tgra4 = (3.387184 - 10.066016*t - 1.260999*t*t + 4.103272*t**3 - 1.160204*t**4)/l**3
        tgra5 = (-1.052057 + 2.479576*t + 1.618868*t*t - 2.030708*t**3+0.516503*t**4)/l**4
        tgra6 = (0.120327 - 0.214118*t - 0.293914*t*t + 0.295687*t**3 - 0.071840*t**4)/l**5
        agraphite10 = (tgra1 + tgra2 + tgra3 + tgra4 + tgra5 + tgra6)*t*l**(-1.475236)

        return(agraphite10)

    def get_dust(self, x, p, model='s2'):
        if model=='g2': 
            dust = self.get_avg2(x, p)
        elif model=='g10': 
            dust = self.get_avg10(x, p)
        elif model=='s2': 
            dust = self.get_avs2(x, p)
        elif model=='s10': 
            dust = self.get_avs10(x, p)
        else:
            raise NotImplementedError(f'Model {model} is invallid. "g2", "g10, "s2" and "s10" are currently available')

        dust[np.where(dust < 0)] = 0

        return dust

    def get_rsg(self, scale, temp, model='10'):

        wavelength_grid, temp_grid = np.meshgrid(self.wavelength.value, temp.value, indexing='ij', sparse=True)
        if model == '10': 
            flux = self.rsg_10((wavelength_grid, temp_grid))
            flux = flux.flatten()
        else: 
            raise NotImplementedError('Only model "10" is currently available')
        normalize = simpson(flux, x=wavelength_grid.flatten())
        # Renormalize the RSG spectrum so it's in units of erg/s/cm2/angstrom for
        # synphot to interpret
        flux = scale * self.FLUX_SCALE * flux/normalize / u.Angstrom

        return flux

    def get_ext_bb(self, p, rsg_model='10', dust_model='g2', sptype='all', masked=True):

        w = self.wavelength
        flux = self.get_rsg(p[1], p[2], model=rsg_model)

        if masked and len(w)!=5157: #5157?
            mask = (w > 3500.0*u.Angstrom) & (w < 1.0e5*u.Angstrom) #why limits?
            kappa = np.loadtxt(self.dustdir+'dust01_trans.dat') #should this be normalized?
            w = w[mask]
            flux = flux[mask]
            kappa = self.kappa[mask]

        if sptype=='i' or sptype=='intrinsic':
            sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=flux)
            return sp

        a_dust = self.get_dust(w.value, p, model=dust_model)
        obsflux = flux * 10**(-0.4*a_dust)
        bb_scale = simpson(flux-obsflux, x=w)

        if sptype=='s' or sptype=='star':
            sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=obsflux)
            return sp 

        # bb = kappa * (w.value*1.0e-5)**-5 * 1.0/(np.exp(143843215.0/(w.value*p[3].value))-1.0) * u.erg/u.s/u.cm**2/u.Angstrom
        bb = kappa * ((8 * np.pi**2 * const.h * const.c**2) / (w**5)) * (1 / (np.exp(const.h * const.c / (w * const.k_B * p[3])) - 1))
        bb = bb.to(u.erg/u.s/u.cm**2/u.Angstrom)
        # We also need to renormalize the bb flux.  The fraction of total luminosity
        # in the bb part of the spectrum is "scale".  So here we can simply multiply
        # by p[1] * FLUX_SCALE
        normalize_bb = simpson(bb, x=w)
        bb = bb_scale * bb / normalize_bb
        energy = (const.h*const.c/w).to(u.erg)

        if sptype=='b' or sptype=='bb' or sptype=='dust' or sptype=='blackbody':
            sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=(bb/energy).value)
        elif sptype=='ss' or sptype=='scaled_star':
            sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=(obsflux/energy).value)
        else:
            sp = synphot.SourceSpectrum(Empirical1D, points=w, lookup_table=((obsflux + bb)/energy).value)

        if sptype=='all':
            Lsol = simpson(sp(sp.waveset)*energy, x=sp.waveset) * 4 * np.pi * (self.dist.to(u.cm).value)**2/((1*u.L_sun).cgs.value)
            try:
                assert np.abs((Lsol-p[1])/p[1]) < 3.0e-2
            except AssertionError:
                print(p)
                print(Lsol)
                print(np.abs((Lsol-p[1])/p[1]))
                print(rsg_model, dust_model, masked)
                sys.exit()

        return sp