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
import subprocess
 
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
            raise NotImplementedError(f'Model {model} is invalid. "g2", "g10, "s2" and "s10" are currently available')

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

        if masked and len(w)!=5157: 
            mask = (w > 3500.0*u.Angstrom) & (w < 1.0e5*u.Angstrom) 
            kappa = np.loadtxt(self.dustdir+'dust01_trans.dat')
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
    

class dusty_gen(object):
    def __init__(self, 
                 dist=10*u.Mpc, 
                 interp_method='cubic', 
                 rewrite_lambda_grid=False,
                 outdir='/Users/aswin/rsg_phot/data/dusty_grid'):

        self.dustdir = 'data/dust/'
        self.dist = dist

        # This is the total integrated flux of a source with Lum = 1 Lsun in units of
        # erg/s/cm2.  Used to renormalize the RSG spectrum.  This is equivalent to
        # 3.839e33/(4 * pi * (10 * 3.08568025e18)^2) for 1 Lsol at 10 pc.
        self.FLUX_SCALE = ((1 * u.L_sun) / (4 * np.pi * self.dist**2)).to(u.erg/u.s/u.cm**2)

        # Dust-to-gas mass ratio assumption
        self.DUST_TO_GAS = 0.01

        # Read MARCS models
        # RSG models for range of temperatures
        self.rsg_temp = np.array([2600,2800,3000,3200,3300,3400,3500,3600,3700,
                        3800,3900,4000,4250,4500,5000,6000,7000,8000]) * u.K
        self.rsg_data = np.loadtxt(self.dustdir+'data10.dat', unpack=True, dtype=float)
        self.rsg_wavelength = np.loadtxt(self.dustdir+'wavelength.dat')
        self.rsg_wavelength = self.rebin(self.rsg_wavelength, 7748) * u.Angstrom
        self.rsg_10 = interpolate.RegularGridInterpolator((self.rsg_wavelength.value, self.rsg_temp.value), self.rsg_data.T, method=interp_method, bounds_error=False)

        # DUSTY setup 
        self.dusty_basedir = '/Users/aswin/dustyV2' #os.environ['DUSTY_PATH'] #full path
        self.dusty_datadir = outdir # full path
        self.dusty_lambda_grid = list(np.logspace(np.log10(0.01), np.log10(0.6), num = 100)) +\
                                 list(np.logspace(np.log10(0.6), np.log10(4.5), num = 1000)) +\
                                 list(np.logspace(np.log10(4.5), np.log10(15), num = 200)) +\
                                 list(np.logspace(np.log10(15), np.log10(3.6e4), num = 200))
        self.dusty_lambda_grid = np.array(self.dusty_lambda_grid)
        self.validate_dusty_dir(rewrite_lambda_grid)
        self.dust_comp = 'sil'
        self.shell_thickness = 2

    def rebin(self, a, newshape):
        newarray = np.zeros(newshape)
        curr = 0
        for i in np.arange(newshape):
            s = slice(curr, curr+int(a.shape[0]/newshape), 1)
            newarray[i] = np.mean(a[s])
            curr += int(a.shape[0]/newshape)
        return newarray
    
    def get_rsg(self, temp, model='10'):

        wavelength_grid, temp_grid = np.meshgrid(self.rsg_wavelength.value, temp.value, indexing='ij', sparse=True)
        if model == '10': 
            flux = self.rsg_10((wavelength_grid, temp_grid))
            flux = flux.flatten()
        else: 
            raise NotImplementedError('Only model "10" is currently available')

        return flux
    
    def validate_dusty_dir(self, rewrite_lambda=False):
        if rewrite_lambda:
            gridfile = os.path.join(self.dusty_basedir, 'lambda_grid.dat')
            parfile = os.path.join(self.dusty_basedir, 'userpar.inc')
            nline = f'npL = {len(self.dusty_lambda_grid)}\n'

            # edit wavelength grid file
            with open(gridfile, 'w') as f:
                f.write(nline)
            with open(gridfile, 'ab') as f:
                np.savetxt(f, self.dusty_lambda_grid)

            # update user parameter file
            with open(parfile, 'r') as f:
                par_lines = f.readlines()
            par_lines[15] = f'      PARAMETER (npL={len(self.dusty_lambda_grid)})'
            with open(parfile, 'w') as f:
                f.writelines(par_lines)

            # recompile dusty
            subprocess.run(['gfortran', 'dustyV2.f', '-std=legacy', '-o', 'dusty'])

        # make sure all required files are present in the dusty directory
        req_files = ['dusty', 'dustyV2.f', 'userpar.inc', 'lambda_grid.dat', 'dusty.inp']
        for fl_ in req_files:
            if not os.path.exists(os.path.join(self.dusty_basedir, fl_)):
                raise ValueError(f"{fl_} not found in {self.dusty_basedir}")

    def setup_input_spec(self, temp, filedir, redo=False):
        outdir = os.path.join(filedir, 'marcs_spec')
        specfilename = os.path.join(outdir, f'marcs_{temp.value}.dat')
        if redo or not os.path.exists(specfilename):
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # get marcs model flux
            rsg_flux = self.get_rsg(temp, model='10')
            rsg_wv = self.rsg_wavelength.to(u.um)
            spec_input = np.array([rsg_wv.value, rsg_flux]).T

            # write marcs model as input spectrum for dusty
            with open(specfilename, 'w') as f:
                f.write(f'MARCS model atmosphere for T={temp.value} K\n')
                f.write(f'  lambda    L_lambda\n')
                f.write(f' (micron)  (arbitrary)\n')
            with open(specfilename, 'ab') as f:
                np.savetxt(f, spec_input)

        return specfilename

    def setup_dusty(self, filedir, p):
        #p - specfilename (path), temp (in K), dust_temp (in K), dust_comp ('sil' or 'grf'), shell_thickness (float), tau (at 0.55 micron)
        #setup input file for dusty
        outdir = os.path.join(filedir, 'dusty_out', f'rsg_{p['temp'].value}_{p['dust_temp'].value}')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        inp_file = os.path.join(outdir, f'rsg_{p['temp'].value}_{p['dust_temp'].value}.inp')

        with open(inp_file, 'w') as f:
            f.write('  I PHYSICAL PARAMETERS\n')
            f.write('     1) External radiation:\n')
            f.write('                Spectrum = 5\n')
            f.write(f'                {p['input_spec']}\n')
            f.write('     2) Dust Properties\n\n')
            f.write('       2.1 Chemical composition\n')
            f.write('           Optical properties index = 1\n')
            f.write('           Abundances for supported grain types:\n')
            f.write('               Sil-Ow  Sil-Oc  Sil-DL  grf-DL  amC-Hn  SiC-Pg\n')
            if p['dust_comp'].lower() == 'sil' or p['dust_comp'].lower() == 'silicate': # use Draine and Lee silicate dust
                f.write('           x =  0.00    0.00    1.00    0.00    0.00    0.00\n\n')
            elif p['dust_comp'].lower() == 'grf' or p['dust_comp'].lower() == 'graphite': # use Draine and Lee graphite dust
                f.write('           x =  0.00    0.00    0.00    1.00    0.00    0.00\n\n')
            f.write('       2.2 Grain size distribution\n\n')
            f.write('          Size distribution = 2 % arbitrary MRN\n')
            f.write('          q = 3.5, a(min) = 0.005 micron, a(max) = 0.25 micron\n\n')
            f.write('       2.3 Dust temperature on inner boundary:\n\n')
            f.write(f'        - temperature = {p['dust_temp'].value} K\n\n')
            f.write('     3) Density Distribution\n')
            f.write('        - density type = 1\n')
            f.write('        - number of powers = 1\n')
            f.write(f'        - shell\'s relative thickness = {p['shell_thickness']}\n')
            f.write('        - power = 2\n\n')
            f.write('     4) Optical Depth\n')
            f.write('        - grid type = 1\n')
            f.write('        - lambda0 = 0.55 micron\n')
            f.write(f'        - tau(min) = {p['tau'][0]}; tau(max) = {p['tau'][1]}\n')
            f.write(f'        - number of models = {p['tau'][2]}\n\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  II NUMERICS\n\n')
            f.write('     - accuracy for flux conservation = 0.05\n\n')
            f.write('  ----------------------------------------------------------------------\n\n')
            f.write('  III OUTPUT PARAMETERS\n\n')
            f.write('        FILE DESCRIPTION                               FLAG\n')
            f.write('       ------------------------------------------------------------\n')
            f.write('       - verbosity flag;                               verbose = 1\n')
            f.write('       - properties of emerging spectra;             fname.spp = 0\n')
            f.write('       - detailed spectra for each model;           fname.s### = 2\n')
            f.write('       - images at specified wavelengths;           fname.i### = 0\n')
            f.write('       - visibility function at spec. wavelengths;  fname.v### = 0\n')
            f.write('       - radial profiles for each model;            fname.r### = 0\n')
            f.write('       - detailed run-time messages;                fname.m### = 0\n')
            f.write('       -------------------------------------------------------------\n\n')
            f.write('  The end of the input parameters listing.\n')
    
        return inp_file
    
    def scale_flux(self, wv, flux, scale):
        # Renormalize the RSG spectrum so it's in units of erg/s/cm2/angstrom for
        # synphot to interpret
        normalize = simpson(flux, x=wv.to(u.um).value)
        flux = scale * self.FLUX_SCALE * flux/normalize / u.Angstrom

        return flux.value

    def run_dusty(self, tau, lum, temp, dust_temp, dust_comp=None, shell_thickness=None, filedir=None, redo=False):
        curdir = os.getcwd()

        if dust_comp is None:
            dust_comp = self.dust_comp
        if shell_thickness is None:
            shell_thickness = self.shell_thickness
        if filedir is None:
            filedir = self.dusty_datadir

        if not isinstance(tau, list):
            tau_V = [tau, tau, 1]
        elif len(tau) < 3:
            raise ValueError('Input tau list should be of the format [tau_min, tau_max, n_grid]')
        else:
            tau_V = tau

        input_spec = self.setup_input_spec(temp=temp, filedir=filedir, redo=redo)
        p = {'input_spec': input_spec, 
             'temp' : temp, 
             'dust_temp': dust_temp, 
             'dust_comp': dust_comp, 
             'shell_thickness' : shell_thickness,
             'tau' : tau_V}
        inp_file = self.setup_dusty(filedir=filedir, p=p)

        os.chdir(self.dusty_basedir)
        with open('dusty.inp', 'w') as f:
            f.write(f'{inp_file.split('.inp')[0]}')

        subprocess.run(['./dusty'])
        os.chdir(curdir)