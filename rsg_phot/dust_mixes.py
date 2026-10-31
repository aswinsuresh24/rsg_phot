'''
Default dust mixes for the SN 2026sqf DUSTY grids: photosphere, grain composition and size distribution, by name.
dust.py and dust_twoshell.py build a grid from one (--mix), and mcmc_nest.py fits that mix's grid (--mix).

Abundances are DUSTY's (by volume, normalised over the built-in and .nk grains together). .nk files are looked up as
given, then in $DUSTY_PATH/Lib_nk, then in data/optool_lnk (dusty_likelihood.find_nk_file).

Both presets share one grain-size distribution, the O-rich GRAMS one (Sargent+11, after Sargent+10): KMH
n(a) ~ a^-3.5 exp(-a/a0) with a_min 0.01 um, a0 0.1 um. It fits SN 2026sqf best of those tried for both mixes
(MRN 0.005/0.01-0.25 and 0.01-0.1, KMH a0 0.25/0.5/1: 2026sqf_tmp/carbon_kmh_check.py, silicate_amin_check.py,
silicate_kmh_check.py). Grids built before this (shell_amC_SiC20*.npz, twoshell_amC_SiC20.npz) are MRN 0.005-0.25.
'''
import numpy as np

KMH = {'q': 3.5, 'amin': 0.01, 'a0': 0.1}

MIXES = {
    # C-rich AGB: COMARCS ([Fe/H] -0.5, 2 Msun, C/O 1.4) + 80% amC (amC-zb2.nk, Zubko+96) / 20% SiC (SiC-Pg). The GRAMS carbon
    # grid's a0 = 1 um (Srinivasan+11) is disfavoured here (chi2 34 vs 11), as is a0 0.5 (18); a0 0.25 fits as well as 0.1
    'carbon': dict(tag='amC_SiC20_kmh', modeldir='data/marcs/comarcs', logz=-0.5, mass=2.0, c_o=1.4,
                   comp={'sic': 0.2}, nk_comp={'amC-zb2.nk': 0.8}, size_dist='kmh', size_par=KMH,
                   temps=np.arange(2600., 4400. + 1e-6, 200.), teff=4000.),
    # O-rich AGB/RSG (Ventura+12, Dell'Agli+17): MARCS ([Fe/H] -0.5, log g 0) + 85% Sil-Ow (Ossenkopf+92) / 15% corundum
    # (Koike+95) by mass, i.e. 0.8729 / 0.1271 by volume (densities 3.3, 4.0), as composite grains at one temperature
    # (core-mantle check: 2026sqf_tmp/sagb_coremantle.py). Teff on the MARCS nodes, which skip 2900-3200 K
    'silicate': dict(tag='SilOw_cor15_kmh', modeldir='data/marcs/m1.00_g0.00', logz=-0.5, mass=1.0, c_o=None,
                     comp={'Sil-Ow': 0.8729}, nk_comp={'cor-c-Koike1995.nk': 0.1271}, size_dist='kmh', size_par=KMH,
                     temps=np.array([2600., 2800., 3400., 3600., 3800., 4000., 4250., 4500., 4750., 5000.]), teff=4500.),
}

def builder_kwargs(name, **overrides):
    '''
    dusty_likelihood (or twoshell_likelihood) keyword arguments for the photosphere and grains of mix name. Overrides
    that are not None replace the preset values, e.g. builder_kwargs('carbon', c_o=2.0).
    '''
    m = MIXES[name]
    kw = dict(modeldir=m['modeldir'], logg=0.0, logz=m['logz'], mass=m['mass'], c_o=m['c_o'], comp=dict(m['comp']),
              nk_comp=dict(m['nk_comp']), size_dist=m['size_dist'], size_par=dict(m['size_par']))
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return kw

def grid_file(name, kind='shell'):
    # where the fitted grid of mix name lives (kind 'shell' from dust.py, 'twoshell' from dust_twoshell.py)
    return f'data/dusty_grids/2026sqf/{kind}_{MIXES[name]["tag"]}.npz'
