import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
import palettable 
from matplotlib.ticker import MaxNLocator
from scipy import optimize
from astropy import constants as const
import astropy.units as u 
from SpectraTools.fit_line import doppler2wave
import os 
import sys 
from lmfit import Parameters
import cPickle as pickle 
from lmfit.models import GaussianModel
from SpectraTools.fit_line import doppler2wave, wave2doppler, PseudoContinuum
from lmfit import Model
from barak import spec

set_plot_properties() # change style 
cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

def mfica_component_weights():

    xs = [np.arange(0.1, 0.6, 0.01),
          np.arange(0.1, 0.6, 0.01),
          np.arange(0.0, 0.4, 0.01),
          np.arange(0.0, 0.15, 0.004),
          np.arange(0.0, 0.15, 0.004),
          np.arange(0.0, 0.15, 0.004)]

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.mfica_flag == 1]

    col_list = ['mfica_w1',
                'mfica_w2',
                'mfica_w3',
                'mfica_w4',
                'mfica_w5',
                'mfica_w6']

    titles = [r'$w_1$',
              r'$w_2$',
              r'$w_3$',
              r'$w_4$',
              r'$w_5$',
              r'$w_6$']
    
    fname = '/data/vault/phewett/ICAtest/DR12exp/Spectra/hbeta_2154_c10.weight'
    t = np.genfromtxt(fname)   

    fig, axs = plt.subplots(3, 2, figsize=figsize(1, vscale=1.2))

    
    for i, ax in enumerate(axs.reshape(-1)):
          
        w_norm = df[col_list[i]] / df[col_list[:6]].sum(axis=1) # sum positive components 
        w_norm = w_norm[~np.isnan(w_norm) & ~np.isinf(w_norm)]
    
        hist = ax.hist(w_norm,
                       normed=True,
                       bins=xs[i],
                       histtype='step',
                       color=cs[1],
                       zorder=1)
        
        w_norm = t[:, i] / np.sum(t[:, :6], axis=1) # sum positive components 
    
        hist = ax.hist(w_norm,
                       normed=True,
                       bins=xs[i],
                       histtype='step',
                       color=cs[8], 
                       zorder=0)   

        ax.set_yticks([]) 
        ax.get_xaxis().tick_bottom()
        ax.set_title(titles[i])
        ax.xaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, left=0.05)

    fig.text(0.50, 0.03, r"$\displaystyle\frac{w_i}{\sum_{i=1}^6 w_i}$", ha='center')
    fig.text(0.02, 0.6, 'Normalised counts', rotation=90)

    fig.savefig('/home/lc585/thesis/figures/chapter04/mfica_component_weights.pdf')

    plt.show() 

    return None 


def redshift_comparison(): 

    fig, axs = plt.subplots(3, 1, figsize=figsize(0.6, 2))

 
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 == 1]
    df = df[df.OIII_FIT_HB_Z_FLAG == 1]

    xi = const.c.to(u.km/u.s)*(df.OIII_FIT_Z_FULL_OIII_PEAK - df.OIII_FIT_HB_Z)/(1.0 + df.OIII_FIT_Z_FULL_OIII_PEAK)

    axs[0].hist(xi,
                histtype='stepfilled',
                color=cs[1],
                bins=np.arange(-1000, 1000, 100),
                zorder=1,
                normed=True)

    def gaussian(mu, sig, x):
        return (2.0 * np.pi * sig**2)**-0.5 * np.exp(-(x - mu)**2 / (2.0*sig**2))

    def log_likelihood(p, x):
        return np.sum(np.log(gaussian(p[0], p[1], x.value) ))

 
    min_func = lambda p: -log_likelihood(p, xi)
    p_fit = optimize.fmin(min_func, x0=[0.0, 200.0])

    axs[0].plot(np.arange(-1000, 1000, 1), 
                gaussian(p_fit[0], p_fit[1], np.arange(-1000, 1000, 1)),
                color=cs[0])

    axs[0].axvline(0.0, color='black', linestyle='--')

    axs[0].text(0.05, 0.9, r'$\mu = {0:.0f}$'.format(p_fit[0]),
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[0].transAxes)

    axs[0].text(0.05, 0.82, r'$\sigma = {0:.0f}$'.format(p_fit[1]),
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[0].transAxes)

    #------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 == 1]
    df = df[df.OIII_FIT_HA_Z_FLAG == 1]

    xi = const.c.to(u.km/u.s)*(df.OIII_FIT_Z_FULL_OIII_PEAK - df.OIII_FIT_HA_Z)/(1.0 + df.OIII_FIT_Z_FULL_OIII_PEAK)

    axs[1].hist(xi,
                histtype='stepfilled',
                color=cs[1],
                bins=np.arange(-1000, 1000, 100),
                zorder=1,
                normed=True)

    min_func = lambda p: -log_likelihood(p, xi)
    p_fit = optimize.fmin(min_func, x0=[0.0, 200.0])

    axs[1].plot(np.arange(-1000, 1000, 1), 
                gaussian(p_fit[0], p_fit[1], np.arange(-1000, 1000, 1)),
                color=cs[0])

    axs[1].axvline(0.0, color='black', linestyle='--')

    axs[1].text(0.05, 0.9, r'$\mu = {0:.0f}$'.format(p_fit[0]),
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[1].transAxes)

    axs[1].text(0.05, 0.82, r'$\sigma = {0:.0f}$'.format(p_fit[1]),
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[1].transAxes)

    #-------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FIT_HB_Z_FLAG == 1]
    df = df[df.OIII_FIT_HA_Z_FLAG == 1]

    xi = const.c.to(u.km/u.s)*(df.OIII_FIT_HB_Z - df.OIII_FIT_HA_Z)/(1.0 + df.OIII_FIT_HB_Z)

    axs[2].hist(xi,
                histtype='stepfilled',
                color=cs[1],
                bins=np.arange(-1000, 1000, 100),
                zorder=1,
                normed=True)

    min_func = lambda p: -log_likelihood(p, xi)
    p_fit = optimize.fmin(min_func, x0=[0.0, 200.0])

    axs[2].plot(np.arange(-1000, 1000, 1), 
                gaussian(p_fit[0], p_fit[1], np.arange(-1000, 1000, 1)),
                color=cs[0])

    axs[2].axvline(0.0, color='black', linestyle='--')

    axs[2].text(0.05, 0.9, r'$\mu = {0:.0f}$'.format(p_fit[0]),
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[2].transAxes)

    axs[2].text(0.05, 0.82, r'$\sigma = {0:.0f}$'.format(p_fit[1]),
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[2].transAxes)

    #----------------------------------------------------------


    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])

    axs[0].xaxis.set_ticks_position('bottom')
    axs[1].xaxis.set_ticks_position('bottom')
    axs[2].xaxis.set_ticks_position('bottom')

    axs[0].set_xlabel(r'$c(z_{[{\rm OIII}]} - z_{{\rm H}\beta}) / (1 + z_{[{\rm OIII}]})$ [km~$\rm{s}^{-1}$]')
    axs[1].set_xlabel(r'$c(z_{[{\rm OIII}]} - z_{{\rm H}\alpha}) / (1 + z_{[{\rm OIII}]})$ [km~$\rm{s}^{-1}$]')
    axs[2].set_xlabel(r'$c(z_{{\rm H}\beta} - z_{{\rm H}\alpha}) / (1 + z_{{\rm H}\beta})$ [km~$\rm{s}^{-1}$]')



    # # ax.get_xaxis().tick_bottom()

    # # ax.set_xlabel(r'$\Delta z / (1 + z)$ [km~$\rm{s}^{-1}$]')
    
    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter04/redshift_comparison.pdf')

    plt.show() 

    return None 




def bal_hists():

    fig, axs = plt.subplots(2, 1, figsize=figsize(0.7, vscale=1.4))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 == 1]
    df = df[df.BAL_FLAG != 1]
    
    s = axs[0].hist(df.OIII_5007_W80, 
                    normed=True, 
                    histtype='stepfilled', 
                    edgecolor='None',
                    facecolor=cs[1],
                    bins=np.arange(500, 3500, 300),
                    cumulative=False)

    s = axs[1].hist(-df.OIII_5007_V10_CORR, 
                    normed=True, 
                    histtype='stepfilled', 
                    facecolor=cs[1],
                    edgecolor='None',
                    bins=np.arange(-500, 4000, 500),
                    cumulative=False)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 == 1]
    df = df[df.BAL_FLAG == 1]
    
    s = axs[0].hist(df.OIII_5007_W80, 
                    normed=True, 
                    histtype='step', 
                    edgecolor='black',
                    bins=np.arange(500, 3500, 300),
                    cumulative=False)

    s = axs[1].hist(-df.OIII_5007_V10_CORR, 
                    normed=True, 
                    histtype='step', 
                    edgecolor='black',
                    bins=np.arange(-500, 4000, 500),
                    cumulative=False)

    axs[0].set_yticks([])
    axs[1].set_yticks([])

    axs[0].set_xlabel(r'$w_{80}$ [km~$\rm{s}^{-1}$]')
    axs[1].set_xlabel(r'$v_{10}$ [km~$\rm{s}^{-1}$]')
    
    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter04/bal_hists.pdf')

    plt.show()

    return None 

def civ_blueshift_oiii_strength():

    set_plot_properties() # change style 

    fig, ax = plt.subplots(figsize=figsize(0.9, vscale=0.7))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[(df.mfica_flag == 1)]
    df = df[df.WARN_CIV_BEST == 0]
    
    df.dropna(subset=['Median_CIV_BEST', 'OIII_5007_MFICA_W80'], inplace=True)
    
    w0 = np.mean([1548.202,1550.774])*u.AA  
    median_wav = doppler2wave(df.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df.z_ICA_FIT)) / w0

    col_list = ['mfica_w1',
                'mfica_w2',
                'mfica_w3',
                'mfica_w4',
                'mfica_w5',
                'mfica_w6',
                'mfica_w7',
                'mfica_w8',
                'mfica_w9',
                'mfica_w10']

    w_norm = df[col_list[3:5]].sum(axis=1) / df[col_list[:6]].sum(axis=1) # sum positive components 
    w_norm = w_norm[~np.isnan(w_norm) & ~np.isinf(w_norm)]
    
    from LiamUtils import colormaps as cmaps
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)


    im = ax.scatter(blueshift_civ,
                    w_norm,
                    c=df.LogL5100,
                    edgecolor='None',
                    zorder=2,
                    s=30)    

    
    cb = fig.colorbar(im)
    cb.set_label(r'log L$_{5100{\rm \AA}}$')

    # ax.scatter(blueshift_civ,
    #           w_norm,
    #           facecolor=cs[1], 
    #           edgecolor='None',
    #           zorder=2,
    #           s=30)

    ax.set_xlim(-1500, 6000)
    ax.set_ylim(0.03, 0.4)

    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r"$\displaystyle {(w_4 + w_5)} / {\sum_{i=1}^6 w_i}$")

    ax.set_yscale('log')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/civ_blueshift_oiii_strength.pdf')

    plt.show() 

    return None 


def oiii_core_strength_blueshift():

    set_plot_properties() # change style 

    fig, ax = plt.subplots(figsize=figsize(0.9, vscale=0.7))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[(df.mfica_flag == 1)]
    
    df.dropna(subset=['OIII_5007_MFICA_W80'], inplace=True)
    
    col_list = ['mfica_w1',
                'mfica_w2',
                'mfica_w3',
                'mfica_w4',
                'mfica_w5',
                'mfica_w6',
                'mfica_w7',
                'mfica_w8',
                'mfica_w9',
                'mfica_w10']

    w_norm1 = df[col_list[3:5]].sum(axis=1) / df[col_list[:6]].sum(axis=1) # sum positive components 
    w_norm1 = w_norm1[~np.isnan(w_norm1) & ~np.isinf(w_norm1)]

    w_norm2 = df[col_list[5]] / df[col_list[3:5]].sum(axis=1) 
    w_norm2 = w_norm2[~np.isnan(w_norm2) & ~np.isinf(w_norm2)]
    
    im = ax.scatter(w_norm1,
                    w_norm2,
                    edgecolor='None',
                    facecolor=cs[1],
                    zorder=2,
                    s=10)    

    ax.set_xlim(0.025, 0.4)
    ax.set_ylim(0, 3)

    # ax.set_yscale('log')

    ax.set_xlabel(r"$\displaystyle {(w_4 + w_5)} / {\sum_{i=1}^6 w_i}$")
    ax.set_ylabel(r"$\displaystyle w_6 / {(w_4 + w_5)}$")

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/oiii_core_strength_blueshift.pdf')

    plt.show() 

    return None 

def compare_gaussian_ica(name):


    """
    Compare ICA to Gaussian fits for QSO538, where Gaussians do a very bad job

    Need to fix this if used different redshifts in ICA and multigaussian fits

    """

    set_plot_properties() # change style 

    import sys
    sys.path.append('/home/lc585/Dropbox/IoA/nirspec/python_code')
    from get_nir_spec import get_nir_spec
    from SpectraTools.fit_line import make_model_mfica, mfica_model, mfica_get_comp
    from SpectraTools.fit_line import PLModel

    cs_light = palettable.colorbrewer.qualitative.Pastel1_9.mpl_colors

    #---------------------------------------------------------------
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    
    #-----------------------------------------------------------------------------

    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, 'OIII')
    z = df.loc[name, 'z_IR_OIII_FIT'] # be careful if this changes 
    w0=4862.721*u.AA
    
    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params = Parameters()
    params.load(parfile)
    parfile.close()
    
    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()
    
    parfile = open(os.path.join(save_dir,'my_params_bkgd.txt'), 'r')
    params_bkgd = Parameters()
    params_bkgd.load(parfile)
    parfile.close()
        
    # Get data -------------------------------------------------------------------

    if os.path.exists(df.loc[name, 'NIR_PATH']):
        wav, dw, flux, err = get_nir_spec(df.loc[name, 'NIR_PATH'], df.loc[name, 'INSTR'])
    
    wav =  wav / (1.0 + z)
    spec_norm = 1.0 / np.median(flux[(flux != 0.0) & ~np.isnan(flux)])
    flux = flux * spec_norm 
    err = err * spec_norm 
    
    oiii_region = (wav > 4700) & (wav < 5100) & (err > 0.0)
    
    wav = wav[oiii_region]
    flux = flux[oiii_region]
    err = err[oiii_region]

     
    #-----------------------------------------------
    mod = GaussianModel(prefix='oiii_4959_n_')
    
    mod += GaussianModel(prefix='oiii_5007_n_')
    
    mod += GaussianModel(prefix='oiii_4959_b_')
    
    mod += GaussianModel(prefix='oiii_5007_b_')
    
    if df.loc[name, 'HB_NARROW'] is True: 
        mod += GaussianModel(prefix='hb_n_')  
    
    for i in range(df.loc[name, 'HB_NGAUSSIANS']):
        mod += GaussianModel(prefix='hb_b_{}_'.format(i))  

    flux_gaussians = mod.eval(params=params, x=wave2doppler(wav*u.AA, w0=w0).value/sd)

    #---------------------------------------------------
    
    if df.loc[name, 'OIII_SUBTRACT_FE']:
        bkgdmod = Model(PseudoContinuum, 
                        param_names=['amplitude',
                                     'exponent',
                                     'fe_norm',
                                     'fe_sd',
                                     'fe_shift'], 
                        independent_vars=['x']) 
    else: 
        bkgdmod = Model(PLModel, 
                        param_names=['amplitude','exponent'], 
                        independent_vars=['x'])     


    fname = os.path.join('/home/lc585/SpectraTools/irontemplate.dat')
    fe_wav, fe_flux = np.genfromtxt(fname, unpack=True)
    fe_flux = fe_flux / np.median(fe_flux)
    sp_fe = spec.Spectrum(wa=10**fe_wav, fl=fe_flux)

    bkgd_gaussians = bkgdmod.eval(params=params_bkgd, x=wav, sp_fe=sp_fe)

    params_bkgd_tmp = params_bkgd.copy()
    params_bkgd_tmp['fe_sd'].value = 1.0 
    bkgd_gaussians_noconvolve = bkgdmod.eval(params=params_bkgd_tmp, x=wav, sp_fe=sp_fe)

    #-------------------------------------------------------------------
   
        
    """
    Take out slope - to compare to MFICA fit 
    """

    bkgdmod_mfica = Model(PLModel, 
                          param_names=['amplitude','exponent'], 
                          independent_vars=['x']) 

    
    fname = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, 'MFICA', 'my_params_remove_slope.txt')

    parfile = open(fname, 'r')
    params_bkgd_mfica = Parameters()
    params_bkgd_mfica.load(parfile)
    parfile.close()

    comps_wav, comps, weights = make_model_mfica(mfica_n_weights=10)

    for i in range(10): 
        weights['w{}'.format(i+1)].value = df.ix[name, 'mfica_w{}'.format(i+1)]

    weights['shift'].value = df.ix[name, 'mfica_shift']

    flux_mfica = mfica_model(weights, comps, comps_wav, wav) * bkgdmod_mfica.eval(params=params_bkgd_mfica, x=wav)


    #-----------------------------------------------------------------

   
    fig, ax = plt.subplots(figsize=figsize(0.9, vscale=0.8)) 

    ax.plot(wav, flux_gaussians + bkgd_gaussians, color=cs[0], zorder=1)
    ax.plot(wav, bkgd_gaussians, color=cs[1], zorder=5)
    ax.plot(wav, bkgd_gaussians_noconvolve, color=cs_light[1], zorder=5)
    ax.plot(wav, flux, color=cs[8], zorder=0)

    g1 = GaussianModel()
    p1 = g1.make_params()

    p1['center'].value = params['oiii_5007_n_center'].value
    p1['sigma'].value = params['oiii_5007_n_sigma'].value
    p1['amplitude'].value = params['oiii_5007_n_amplitude'].value

    ax.plot(wav, 
            g1.eval(params=p1, x=wave2doppler(wav*u.AA, w0=w0).value/sd) + bkgd_gaussians,
            c=cs_light[4],
            linestyle='-')

    g1 = GaussianModel()
    p1 = g1.make_params()

    p1['center'].value = params['oiii_4959_n_center'].value
    p1['sigma'].value = params['oiii_4959_n_sigma'].value
    p1['amplitude'].value = params['oiii_4959_n_amplitude'].value

    ax.plot(wav, 
            g1.eval(params=p1, x=wave2doppler(wav*u.AA, w0=w0).value/sd) + bkgd_gaussians,
            c=cs_light[4],
            linestyle='-')

    g1 = GaussianModel()
    p1 = g1.make_params()

    p1['center'].value = params['oiii_5007_b_center'].value
    p1['sigma'].value = params['oiii_5007_b_sigma'].value
    p1['amplitude'].value = params['oiii_5007_b_amplitude'].value

    ax.plot(wav, 
            g1.eval(params=p1, x=wave2doppler(wav*u.AA, w0=w0).value/sd) + bkgd_gaussians,
            c=cs_light[4],
            linestyle='-')    

    g1 = GaussianModel()
    p1 = g1.make_params()

    p1['center'].value = params['oiii_4959_b_center'].value
    p1['sigma'].value = params['oiii_4959_b_sigma'].value
    p1['amplitude'].value = params['oiii_4959_b_amplitude'].value   

    ax.plot(wav, 
            g1.eval(params=p1, x=wave2doppler(wav*u.AA, w0=w0).value/sd) + bkgd_gaussians,
            c=cs_light[4],
            linestyle='-')        

    for i in range(df.loc[name, 'HB_NGAUSSIANS']):

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['hb_b_{}_center'.format(i)].value
        p1['sigma'].value = params['hb_b_{}_sigma'.format(i)].value
        p1['amplitude'].value = params['hb_b_{}_amplitude'.format(i)].value  

        ax.plot(wav, 
                g1.eval(params=p1, x=wave2doppler(wav*u.AA, w0=w0).value/sd) + bkgd_gaussians,
                c=cs_light[4],
                linestyle='-')



    # ax.plot(wav, 
    #         flux_mfica, 
    #         color=cs[1], 
    #         lw=1,
    #         zorder=2)

    ax.set_xlabel(r'Wavelength [\AA]') 
    ax.set_ylabel(r'$F_{\lambda}$ [Arbitrary units]')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/compare_gaussian_ica_' + name + '.pdf')


    plt.show()

    return None 


def snr_test():

    from get_errors_oiii import get_errors_oiii
    import collections 

    cs = palettable.colorbrewer.sequential.Blues_3.mpl_colors  
    set1 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)

    snrs = ['2p5', '5p0', '7p5', '15p0', '20p0', '50p0']    
    
    new_dict = collections.OrderedDict() 

    fig, axs = plt.subplots(2, 1, figsize=figsize(0.9,  vscale=(np.sqrt(5.0)-1.0)), sharex=True)

    for i, name in enumerate(['QSO559', 'QSO004']): 

        for snr in snrs:                   
    
            # returns dictionary of dictionaries 
            new_dict.update({snr: get_errors_oiii(name + '_snr_' + snr, plot=False, snr_test=True)})
    
        p50 = np.array([x['oiii_5007_w80']['p50'] for x in new_dict.values()])
        p16 = np.array([x['oiii_5007_w80']['p16'] for x in new_dict.values()])
        p84 = np.array([x['oiii_5007_w80']['p84'] for x in new_dict.values()])
    
        lower_error = p50 - p16
        upper_error = p84 - p50 
    
        ytrue = new_dict['50p0']['oiii_5007_w80']['p50'] 
    
        axs[i].errorbar([2.5, 5, 7.5, 15, 20, 50],
                    p50 / ytrue, 
                    yerr=[lower_error / ytrue, upper_error / ytrue],
                    color='black')
        
        axs[i].axhline(1.0, color='black', linestyle='--')
        
        axs[i].set_xlim(0, 55) 
        axs[i].set_ylim(0.6, 1.4)
    
        axs[i].axhspan(0.9, 1.1, color=cs[1])
        # axs[i].axhspan(0.8, 0.9, color=cs[0])
        # axs[i].axhspan(1.1, 1.2, color=cs[0])
    
        axs[i].axvline(df.ix[name].OIII_FIT_SNR_CONTINUUM, color=set1[0], linestyle='--')
    
        axs[i].set_ylabel(r'$\Delta w_{80}$')
        axs[i].set_yticks([0.6, 0.8, 1, 1.2, 1.4])

    axs[1].set_xlabel('S/N')

    fig.tight_layout()

    fig.subplots_adjust(hspace=0.05)

    fig.savefig('/home/lc585/thesis/figures/chapter04/snr_test.pdf') 

    plt.show()



    return None 


def oiii_strength_hist():

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.mfica_flag == 1]


    fig, ax = plt.subplots(figsize=figsize(0.8, vscale=0.8))

    ax.hist(df.oiii_strength,
            bins=np.arange(0, 0.5, 0.025),
            facecolor=cs[1],
            histtype='stepfilled')
    ax.axvline(0.1165, color=cs[0], linestyle='--')

    ax.set_xlabel(r"$\displaystyle {\sum_{i=3}^6 w_i} / {\sum_{i=1}^6 w_i}$")
    ax.set_ylabel(r"Count")

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/oiii_strength_hist.pdf') 

    plt.show() 


    return None 