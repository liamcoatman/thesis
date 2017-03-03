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
import matplotlib.gridspec as gridspec
from astropy.table import Table 
from scipy import stats 
from lmfit.models import GaussianModel, LorentzianModel, PowerLawModel, ConstantModel, LinearModel
from scipy.interpolate import interp1d 
import numpy.ma as ma 

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

    fig.savefig('/home/lc585/thesis/figures/chapter04/mfica_component_weights.pdf')

    plt.show() 

    return None 


def redshift_comparison(): 

    fig, axs = plt.subplots(3, 1, figsize=figsize(1, 2))

 
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_SNR_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    
    df = df[df.OIII_FIT_HB_Z_FLAG == 1] # need to relax this 

    print len(df)

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

    axs[0].text(0.9, 0.9, '(a)',
                horizontalalignment='center',
                verticalalignment='center',
                transform = axs[0].transAxes)

    #------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_SNR_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]

    df = df[df.OIII_FIT_HA_Z_FLAG == 1] # need to relax this 

    print len(df)

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

    axs[1].text(0.9, 0.9, '(b)',
                horizontalalignment='center',
                verticalalignment='center',
                transform = axs[1].transAxes)
    #-------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FIT_HB_Z_FLAG == 1] # need to relax this 
    df = df[df.OIII_FIT_HA_Z_FLAG == 1] # need to relax this 

    print len(df)

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

    axs[2].text(0.9, 0.9, '(c)',
                horizontalalignment='center',
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

def civ_blueshift_oiii_eqw():

    set_plot_properties() # change style 

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.7))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
        
    df.dropna(subset=['Median_CIV_BEST'], inplace=True)
    
    w0 = np.mean([1548.202,1550.774])*u.AA  
    median_wav = doppler2wave(df.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df.z_ICA_FIT)) / w0

    
    from LiamUtils import colormaps as cmaps
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)


    im = ax.scatter(blueshift_civ,
                    df.OIII_5007_EQW_3,
                    c=df.LogL5100,
                    edgecolor='None',
                    zorder=2,
                    s=30)    

    
    cb = fig.colorbar(im)
    cb.set_label(r'log L$_{5100{\rm \AA}}$')


    ax.set_xlim(-1000, 5000)
    ax.set_ylim(-10, 100)

    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r"[O\,{\sc iii}] EQW [\AA]")


    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/civ_blueshift_oiii_eqw.pdf')

    plt.show() 

    return None  

def civ_blueshift_oiii_blueshift():

    set_plot_properties() # change style 

    fig, ax = plt.subplots(figsize=figsize(0.9, vscale=0.8))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_SNR_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df1 = df[df.OIII_EXTREM_FLAG == 0]
    df2 = df[df.OIII_EXTREM_FLAG == 1]

  
    w0 = np.mean([1548.202,1550.774])*u.AA  

    median_wav = doppler2wave(df1.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df1.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df1.OIII_FIT_Z_FULL_OIII_PEAK)) / w0


    ax.plot(blueshift_civ, 
            df1.OIII_FIT_VEL_FULL_OIII_PEAK - df1.OIII_5007_V10,
            linestyle='',
            marker='o', 
            markerfacecolor=cs[1],
            markeredgecolor='None')

    median_wav = doppler2wave(df2.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df2.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df2.OIII_FIT_Z_FULL_OIII_PEAK)) / w0


    ax.plot(blueshift_civ, 
            df2.OIII_FIT_VEL_FULL_OIII_PEAK - df2.OIII_5007_V10,
            linestyle='',
            marker='o', 
            markerfacecolor=cs[0],
            markeredgecolor='None')    



    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'[O\,{\sc iii}] Blueshift [km~$\rm{s}^{-1}$]')
    
    
    fig.tight_layout()
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/civ_blueshift_oiii_blueshift.pdf')

    plt.show() 

    return None 

def ev1():

    """
    Try change CIV blueshift to relative to ICA redshift
    """ 

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    # df = df[df.OIII_EQW_FLAG == 0]
    # df = df[df.OIII_SNR_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_FIT_HB_Z_FLAG > 0] # not really what this flag was for, so be careful 
    df = df[(df.WARN_CIV_BEST == 0) | (df.WARN_CIV_BEST == 1)]
    df = df[df.BAL_FLAG != 1]
    df = df[np.log10(df.EQW_CIV_BEST) > 1.2]

    fig, axs = plt.subplots(2, 1, figsize=figsize(1, vscale=1.6), sharex=True)

    from LiamUtils import colormaps as cmaps
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)

    # ---------------------------------------------------------------------------------------------------

    t_ica = Table.read('/data/vault/phewett/LiamC/liam_civpar_zica_160115.dat', format='ascii') # new ICA 
        
    m1, m2 = t_ica['col2'], np.log10( t_ica['col3'])

    badinds = np.isnan(m1) | np.isnan(m2) | np.isinf(m1) | np.isinf(m2)

    m1 = m1[~badinds]
    m2 = m2[~badinds]

    xmin = -1000.0
    xmax = 3500.0
    ymin = 1.0
    ymax = 2.5

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    CS = axs[0].contour(X, Y, Z, colors=[cs[-1]])
    CS = axs[1].contour(X, Y, Z, colors=[cs[-1]])

    #----------------------------------------------------

    im = axs[0].scatter(df.Blueshift_CIV_Balmer_Best,
                        np.log10(df.EQW_CIV_BEST),
                        c = np.log10(df.OIII_5007_EQW_3), 
                        edgecolor='None',
                        s=25,
                        vmin=-0.4, vmax=1.4)

    cb = fig.colorbar(im, ax=axs[0])
    cb.set_label(r'log [O\,{\sc iii}] EW [\AA]')

    axs[0].set_xlim(-1000, 5000)
    axs[0].set_ylim(1,2.2)

    axs[0].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')

    # -----------------------------------------

    im = axs[1].scatter(df.Blueshift_CIV_Balmer_Best,
                        np.log10(df.EQW_CIV_BEST),
                        c = df.FWHM_Broad_Hb, 
                        edgecolor='None',
                        s=25,
                        vmin=1500, vmax=10000)

    cb = fig.colorbar(im, ax=axs[1])
    cb.set_label(r'H$\beta$ FHWM [km~$\rm{s}^{-1}$]')

    axs[1].set_xlim(-1000, 5000)
    axs[1].set_ylim(1,2.2)

    axs[1].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')
    axs[1].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')

    # -----------------------------------------

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/ev1.pdf')
    
    plt.show()

def test():

    """
    Try change CIV blueshift to relative to ICA redshift
    """ 

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.mfica_flag > 0]
    df = df[(df.WARN_CIV_BEST == 0)]
    df = df[df.BAL_FLAG != 1]
    df = df[np.log10(df.EQW_CIV_BEST) > 1.2]

    fig, ax = plt.subplots(1, 1, figsize=figsize(1, vscale=0.8))

    from LiamUtils import colormaps as cmaps
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)

    # ---------------------------------------------------------------------------------------------------

    # t_ica = Table.read('/data/vault/phewett/LiamC/liam_civpar_zica_160115.dat', format='ascii') # new ICA 
        
    # m1, m2 = t_ica['col2'], np.log10( t_ica['col3'])

    # badinds = np.isnan(m1) | np.isnan(m2) | np.isinf(m1) | np.isinf(m2)

    # m1 = m1[~badinds]
    # m2 = m2[~badinds]

    # xmin = -1000.0
    # xmax = 3500.0
    # ymin = 1.0
    # ymax = 2.5

    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([m1, m2])

    # kernel = stats.gaussian_kde(values)
      
    # Z = np.reshape(kernel(positions).T, X.shape)

    # CS = ax.contour(X, Y, Z, colors=[cs[-1]])
  
    #----------------------------------------------------

    im = ax.scatter(df.Blueshift_CIV_Balmer_Best,
                    np.log10(df.EQW_CIV_BEST),
                    c = np.log10(df.OIII_5007_EQW_3), 
                    edgecolor='None',
                    s=25)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label('')

    ax.set_xlim(-1000, 5000)
    ax.set_ylim(1,2.2)

    ax.set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')
    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')

  
    fig.tight_layout()

    
    plt.show()

    return None 


def eqw_lum():

    fig, ax = plt.subplots(figsize=figsize(0.8, vscale=0.9))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df1 = df[df.OIII_EQW_FLAG == 0]
    df2 = df[df.OIII_EQW_FLAG == 1]

    
    s = ax.scatter(np.log10(9.26) + df1.LogL5100,
                   np.log10(df1.OIII_5007_EQW_3), 
                   facecolor=cs[1], 
                   edgecolor='None',
                   s=10,
                   zorder=10)



    df2.loc[df2.OIII_5007_EQW_MEAN < 0.0, 'OIII_5007_EQW_MEAN'] = 0.0 
    ul = df2.OIII_5007_EQW_MEAN + df2.OIII_5007_EQW_STD

    s = ax.errorbar(np.log10(9.26) + df2.LogL5100,
                    np.log10(ul), 
                    yerr=0.2,
                    zorder=10, 
                    linestyle='None', 
                    uplims=True,
                    color=cs[1])



    t = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits')
    t = t[t['LOGLBOL'] > 0.0]
    t = t[t['EW_OIII_5007'] > 0.0]
    
    m1, m2 = t['LOGLBOL'], np.log10(t['EW_OIII_5007'])
    
    xmin = 44.0
    xmax = 48.0
    ymin = -1.0
    ymax = 3.0
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    
    kernel = stats.gaussian_kde(values)
    
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = ax.contour(X, Y, Z, colors=[cs[-1]])
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    ax.plot(x, 
            y, 
            markerfacecolor=cs[-1], 
            markeredgecolor='None', 
            linestyle='', 
            marker='o', 
            markersize=2, 
            label='SDSS DR7')
    
    ax.set_ylabel(r'log EQW (\AA)')
    ax.set_xlabel(r'log $L_{\mathrm{Bol}}$ [erg/s]')
    
    ax.set_xlim(44.5, 49)
    ax.set_ylim(-1, 3)

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter04/eqw_lum.pdf')

    plt.show() 

    return None 

# eqw_lum() 


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

    ids = ['J100627+480420', 'J124948+060714']

    for i, name in enumerate(['QSO011', 'QSO028']): 

        for snr in snrs:                   
    
            # returns dictionary of dictionaries 
            new_dict.update({snr: get_errors_oiii(name + '_snr_' + snr, plot=False, snr_test=True, model='gaussians')})
    
        p50 = np.array([x['oiii_5007_w80']['p50'] for x in new_dict.values()])
        p16 = np.array([x['oiii_5007_w80']['p16'] for x in new_dict.values()])
        p84 = np.array([x['oiii_5007_w80']['p84'] for x in new_dict.values()])
    
        lower_error = p50 - p16
        upper_error = p84 - p50 
    
        ytrue = new_dict['50p0']['oiii_5007_w80']['p50'] 
    
        axs[i].errorbar([2.5, 5, 7.5, 15, 20, 50],
                        p50 / ytrue, 
                        yerr=[lower_error / ytrue, upper_error / ytrue],
                        color='black',
                        label=ids[i])
        
        axs[i].axhline(1.0, color='black', linestyle='--')
        
        axs[i].set_xlim(0, 55) 
        axs[i].set_ylim(0.6, 1.4)
    
        axs[i].axhspan(0.9, 1.1, color=cs[1])
        # axs[i].axhspan(0.8, 0.9, color=cs[0])
        # axs[i].axhspan(1.1, 1.2, color=cs[0])
    
        axs[i].axvline(df.ix[name].OIII_FIT_SNR_CONTINUUM, color=set1[0], linestyle='--')
    
        axs[i].set_ylabel(r'$\Delta w_{80}$')
        axs[i].set_yticks([0.6, 0.8, 1, 1.2, 1.4])
        
        axs[i].annotate(ids[i], xy=(50, 1.3), ha='right') 

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

def oiii_eqw_hist():

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]

    fig, ax = plt.subplots(figsize=figsize(0.8, vscale=0.8))

    ax.hist(df.loc[df.OIII_5007_EQW_3 > 8.0, 'OIII_5007_EQW_3'],
            bins=np.arange(0, 100, 8),
            facecolor=cs[1],
            edgecolor='None',
            histtype='stepfilled')

    ax.hist(df.loc[df.OIII_5007_EQW_3 <= 8.0, 'OIII_5007_EQW_3'],
            bins=np.arange(0, 100, 8),
            facecolor=cs[0],
            edgecolor='None', 
            histtype='stepfilled')


    # ax.axvline(0.1165, color=cs[0], linestyle='--')

    ax.set_xlabel(r"[O\,{\sc iii}] EQW [\AA]")
    ax.set_ylabel(r"Count")

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/oiii_eqw_hist.pdf') 

    plt.show() 


    return None 

def example_spectra(name, 
                    ax, 
                    nrebin, 
                    plot_model=True, 
                    data_color='black',
                    voffset=0.0):

    from lmfit import Model

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    cs_light = palettable.colorbrewer.qualitative.Pastel1_9.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    instr = df.ix[name, 'INSTR']

    import sys
    sys.path.insert(1, '/home/lc585/Dropbox/IoA/nirspec/python_code')
    
    if instr == 'FIRE': from fit_properties_oiii_fire import get_line_fit_props
    if instr == 'GNIRS': from fit_properties_oiii_gnirs import get_line_fit_props
    if instr == 'ISAAC': from fit_properties_oiii_isaac import get_line_fit_props
    if instr == 'LIRIS': from fit_properties_oiii_liris import get_line_fit_props
    if instr == 'NIRI': from fit_properties_oiii_niri import get_line_fit_props
    if instr == 'NIRSPEC': from fit_properties_oiii_nirspec import get_line_fit_props
    if instr == 'SOFI_JH': from fit_properties_oiii_sofi_jh import get_line_fit_props
    if instr == 'SOFI_LC': from fit_properties_oiii_sofi_lc import get_line_fit_props
    if instr == 'TRIPLE': from fit_properties_oiii_triple import get_line_fit_props
    if instr == 'TRIPLE_S15': from fit_properties_oiii_triple_shen15 import get_line_fit_props
    if instr == 'XSHOOT': from fit_properties_oiii_xshooter import get_line_fit_props
    if instr == 'SINF': from fit_properties_oiii_sinfoni import get_line_fit_props
    if instr == 'SINF_KK': from fit_properties_oiii_sinfoni_kurk import get_line_fit_props
    
    q = get_line_fit_props().all_quasars()
    p = q[df.ix[name, 'NUM']]

    w0 = 4862.721*u.AA
 
    xs, step = np.linspace(-20000,
                            20000,
                            1000,
                           retstep=True)

    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, 'OIII')

    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params = Parameters()
    params.load(parfile)
    parfile.close()

    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav = pickle.load(parfile)
    parfile.close()

    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx = pickle.load(parfile)
    parfile.close()

    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err = pickle.load(parfile)
    parfile.close()

    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()

    vdat = wave2doppler(wav, w0)


    if plot_model: 

        mod = GaussianModel(prefix='oiii_4959_n_')
    
        mod += GaussianModel(prefix='oiii_5007_n_')
    
        mod += GaussianModel(prefix='oiii_4959_b_')
    
        mod += GaussianModel(prefix='oiii_5007_b_')
    
        if p.hb_narrow is True: 
            mod += GaussianModel(prefix='hb_n_')  
    
        for i in range(p.hb_nGaussians):
    
            mod += GaussianModel(prefix='hb_b_{}_'.format(i))  
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_5007_n_center'].value
        p1['sigma'].value = params['oiii_5007_n_sigma'].value
        p1['amplitude'].value = params['oiii_5007_n_amplitude'].value
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_4959_n_center'].value
        p1['sigma'].value = params['oiii_4959_n_sigma'].value
        p1['amplitude'].value = params['oiii_4959_n_amplitude'].value
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')        
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_5007_b_center'].value
        p1['sigma'].value = params['oiii_5007_b_sigma'].value
        p1['amplitude'].value = params['oiii_5007_b_amplitude'].value
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')       
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_4959_b_center'].value
        p1['sigma'].value = params['oiii_4959_b_sigma'].value
        p1['amplitude'].value = params['oiii_4959_b_amplitude'].value   
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')             
    
        for i in range(p.hb_nGaussians):
    
            g1 = GaussianModel()
            p1 = g1.make_params()
    
            p1['center'].value = params['hb_b_{}_center'.format(i)].value
            p1['sigma'].value = params['hb_b_{}_sigma'.format(i)].value
            p1['amplitude'].value = params['hb_b_{}_amplitude'.format(i)].value  
    
            ax.plot(np.sort(vdat.value) - voffset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=cs_light[4])  
    
        if p.hb_narrow is True: 
    
            g1 = GaussianModel()
            p1 = g1.make_params()
    
            p1['center'] = params['hb_n_center']
            p1['sigma'] = params['hb_n_sigma']
            p1['amplitude'] = params['hb_n_amplitude']   
    
            ax.plot(np.sort(vdat.value) - voffset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=cs_light[4],
                    linestyle='-')                    
    
    
        # vdat, flx, err = rebin(vdat.value, flx, err, nrebin)
        vdat = vdat.value
    
        ax.plot(xs - voffset,
                mod.eval(params=params, x=xs/sd) ,
                color='black',
                lw=1,
                zorder=6)

    ax.plot(vdat - voffset,
            flx,
            linestyle='-',
            color=data_color,
            lw=1,
            alpha=1,
            zorder=0)


    ax.axhline(0.0, color='black', linestyle=':')

    return None 

def example_residual(name, ax, voffset=0.0):

    from lmfit import Model

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    cs_light = palettable.colorbrewer.qualitative.Pastel1_9.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    instr = df.ix[name, 'INSTR']

    import sys
    sys.path.insert(1, '/home/lc585/Dropbox/IoA/nirspec/python_code')
    
    if instr == 'FIRE': from fit_properties_oiii_fire import get_line_fit_props
    if instr == 'GNIRS': from fit_properties_oiii_gnirs import get_line_fit_props
    if instr == 'ISAAC': from fit_properties_oiii_isaac import get_line_fit_props
    if instr == 'LIRIS': from fit_properties_oiii_liris import get_line_fit_props
    if instr == 'NIRI': from fit_properties_oiii_niri import get_line_fit_props
    if instr == 'NIRSPEC': from fit_properties_oiii_nirspec import get_line_fit_props
    if instr == 'SOFI_JH': from fit_properties_oiii_sofi_jh import get_line_fit_props
    if instr == 'SOFI_LC': from fit_properties_oiii_sofi_lc import get_line_fit_props
    if instr == 'TRIPLE': from fit_properties_oiii_triple import get_line_fit_props
    if instr == 'TRIPLE_S15': from fit_properties_oiii_triple_shen15 import get_line_fit_props
    if instr == 'XSHOOT': from fit_properties_oiii_xshooter import get_line_fit_props
    if instr == 'SINF': from fit_properties_oiii_sinfoni import get_line_fit_props
    if instr == 'SINF_KK': from fit_properties_oiii_sinfoni_kurk import get_line_fit_props
    
    q = get_line_fit_props().all_quasars()
    p = q[df.ix[name, 'NUM']]

    w0 = 4862.721*u.AA
  
    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, 'OIII')

    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params = Parameters()
    params.load(parfile)
    parfile.close()

    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav = pickle.load(parfile)
    parfile.close()

    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx = pickle.load(parfile)
    parfile.close()

    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err = pickle.load(parfile)
    parfile.close()

    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()
  
    vdat = wave2doppler(wav, w0)
   
    mod = GaussianModel(prefix='oiii_4959_n_')

    mod += GaussianModel(prefix='oiii_5007_n_')

    mod += GaussianModel(prefix='oiii_4959_b_')

    mod += GaussianModel(prefix='oiii_5007_b_')

    if p.hb_narrow is True: 
        mod += GaussianModel(prefix='hb_n_')  

    for i in range(p.hb_nGaussians):
        mod += GaussianModel(prefix='hb_b_{}_'.format(i))  

    ax.plot(vdat - voffset,
            (flx - mod.eval(params=params, x=vdat.value/sd)) / err,
            color='lightgray',
            lw=1)

    ax.axhline(0.0, color='black', linestyle=':')


    return None 

def example_spectrum_grid_extreme_oiii():

    fig = plt.figure(figsize=figsize(1, vscale=1.8))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(6, 3, wspace=0.0, hspace=0.15)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    


    names = ['QSO007',
             'QSO053',
             'QSO055',
             'QSO058',
             'QSO107',
             'QSO152',
             'QSO335',
             'QSO354',
             'QSO360',
             'QSO361',
             'QSO368',
             'QSO375',
             'QSO423',
             'QSO424',
             'QSO522',
             'QSO602',
             'QSO620',
             'QSO615']      


    ylims = [[0.0, 0.8],
             [0.0, 0.9],
             [0.0, 0.5],
             [0.0, 0.8],
             [0.0, 0.8],
             [0.0, 0.7],
             [0.0, 1.0],
             [0.0, 0.8],
             [0.0, 0.7],
             [0.0, 0.8],
             [0.0, 2.0],
             [0.0, 0.8],
             [0.0, 1.0],
             [0.0, 0.5],
             [0.0, 0.4],
             [0.0, 0.8],
             [0.0, 0.5],
             [0.0, 1.5]]

    titles = ['J040954-041137',
              'J162549+264659',
              'J163456+301438',
              'J011150+140141',
              'J001708+813508',
              'J144516+095836',
              'J005758-264315',
              'J112443-170517',
              'J133336+164904',
              'J134427-103542',
              'J145103-232931',
              'J120148+120630',
              'J110325-264516',
              'J024008-230915',
              'J084402+050358',
              'J110916-115449',
              'J220530-254222',
              'J144424-104542'] 

    rebins = [1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1] 
             
    for i in range(len(names)):
 
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        ax = plt.Subplot(fig, inner_grid[:3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], ax, rebins[i])
        ax.set_ylim(ylims[i])
        ax.set_xlim(-5000, 15000)
        ax.set_title(titles[i], size=9, y=0.95)



        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_grid[3])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], ax)
        ax.set_xlim(-5000, 15000)
        ax.set_ylim(-8, 8)
        
        if (i % 3 == 0):
            ax.set_yticks([-5,0,5])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])

        if i < 15:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 5000, 10000])
            ax.xaxis.set_ticks_position('bottom')

        fig.add_subplot(ax)

    fig.text(0.5, 0.05, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'Relative $F_{\lambda}$', rotation=90)
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/example_spectrum_grid_extreme_oiii.pdf')

    plt.show() 


    
    return None 

def example_spectrum_grid():

    fig = plt.figure(figsize=figsize(1, vscale=1.5))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(5, 3, wspace=0.0, hspace=0.15)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    


    names = ['QSO003',
             'QSO009',
             'QSO012',
             'QSO032',
             'QSO034',
             'QSO035',
             'QSO113',
             'QSO120',
             'QSO176',
             'QSO177',
             'QSO178',
             'QSO329',
             'QSO478',
             'QSO517',
             'QSO567']      


    ylims = [[0.0, 1],
             [0.0, 2.5],
             [0.0, 3],
             [0.0, 1.8],
             [0.0, 9],
             [0.0, 13],
             [0.0, 1.2],
             [0.0, 2],
             [0.0, 0.3],
             [0.0, 1],
             [0.0, 0.8],
             [0.0, 0.6],
             [0.0, 1.2],
             [0.0, 0.9],
             [0.0, 0.8]]

    titles = ['J013930+001331',
              'J091209+005857',
              'J101900-005420',
              'J080050+354250',
              'J084158+392121',
              'J084200+392140',
              'J023146+132255',
              'J080151+113456',
              'J003136+003421',
              'J091432+010912',
              'J093226+092526',
              'J222007-280323',
              'J002948-095639',
              'J025906+001122',
              'J010737-385325'] 

    rebins = [1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1] 
             
    for i in range(len(names)):
 
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        ax = plt.Subplot(fig, inner_grid[:3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], 
                        ax, 
                        rebins[i], 
                        data_color='lightgrey', 
                        voffset=df.loc[names[i], 'OIII_FIT_VEL_FULL_OIII_PEAK'])
        ax.set_ylim(ylims[i])
        ax.set_xlim(-5000, 15000)
        ax.set_title(titles[i], size=9, y=0.95)



        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_grid[3])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], ax)
        ax.set_xlim(-5000, 15000)
        ax.set_ylim(-8, 8)
        
        if (i % 3 == 0):
            ax.set_yticks([-5,0,5])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])

        if i < 12:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 5000, 10000])
            ax.xaxis.set_ticks_position('bottom')

        fig.add_subplot(ax)

    fig.text(0.5, 0.05, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'Relative $F_{\lambda}$', rotation=90)
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/example_spectrum_grid.pdf')

    plt.show() 


    
    return None 

def example_spectrum_grid_extreme_fe():

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    fig = plt.figure(figsize=figsize(1, vscale=1.5))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(6, 4, wspace=0.0, hspace=0.17)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    
    names = ['QSO015',
             'QSO038',
             'QSO169',
             'QSO307',
             'QSO381',
             'QSO537',
             'QSO538',
             'QSO540',
             'QSO546',
             'QSO551',
             'QSO560',
             'QSO569',
             'QSO570',
             'QSO587',
             'QSO589',
             'QSO590',
             'QSO601',
             'QSO611',
             'QSO618',
             'QSO619',
             'QSO624',
             'QSO629',
             'QSO640']

    ylims = [[0.0, 0.7],
             [0.0, 1.2],
             [0.0, 0.7],
             [0.0, 0.4],
             [0.0, 1.0],
             [0.0, 0.5],
             [0.0, 1.2],
             [0.0, 0.9],
             [0.0, 1.0],
             [0.0, 0.2],
             [0.0, 0.7],
             [0.0, 0.9],
             [0.0, 0.8],
             [0.0, 0.4],
             [0.0, 0.5],
             [0.0, 1.1],
             [0.0, 0.7],
             [0.0, 1],
             [0.0, 0.5],
             [0.0, 0.5],
             [0.0, 0.7],
             [0.0, 0.3],
             [0.0, 0.8]]  


    titles = ['J104915-011038',
              'J092747+290721',
              'J212912-153841',
              'J214507-303046',
              'J102510+045247',
              'J123355+031328',
              'J125141+080718',
              'J141949+060654',
              'J204010-065403',
              'J223820-092106',
              'J005202+010129',
              'J012257-334844',
              'J012337-323828',
              'J025055-361635',
              'J025634-401300',
              'J030211-314030',
              'J105651-114122',
              'J134104-073947',
              'J214950-444405',
              'J215052-315824',
              'J223246-363203',
              'J232539-065259',
              'J115302+215118']


    rebins = [1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1,
              1]
             
    for i in range(len(names)):
 
        
        
        ax = plt.subplot(outer_grid[i])
        ax.set_xticks([])
        ax.set_yticks([])
        example_spectra(names[i], ax, rebins[i], plot_model=False, data_color=cs[-1])
        ax.set_ylim(ylims[i])
        ax.set_xlim(-5000, 15000)
        ax.set_title(titles[i], size=9, y=0.94)
        # ax.axvline(df.loc[names[i], 'OIII_FIT_VEL_HB_PEAK'], c=cs[0], linestyle=':')
        v4959 = wave2doppler(4960.295*u.AA, w0=4862.721*u.AA).value 
        v5007 = wave2doppler(5008.239*u.AA, w0=4862.721*u.AA).value 
        ax.axvline(df.loc[names[i], 'OIII_FIT_VEL_HB_PEAK'] + v4959, c=cs[0], linestyle=':')
        ax.axvline(df.loc[names[i], 'OIII_FIT_VEL_HB_PEAK'] + v5007, c=cs[0], linestyle=':')

        fig.add_subplot(ax)

        if i < 19:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 5000, 10000])
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_ticklabels(['0', '5000', '10000'], rotation=45)

        fig.add_subplot(ax)

    fig.text(0.5, 0.02, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'Relative $F_{\lambda}$', rotation=90)
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/example_spectrum_grid_extreme_fe.pdf')

    plt.show() 


    
    return None     


def mfica_components(name): 

    from SpectraTools.fit_line import make_model_mfica, mfica_model, mfica_get_comp

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.8))

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 

    comps_wav, comps, weights = make_model_mfica(mfica_n_weights=10)
    
    for i in range(10): 
        weights['w{}'.format(i+1)].value = df.loc[name, 'mfica_w{}'.format(i+1)]    

    weights['shift'].value = df.loc[name, 'mfica_shift']    

    flux = mfica_model(weights, comps, comps_wav, comps_wav)

    ax.plot(comps_wav, 
            flux, 
            color='black', 
            lw=1)

    set2 = palettable.colorbrewer.qualitative.Set2_3.mpl_colors 
    set3 = palettable.colorbrewer.qualitative.Set3_5.mpl_colors 
    colors = [set3[0], set3[2], set3[3], set3[4]]
    

    labels = ['w1', 'w2', 'w3', 'w4+w5+w6']

    for i in range(3):

        ax.plot(comps_wav, 
                mfica_get_comp(i+1, weights, comps, comps_wav, comps_wav),
                color=colors[i],
                label=labels[i])

    flx_oiii_4 = mfica_get_comp(4, weights, comps, comps_wav, comps_wav)
    flx_oiii_5 = mfica_get_comp(5, weights, comps, comps_wav, comps_wav)
    flx_oiii_6 = mfica_get_comp(6, weights, comps, comps_wav, comps_wav)

    ax.plot(comps_wav, 
            flx_oiii_4 + flx_oiii_5 + flx_oiii_6,
            color=colors[3],
            label=labels[3])    

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=4, fancybox=True, shadow=True)


    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, 'MFICA')

    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav = pickle.load(parfile)
    parfile.close()

    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx = pickle.load(parfile)
    parfile.close()

    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err = pickle.load(parfile)
    parfile.close()

    ax.plot(wav,
            flx,
            linestyle='-',
            color='lightgrey',
            lw=1,
            alpha=1,
            zorder=0)

 
    ax.set_xlim(4700, 5100)
    ax.set_ylim(0, 2.5)

    ax.set_xlabel(r'Wavelength [\AA]') 
    ax.set_ylabel(r'$F_{\lambda}$ [Arbitrary units]')

    fig.tight_layout() 


    

    fig.savefig('/home/lc585/thesis/figures/chapter04/mfica_components.pdf')

    plt.show() 


    return None 

def oiii_reconstruction(name):

    from SpectraTools.fit_line import make_model_mfica, mfica_model, mfica_get_comp

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 

    cs = palettable.colorbrewer.diverging.RdBu_7.mpl_colors 
    cs_light = palettable.colorbrewer.qualitative.Pastel1_9.mpl_colors


    comps_wav, comps, w = make_model_mfica(mfica_n_weights=10)
    
    w['w1'].value = 0.0 
    w['w2'].value = 0.0
    w['w3'].value = 0.0
    w['w4'].value = df.loc[name, 'mfica_w4']
    w['w5'].value = df.loc[name, 'mfica_w5']
    w['w6'].value = df.loc[name, 'mfica_w6']
    w['w7'].value = df.loc[name, 'mfica_w7']
    w['w8'].value = df.loc[name, 'mfica_w8']
    w['w9'].value = df.loc[name, 'mfica_w9']
    w['w10'].value = df.loc[name, 'mfica_w10']
    w['shift'].value = df.loc[name, 'mfica_shift']

    flux = mfica_model(w, comps, comps_wav, comps_wav)

    """
    We use the blue wing of the 4960 peak to reconstruct the 
    5008 peak
    """

    peak_diff = 5008.239 - 4960.295

    # just made up these boundaries
    inds1 = (comps_wav > 4900.0) & (comps_wav < 4980 - peak_diff)
    inds2 = (comps_wav > 4980.0) & (comps_wav < 5050.0)

    wav_5008 = np.concatenate((comps_wav[inds1] + peak_diff, comps_wav[inds2]))
    flux_5008 = np.concatenate((flux[inds1], flux[inds2]))

    """
    Fit linear model and subtract background
    """

    xfit = np.concatenate((wav_5008[:10], wav_5008[-10:]))
    yfit = np.concatenate((flux_5008[:10], flux_5008[-10:]))

    mod = LinearModel()
    out = mod.fit(yfit, x=xfit, slope=0.0, intercept=0.0)

    flux_5008 = flux_5008 - mod.eval(params=out.params, x=wav_5008)

    """
    If flux is negative set to zero
    """

    flux_5008[flux_5008 < 0.0] = 0.0 


    xs = np.arange(wav_5008.min(), wav_5008.max(), 0.01)
    vs = wave2doppler(xs*u.AA, 5008.239*u.AA)

    f = interp1d(wav_5008, flux_5008)

    cdf = np.cumsum(f(xs) / np.sum(f(xs))) 

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.8))
  

    ax.plot(comps_wav, flux, color='grey') 
    ax.axhline(0.0, color='black', linestyle='--')
    ax.set_xlim(4900, 5100)
    
    ax.plot(wav_5008, flux_5008, color='black', lw=2)

    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, 'MFICA')

    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav = pickle.load(parfile)
    parfile.close()

    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx = pickle.load(parfile)
    parfile.close()

    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err = pickle.load(parfile)
    parfile.close()

    comps_wav, comps, weights = make_model_mfica(mfica_n_weights=10) 
    weights['w1'].value = df.loc[name, 'mfica_w1']
    weights['w2'].value = df.loc[name, 'mfica_w2']
    weights['w3'].value = df.loc[name, 'mfica_w3']

    wav = np.array(wav[~ma.getmask(wav)])
    flx = np.array(flx[~ma.getmask(flx)])



    flx -=  mfica_get_comp(1, weights, comps, comps_wav, wav)
    flx -=  mfica_get_comp(2, weights, comps, comps_wav, wav)
    flx -=  mfica_get_comp(3, weights, comps, comps_wav, wav)
    flx -=  mod.eval(params=out.params, x=wav)

    ax.plot(wav,
            flx,
            linestyle='-',
            color=cs_light[1],
            lw=1,
            alpha=1,
            zorder=0)

    ax.set_xlim(4925, 5075)
    ax.set_ylim(-1, 6)

    ax.set_xlabel(r'Wavelength [\AA]') 
    ax.set_ylabel(r'$F_{\lambda}$ [Arbitrary units]')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/oiii_reconstruction.pdf')

    plt.show()

    return None 


def parameters_grid():

    from PlottingTools.corner_plot import corner_plot

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_SNR_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_5007_EQW_3 < 100.0] # just for the purposes of plotting 
    df = df[df.OIII_BROAD_OFF == False] # otherwise asymetry always zero

    print len(df)
    
    chain = np.array([df.OIII_5007_W80.values*1e-3, df.OIII_5007_R, df.OIII_5007_EQW_3]).T
    
    corner_plot(chain,
                nbins=20,
                axis_labels=[r'$w_{80}$ [1000~km~$\rm{s}^{-1}$]', 'Asymmetry', r'EQW [\AA]'],
                wspace=0.07,
                hspace=0.07,
                nticks=5,
                figsize=figsize(1, 1),
                fontsize=10, 
                tickfontsize=10,
                fname='/home/lc585/thesis/figures/chapter04/parameters_grid.pdf')


    plt.show() 

    return None