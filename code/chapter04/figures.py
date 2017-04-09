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
from astropy.cosmology import WMAP9 as cosmoWMAP
import math
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut
from astropy.convolution import Gaussian1DKernel, convolve

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

    from sklearn.neighbors import KernelDensity
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import LeaveOneOut

    fig, axs = plt.subplots(3, 1, figsize=figsize(1, 2), sharex=True, sharey=True)

    mean, median, sigma = np.zeros(3), np.zeros(3), np.zeros(3)

    # OIII vs Hb -----------------------------------------------------------
 
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EXTREM_FLAG == 0]

    df = df[df.OIII_FIT_HB_Z_FLAG > 0 ] 
    df = df[df.OIII_FIT_VEL_HB_PEAK_ERR < 600.0] # Really bad 
    df = df[df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0] # Really bad 

    x = df.OIII_FIT_VEL_FULL_OIII_PEAK - df.OIII_FIT_VEL_HB_PEAK 
    mean[0], median[0], sigma[0] = np.mean(x), np.median(x), np.std(x)
 
    norm = np.std(x)
    x = x / norm

    x_d = np.linspace(-4, 4, 1000)
    
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut(len(x)))
    grid.fit(x[:, None]);

    print grid.best_params_['bandwidth'] * norm 
        
    kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
    kde.fit(x[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    axs[0].fill_between(x_d*norm, np.exp(logprob), color=cs[1])
    axs[0].plot(x*norm, np.full_like(x, -0.01), '|k', markeredgewidth=1)

    axs[0].grid() 

    

    # OIII vs Ha-----------------------------------------------------------------------------


    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EXTREM_FLAG == 0]

    df = df[df.OIII_FIT_HA_Z_FLAG > 0] 
    df = df[df.OIII_FIT_VEL_HA_PEAK_ERR < 400.0] # Really bad 
    df = df[df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0] # Really bad 

    print len(df)

    x = const.c.to(u.km/u.s)*(df.OIII_FIT_Z_FULL_OIII_PEAK - df.OIII_FIT_HA_Z)/(1.0 + df.OIII_FIT_Z_FULL_OIII_PEAK)
    x = x.value 
    mean[1], median[1], sigma[1] = np.mean(x), np.median(x), np.std(x)
 
    norm = np.std(x)
    x = x / norm

    x_d = np.linspace(-4, 4, 1000)
    
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut(len(x)))
    grid.fit(x[:, None]);

    print grid.best_params_['bandwidth'] * norm 
        
    kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
    kde.fit(x[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    axs[1].fill_between(x_d*norm, np.exp(logprob), color=cs[1])
    axs[1].plot(x*norm, np.full_like(x, -0.01), '|k', markeredgewidth=1)

    axs[1].grid() 

    # Hb vs Ha-----------------------------------------------------------------------------


    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    
    df.drop(['QSO394', 'QSO391'], inplace=True) # Duplicates 
    
    df = df[df.OIII_FIT_HA_Z_FLAG > 0 ]
    df = df[df.OIII_FIT_HB_Z_FLAG > 0 ]
    
    df = df[df.OIII_FIT_VEL_HA_PEAK_ERR < 400.0]
    df = df[df.OIII_FIT_VEL_HB_PEAK_ERR < 600.0]

    df.drop(['QSO546'], inplace=True) # bad fit

    print len(df)

    x = const.c.to(u.km/u.s)*(df.OIII_FIT_HB_Z - df.OIII_FIT_HA_Z)/(1.0 + df.OIII_FIT_HA_Z)
    x = x.value 
    mean[2], median[2], sigma[2] =  np.mean(x), np.median(x), np.std(x)
 
    norm = np.std(x)
    x = x / norm

    x_d = np.linspace(-4, 4, 1000)
    
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut(len(x)))
    grid.fit(x[:, None]);
        
    print grid.best_params_['bandwidth'] * norm 

    kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
    kde.fit(x[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    axs[2].fill_between(x_d*norm, np.exp(logprob), color=cs[1])
    axs[2].plot(x*norm, np.full_like(x, -0.01), '|k', markeredgewidth=1)

    axs[2].grid() 

    # ----------------------------------------------------------------------------------

    axs[0].set_ylim(-0.05, 0.45)

    labels = ['(a)', '(b)', '(c)']


    for i, label in enumerate(labels):

        axs[i].text(0.70, 
                    0.93, 
                    labels[i] + ' \n'\
                    + r'$\mu = {0:.0f}$'.format(mean[i]) + '\n'\
                    + r'median$ = {0:.0f}$'.format(median[i]) + '\n'\
                    + r'$\sigma = {0:.0f}$'.format(sigma[i]),
                    horizontalalignment='left',
                    verticalalignment='top',
                    multialignment='left',
                    transform = axs[i].transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # #----------------------------------------------------------


    axs[0].set_xlabel(r'$c(z{[{\rm OIII}]} - z{{\rm H}\beta}) / (1 + z{[{\rm OIII}]})$ [km~$\rm{s}^{-1}$]')
    axs[1].set_xlabel(r'$c(z{[{\rm OIII}]} - z{{\rm H}\alpha}) / (1 + z{[{\rm OIII}]})$ [km~$\rm{s}^{-1}$]')
    axs[2].set_xlabel(r'$c(z{{\rm H}\beta} - z{{\rm H}\alpha}) / (1 + z{{\rm H}\beta})$ [km~$\rm{s}^{-1}$]')

    axs[0].set_ylabel('PDF')
    axs[1].set_ylabel('PDF')
    axs[2].set_ylabel('PDF')


    
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
    df = df[(df.WARN_CIV_BEST == 0) | (df.WARN_CIV_BEST == 1)]
    df = df[df.BAL_FLAG != 1]
    df = df[np.log10(df.EQW_CIV_BEST) > 1.2] # need to say this 

    df['z'] = np.nan
    
    useoiii = (df.OIII_EQW_FLAG == 0) & (df.OIII_EXTREM_FLAG == 0) & (df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 200.0)
    df.loc[useoiii, 'z'] = df.loc[useoiii, 'OIII_FIT_Z_FULL_OIII_PEAK'] 
    
    useha = df.z.isnull() & (df.OIII_FIT_HA_Z_FLAG > 0) & (df.OIII_FIT_VEL_HA_PEAK_ERR < 200.0)
    df.loc[useha, 'z'] = df.loc[useha, 'OIII_FIT_HA_Z'] 
        
    usehb = df.z.isnull() & (df.OIII_FIT_HB_Z_FLAG >= 0) & (df.OIII_FIT_VEL_HB_PEAK_ERR < 375.0)
    df.loc[usehb, 'z'] = df.loc[usehb, 'OIII_FIT_HB_Z']

    df.dropna(subset=['Median_CIV_BEST'], inplace=True)
    
    w0 = np.mean([1548.202,1550.774])*u.AA  
    median_wav = doppler2wave(df.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df.z)) / w0

    im = ax.scatter(blueshift_civ,
                    df.OIII_5007_EQW_3,
                    c=df.LogL5100,
                    edgecolor='None',
                    zorder=2,
                    cmap=palettable.matplotlib.Viridis_10.mpl_colormap,
                    s=30)   

    print df.loc[blueshift_civ.value < 500.0, 'OIII_5007_EQW_3'].mean(), df.loc[blueshift_civ.value > 2000.0, 'OIII_5007_EQW_3'].mean() 

    
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

def lum_w80():

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    
    fig, ax = plt.subplots(figsize=figsize(1, 0.8))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_EXTREM_FLAG == 1]
    print 'Number of extreme: {}'.format(len(df))
    
    s = ax.scatter(np.log10(9.26) + df.LogL5100,
                   df.OIII_5007_W80, 
                   facecolor=cs[0], 
                   edgecolor='None',
                   s=25,
                   zorder=1)

    print df.LogL5100.median() + np.log10(9.26)
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_EXTREM_FLAG == 0]

    print df.LogL5100.median() + np.log10(9.26)
    
    s = ax.scatter(np.log10(9.26) + df.LogL5100,
                   df.OIII_5007_W80, 
                   facecolor=cs[1], 
                   edgecolor='None',
                   s=25,
                   zorder=0)

    ax.set_xlabel('log L$_{\mathrm{Bol}}$ [erg~s$^{-1}$]')
    ax.set_ylabel(r'$w_{80}$ [km~$\rm{s}^{-1}$]')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/lum_w80.pdf')

    plt.show() 



    return None 

def civ_blueshift_oiii_blueshift(check_lum=False):

    set_plot_properties() # change style 

    cs = palettable.matplotlib.Viridis_3.mpl_colors

   
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0] # more or less greater than 8 A 
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[(df.WARN_CIV_BEST == 0) | (df.WARN_CIV_BEST == 1)] 
    df = df[df.BAL_FLAG != 1]
    
    df['yval'] = df['OIII_5007_V10_Blueshift']
    df['yerr'] = df['OIII_5007_V10_Blueshift_ERR']

    ycut = 250.0 
    
    # two XSHOOTER errors are missing
    # the fits both look good, so just assign typical error 
    print np.sum(df.Median_CIV_BEST_Err.isnull()), np.sum(df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR.isnull())
    df.Median_CIV_BEST_Err.fillna(df.Median_CIV_BEST_Err.median(), inplace=True)
    
    w0 = np.mean([1548.202,1550.774])*u.AA  
    median_wav = doppler2wave(df.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df.OIII_FIT_Z_FULL_OIII_PEAK)) / w0
    
    df['xval'] = blueshift_civ.value 
    df['xerr'] = np.sqrt(df['OIII_FIT_VEL_FULL_OIII_PEAK_ERR']**2 + df['Median_CIV_BEST_Err']**2).values
    
    # check for missing data 
    print df.yerr.isnull().any()
    print df.xerr.isnull().any()
    print df.xval.isnull().any()
    print df.yval.isnull().any()
    
    xcut = 125.0 

    # cheeky, but this is low S/N
    df.ix['QSO642', 'yval'] = np.nan


    if not check_lum: 

        fig = plt.figure(figsize=figsize(1.0, vscale=1.5))
        gs = gridspec.GridSpec(3, 2) 
    
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1:, :])
    
        ax1.plot(df.loc[(df.yerr > ycut) & (df.OIII_EXTREM_FLAG == 0), 'yval'], 
                 df.loc[(df.yerr > ycut) & (df.OIII_EXTREM_FLAG == 0), 'yerr'], 
                 marker='o', 
                 markeredgecolor='None',
                 markerfacecolor=cs[1],
                 markersize=3,
                 linestyle='')
        
        ax1.plot(df.loc[(df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'yval'], 
                 df.loc[(df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'yerr'], 
                 marker='o', 
                 markeredgecolor='None',
                 markerfacecolor=cs[1],
                 markersize=3,
                 linestyle='')
        
        
        ax1.axhline(ycut, color='black', linestyle='--')
        
        
    
        ax2.plot(df.loc[(df.xerr > xcut) & (df.OIII_EXTREM_FLAG == 0), 'xval'], 
                 df.loc[(df.xerr > xcut) & (df.OIII_EXTREM_FLAG == 0), 'xerr'], 
                 marker='o', 
                 markeredgecolor='None',
                 markerfacecolor=cs[1],
                 markersize=3,
                 linestyle='')
        
        
        ax2.plot(df.loc[(df.xerr < xcut) & (df.OIII_EXTREM_FLAG == 0), 'xval'], 
                 df.loc[(df.xerr < xcut) & (df.OIII_EXTREM_FLAG == 0), 'xerr'], 
                 marker='o', 
                 markeredgecolor='None',
                 markerfacecolor=cs[1],
                 markersize=3,
                 linestyle='')
        
        
        ax2.axhline(xcut, color='black', linestyle='--')
    
        
        im = ax3.scatter(df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'xval'], 
                         df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'yval'],
                         c = np.log10(9.26) + df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'LogL5100'],
                         marker='o', 
                         edgecolor='None',
                         cmap=palettable.matplotlib.Viridis_10.mpl_colormap)       

        x = df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'xval']
        y = df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'yval']

        from scipy.stats import spearmanr 
        print spearmanr(x, y)
       
        # cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
        # cb = plt.colorbar(im, cax = cbaxes, ticks=[46, 46.5, 47, 47.5, 48])  
        cb = fig.colorbar(im, ticks=[46, 46.5, 47, 47.5, 48], orientation='horizontal')

        cb.set_label('log L$_{\mathrm{Bol}}$ [erg~s$^{-1}$]')

        
        # ax3.plot(df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'xval'], 
        #          df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'yval'],
        #          linestyle='',
        #          marker='o', 
        #          markerfacecolor=cs[1],
        #          markersize = 4,
        #          markeredgecolor='None')
    
        ax3.set_ylim(0, 2000)
        
        ax1.xaxis.set_major_locator(MaxNLocator(4)) 
        ax2.xaxis.set_major_locator(MaxNLocator(4)) 
    
        ax1.yaxis.set_major_locator(MaxNLocator(4)) 
        ax2.yaxis.set_major_locator(MaxNLocator(4)) 
        
        ax3.set_xlabel(r'$\Delta v$(C\,{\sc iv}) [km~$\rm{s}^{-1}$]')
        ax3.set_ylabel(r'$\Delta v$([O\,{\sc iii}]) [km~$\rm{s}^{-1}$]')
        
        ax1.set_xlabel(r'$\Delta v$([O\,{\sc iii}])', fontsize=9)
        ax1.set_ylabel(r'$\sigma \Delta v$([O\,{\sc iii}])', fontsize=9)
    
        ax2.set_xlabel(r'$\Delta v$(C\,{\sc iv})', fontsize=9)
        ax2.set_ylabel(r'$\sigma \Delta v$(C\,{\sc iv})', fontsize=9)
    
        fig.tight_layout()


    
        fig.savefig('/home/lc585/thesis/figures/chapter04/civ_blueshift_oiii_blueshift.pdf')
    
    else:

        fig, ax = plt.subplots(figsize=figsize(1, 0.7))
        
        im = ax.scatter(df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'xval'], 
                        df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'yval'],
                        c = np.log10(9.26) + df.loc[(df.xerr < xcut) & (df.yerr < ycut) & (df.OIII_EXTREM_FLAG == 0), 'LogL5100'],
                        marker='o', 
                        edgecolor='None',
                        cmap=palettable.matplotlib.Viridis_10.mpl_colormap)       

        cb = fig.colorbar(im, ticks=[46, 46.5, 47, 47.5, 48])

        cb.set_label('log L$_{\mathrm{Bol}}$ [erg~s$^{-1}$]')


        ax.set_xlabel(r'$\Delta v$(C\,{\sc iv}) [km~$\rm{s}^{-1}$]')
        ax.set_ylabel(r'$\Delta v$([O\,{\sc iii}]) [km~$\rm{s}^{-1}$]')

        fig.tight_layout()

        fig.savefig('/home/lc585/thesis/figures/chapter04/civ_blueshift_oiii_blueshift_luminosity.pdf')


    plt.show() 
    




    return None 

def ev1():


    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[(df.WARN_CIV_BEST == 0) | (df.WARN_CIV_BEST == 1)]
    df = df[df.BAL_FLAG != 1]
    df = df[np.log10(df.EQW_CIV_BEST) > 1.2] # need to say this 
    df.drop('QSO615', inplace=True) # no redshift and civ low S/N anyway 

    # So I guess since I have shown the redshifts are (relatively) unbiased 
    # we can use whatever. 
    
    print len(df)

    df['z'] = np.nan
    
    useoiii = (df.OIII_EQW_FLAG == 0) & (df.OIII_EXTREM_FLAG == 0) & (df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0)
    df.loc[useoiii, 'z'] = df.loc[useoiii, 'OIII_FIT_Z_FULL_OIII_PEAK'] 
    
    useha = df.z.isnull() & (df.OIII_FIT_HA_Z_FLAG > 0) & (df.OIII_FIT_VEL_HA_PEAK_ERR < 400.0)
    df.loc[useha, 'z'] = df.loc[useha, 'OIII_FIT_HA_Z'] 
        
    usehb = df.z.isnull() & (df.OIII_FIT_HB_Z_FLAG >= 0) & (df.OIII_FIT_VEL_HB_PEAK_ERR < 750.0)
    df.loc[usehb, 'z'] = df.loc[usehb, 'OIII_FIT_HB_Z'] 

    df.ix['QSO055', 'z'] = df.ix['QSO055', 'z_ICA']


    fig, axs = plt.subplots(2, 1, figsize=figsize(1, vscale=1.6), sharex=True)
 
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

    w0 = np.mean([1548.202,1550.774])*u.AA  
    median_wav = doppler2wave(df.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df.z)) / w0

    im = axs[0].scatter(blueshift_civ,
                        np.log10(df.EQW_CIV_BEST),
                        c = np.log10(df.OIII_5007_EQW_3), 
                        edgecolor='None',
                        s=25,
                        cmap=palettable.matplotlib.Viridis_10.mpl_colormap,
                        vmin=-0.4, vmax=1.4)

    cb = fig.colorbar(im, ax=axs[0])
    cb.set_label(r'log [O\,{\sc iii}] EW [\AA]')

    axs[0].set_xlim(-1000, 5000)
    axs[0].set_ylim(1,2.2)

    axs[0].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')

    # -----------------------------------------

    im = axs[1].scatter(blueshift_civ,
                        np.log10(df.EQW_CIV_BEST),
                        c = df.OIII_FIT_HB_BROAD_FWHM, 
                        edgecolor='None',
                        s=25,
                        cmap=palettable.matplotlib.Viridis_10.mpl_colormap,
                        vmin=1500, vmax=7000)

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


    return None 

def ev1_lowz():

    from PlottingTools.kde_contours import kde_contours

    from fit_properties_oiii_fire import get_line_fit_props as get_line_fit_props_fire
    from fit_properties_oiii_gnirs import get_line_fit_props as get_line_fit_props_gnirs
    from fit_properties_oiii_isaac import get_line_fit_props as get_line_fit_props_isaac
    from fit_properties_oiii_liris import get_line_fit_props as get_line_fit_props_liris
    from fit_properties_oiii_niri import get_line_fit_props as get_line_fit_props_niri
    from fit_properties_oiii_nirspec import get_line_fit_props as get_line_fit_props_nirspec
    from fit_properties_oiii_sofi_jh import get_line_fit_props as get_line_fit_props_sofi_jh
    from fit_properties_oiii_sofi_lc import get_line_fit_props as get_line_fit_props_sofi_lc
    from fit_properties_oiii_triple import get_line_fit_props as get_line_fit_props_triple
    from fit_properties_oiii_triple_shen15 import get_line_fit_props as get_line_fit_props_triple_shen15
    from fit_properties_oiii_xshooter import get_line_fit_props as get_line_fit_props_xshooter
    from fit_properties_oiii_sinfoni import get_line_fit_props as get_line_fit_props_sinfoni
    from fit_properties_oiii_sinfoni_kurk import get_line_fit_props as get_line_fit_props_sinfoni_kurk
    
    from SpectraTools.fit_line import wave2doppler, doppler2wave

    

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 

    # removes missing hb 
    df = df[df.OIII_FIT_HB_Z_FLAG > 0]   
    print len(df)

    # remove duplicates 
    df.drop(['QSO330',
             'QSO461',
             'QSO424',
             'QSO409',
             'QSO115',
             'QSO348',
             'QSO349',
             'QSO372',
             'QSO379',
             'QSO040',
             'QSO015',
             'QSO384',
             'QSO137',
             'QSO635',
             'QSO146',
             'QSO373',
             'QSO610',
             'QSO048',
             'QSO236',
             'QSO162',
             'QSO412',
             'QSO343',
             'QSO333'],
             inplace=True)
    
    print len(df)



    df['OIII_FIT_FE_STRENGTH'] = df.OIII_FIT_EQW_FE_4434_4684 / df.OIII_FIT_HB_BROAD_EQW


    # Missing Fe --------------------
    
    q_fire = get_line_fit_props_fire().all_quasars() 
    q_gnirs = get_line_fit_props_gnirs().all_quasars() 
    q_isaac = get_line_fit_props_isaac().all_quasars() 
    q_liris = get_line_fit_props_liris().all_quasars() 
    q_niri = get_line_fit_props_niri().all_quasars() 
    q_nirspec = get_line_fit_props_nirspec().all_quasars() 
    q_sofi_jh = get_line_fit_props_sofi_jh().all_quasars() 
    q_sofi_lc = get_line_fit_props_sofi_lc().all_quasars() 
    q_triple = get_line_fit_props_triple().all_quasars() 
    q_triple_shen15 = get_line_fit_props_triple_shen15().all_quasars() 
    q_xshooter = get_line_fit_props_xshooter().all_quasars() 
    q_sinfoni = get_line_fit_props_sinfoni().all_quasars()
    q_sinfoni_kurk = get_line_fit_props_sinfoni_kurk().all_quasars()
    
    q = q_fire + q_gnirs + q_isaac + q_liris + q_niri + q_nirspec + q_sofi_jh + q_sofi_lc + q_triple + q_triple_shen15 + q_xshooter + q_sinfoni + q_sinfoni_kurk
    
    names = np.array([qi.name for qi in q])
    
    w0=4862.721*u.AA
    
    for idx, row in df.iterrows():
         
        qi = q[np.where(names == idx)[0][0]]
        
        continuum_region = qi.continuum_region[0] 
        
        if continuum_region.unit == (u.km/u.s):
            continuum_region = doppler2wave(continuum_region, w0)
    
        df.ix[idx, 'intersection'] = len(set(range(int(continuum_region.value[0]), int(continuum_region.value[1]))).intersection(range(4434, 4684)))
        
            
    df = df[df.intersection > 150.0] # about half the full region (250 pixels)
    
    df.drop('QSO509', inplace=True) # no fe region (might be others, I haven't been very careful)

    print len(df)

    df = df[df.FE_FLAG == 0] # not bad Fe fit 

    print len(df)

    df.dropna(subset=['OIII_FIT_FE_STRENGTH_ERR'], inplace=True) 
    # removes two with nan's, but fe clearly poorly 
    # constrained so just drop

    print len(df)



    # ---------------------------------------------------------------
    

    yerr = df.OIII_FIT_HB_BROAD_FWHM_ERR / df.OIII_FIT_HB_BROAD_FWHM 
    xerr = df.OIII_FIT_FE_STRENGTH_ERR

    good = (yerr < 0.5) & (xerr < 0.5)
    df = df[good]

    print len(df)

    
    # fig, ax = plt.subplots(1, 1, figsize=figsize(1, vscale=0.9))
    
    # ax.plot(df.OIII_FIT_FE_STRENGTH,
    #         df.OIII_FIT_HB_BROAD_FWHM,
    #         linestyle='',
    #         marker='o',
    #         markerfacecolor=cs[0])
    
    
    # SDSS ---------------------------------------------------------------
    
    t = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits') 
    
    xi = t['EW_FE_HB_4434_4684'] / t['EW_BROAD_HB']
    yi = t['FWHM_BROAD_HB'] 
    
    drop = np.isnan(xi) | np.isnan(yi) | np.isinf(xi) | np.isinf(yi)
    
    xi = xi[~drop]
    yi = yi[~drop]
    
    # xmin = 0.0
    # xmax = 3.0
    # ymin = 500.0
    # ymax = 14000.0 
    
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([xi, yi])
    
    # kernel = stats.gaussian_kde(values)
    
    # Z = np.reshape(kernel(positions).T, X.shape)
    
    
    # kde_contours(xi, 
    #              yi, 
    #              ax, 
    #              color='black',
    #              lims=[xmin, xmax, ymin, ymax],
    #              plotpoints=False)
    
    # ax.imshow(np.flipud(Z.T), 
    #       extent=(xmin, xmax, ymin, ymax), 
    #       aspect='auto', 
    #       zorder=0, 
    #       cmap='Blues',
    #       vmax=0.00015)

    # # -----------------------------------------
   
    # ax.set_xlim(None, None)
    # ax.set_ylim(None, 14000) 

    # ax.set_xlabel(r'R$_{\rm FeII}$')
    # ax.set_ylabel(r'FWHM H$\beta$ [km~$\rm{s}^{-1}$]')

    # fig.tight_layout()

    # fig.savefig('/home/lc585/thesis/figures/chapter04/ev1_lowz.pdf')
    
    fig, axs = plt.subplots(2, 1, figsize=figsize(1, vscale=1.3))

    axs[0].hist(xi, 
                normed=True, 
                bins=np.linspace(0, 3, 14), 
                histtype='stepfilled', 
                color=cs[1],
                label='SDSS')

    axs[0].hist(df.OIII_FIT_FE_STRENGTH, 
                normed=True, 
                bins=np.linspace(0, 3, 14), 
                histtype='step',
                color=cs[0],
                lw=2,
                label='This work')

    axs[0].legend(frameon=False) 

    axs[0].set_xlabel(r'R$_{\rm FeII}$')

    axs[1].hist(yi, 
                normed=True, 
                bins=np.arange(1000, 10000, 1000), 
                histtype='stepfilled', 
                color=cs[1])

    axs[1].hist(df.OIII_FIT_HB_BROAD_FWHM, 
                normed=True, 
                bins=np.arange(1000, 10000, 1000), 
                histtype='step',
                color=cs[0],
                lw=2)
    
    axs[1].set_xlabel(r'FWHM H$\beta$ [km~$\rm{s}^{-1}$]')

    axs[0].set_yticks([])
    axs[1].set_yticks([])

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/ev1_hists.pdf')



    plt.show()

    return None 



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

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.9))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 

    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    

    df1 = df[df.OIII_5007_EQW_3 > 1] 
    df2 = df[df.OIII_5007_EQW_3 <= 1] 

    
    ax.plot(np.log10(9.26) + df1.LogL5100,
            df1.OIII_5007_EQW_3,
            marker='o',
            linestyle='',
            markerfacecolor=cs[1],
            markeredgecolor='None',
            markersize=4)



    ax.errorbar(np.log10(9.26) + df2.LogL5100,
                np.ones_like(df2.LogL5100),
                yerr=0.3, 
                uplims=True,
                marker='o',
                linestyle='',
                capsize=2,
                color=cs[1],
                markeredgecolor='None',
                markersize=4)




    t = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits')
    t = t[t['LOGLBOL'] > 0.0]
    t = t[t['EW_OIII_5007'] > 0.0]
    t = t[['LOGLBOL', 'EW_OIII_5007']]
    
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
    
    CS = ax.contour(X, 10**Y, Z, colors=[cs[-1]])
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    ax.plot(x[10**y > 1], 
            10**y[10**y > 1], 
            markerfacecolor=cs[-1], 
            markeredgecolor='None', 
            linestyle='', 
            marker='o', 
            markersize=2, 
            label='SDSS DR7')


    # ax.plot(x[10**y <= 1], 
    #         np.full_like(x[10**y <= 1], 0.9), 
    #         markeredgecolor=cs[-1],
    #         marker='|',
    #         linestyle='', 
    #         markeredgewidth=2)

    ax.errorbar(x[10**y <= 1],
                np.ones_like(x[10**y <= 1]),
                yerr=0.3, 
                uplims=True,
                marker='o',
                linestyle='',
                capsize=1,
                color=cs[-1],
                markeredgecolor='None',
                markersize=0,
                zorder=0)

    #-------------------------------------

    df_sdss = t.to_pandas()
    df.rename(columns={'OIII_5007_EQW_3': 'EW_OIII_5007'}, inplace=True)
    df['LOGLBOL'] = np.log10(9.26) + df.LogL5100
    df = df[['LOGLBOL', 'EW_OIII_5007']]

    df_total = pd.concat([df[df.EW_OIII_5007 > 1.0], df_sdss[df_sdss.EW_OIII_5007 > 1.0]], ignore_index=True) 

    df_total['Binned'] = np.digitize(df_total.LOGLBOL, bins=np.linspace(45, 48, 7))
    grouped = df_total.groupby(by = 'Binned')
    ax.plot(grouped.LOGLBOL.median()[1:-1], grouped.EW_OIII_5007.median()[1:-1], color=cs[0], lw=2)

    print grouped.EW_OIII_5007.median()[1:-1]
    print grouped.LOGLBOL.median()[1:-1]


    ax.grid() 



    ax.set_ylabel(r'log EQW (\AA)')
    ax.set_xlabel(r'log $L_{\mathrm{Bol}}$ [erg/s]')
    
    ax.set_yscale('log')
    
    ax.set_xlim(44.5, 49)
    ax.set_ylim(5e-1, 5e2)

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter04/eqw_lum.pdf')

    plt.show() 

    return None 

def high_eqw_comp():

    df_nir = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 

    df_nir = df_nir[df_nir.OIII_FLAG_2 > 0]
    df_nir = df_nir[df_nir.OIII_BAD_FIT_FLAG == 0]
    df_nir = df_nir[df_nir.FE_FLAG == 0]
    df_nir = df_nir[df_nir.OIII_5007_EQW_3 > 1.0]
    
    x = df_nir.OIII_5007_EQW_3.values
    
    std = np.std(x)
    mean = np.mean(x)
    
    x = (x - mean) / std
    
    x_d = np.linspace(-3, 4, 1000)
    
    if False:
        
        bandwidths = 10 ** np.linspace(-2, 0, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut(len(x)))
        grid.fit(x[:, None]);
    
        print grid.best_params_

        
    kde = KernelDensity(bandwidth=0.25, kernel='gaussian')
    kde.fit(x[:, None])
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    fig, ax = plt.subplots(figsize=figsize(1, 0.8))

    ax.plot(x_d * std + mean, np.exp(logprob), color=cs[1], lw=2, label='This work')
    
    print 'NIR max: {}'.format(x_d[np.exp(logprob).argmax()] * std + mean)

    t = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits')
    t = t[t['LOGLBOL'] > 0.0]
    t = t[t['EW_OIII_5007'] > 0.0]
    t = t[['LOGLBOL', 'EW_OIII_5007']]
    df_sdss = t.to_pandas()
    

    x = df_sdss.EW_OIII_5007.values
    
    x = (x - mean) / std
    
    kde = KernelDensity(bandwidth=0.25, kernel='gaussian')
    kde.fit(x[:, None])
    
    print 'Bandwidth: {}'.format(0.25 * std) 
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    ax.plot(x_d * std + mean, np.exp(logprob), color=cs[0], lw=2, label='SDSS')
    
    ax.set_xlim(1, 150)

    ax.grid()
    ax.legend()

    print 'SDSS max: {}'.format(x_d[np.exp(logprob).argmax()] * std + mean)

    ax.set_xlabel(r"[O\,{\sc iii}] EQW [\AA]")
    ax.set_ylabel('PDF')
  
    fig.tight_layout()
    fig.savefig('/home/lc585/thesis/figures/chapter04/high_eqw_comp.pdf')

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
                    components_color='yellow',
                    voffset=0.0,
                    smooth=None):

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
                c=components_color,
                linestyle='-')
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_4959_n_center'].value
        p1['sigma'].value = params['oiii_4959_n_sigma'].value
        p1['amplitude'].value = params['oiii_4959_n_amplitude'].value
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=components_color,
                linestyle='-')        
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_5007_b_center'].value
        p1['sigma'].value = params['oiii_5007_b_sigma'].value
        p1['amplitude'].value = params['oiii_5007_b_amplitude'].value
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=components_color,
                linestyle='-')       
    
        g1 = GaussianModel()
        p1 = g1.make_params()
    
        p1['center'].value = params['oiii_4959_b_center'].value
        p1['sigma'].value = params['oiii_4959_b_sigma'].value
        p1['amplitude'].value = params['oiii_4959_b_amplitude'].value   
    
        ax.plot(np.sort(vdat.value) - voffset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=components_color,
                linestyle='-')             
    
        for i in range(p.hb_nGaussians):
    
            g1 = GaussianModel()
            p1 = g1.make_params()
    
            p1['center'].value = params['hb_b_{}_center'.format(i)].value
            p1['sigma'].value = params['hb_b_{}_sigma'.format(i)].value
            p1['amplitude'].value = params['hb_b_{}_amplitude'.format(i)].value  
    
            ax.plot(np.sort(vdat.value) - voffset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=components_color)  
    
        if p.hb_narrow is True: 
    
            g1 = GaussianModel()
            p1 = g1.make_params()
    
            p1['center'] = params['hb_n_center']
            p1['sigma'] = params['hb_n_sigma']
            p1['amplitude'] = params['hb_n_amplitude']   
    
            ax.plot(np.sort(vdat.value) - voffset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=components_color,
                    linestyle='-')                    
    
    
        # vdat, flx, err = rebin(vdat.value, flx, err, nrebin)
        vdat = vdat.value
    
        ax.plot(xs - voffset,
                mod.eval(params=params, x=xs/sd) ,
                color='black',
                lw=1,
                zorder=6)

  

    if smooth is not None:

        smooth_scale = np.rint(smooth / np.median(np.diff(wave2doppler(wav, w0).value)))
        gauss = Gaussian1DKernel(stddev=smooth_scale)
        flx = convolve(flx, gauss)

    ax.plot(vdat - voffset,
            flx,
            linestyle='-',
            color=data_color,
            lw=1,
            alpha=1,
            zorder=0)


    ax.axhline(0.0, color='black', linestyle=':')

    return None 

def example_residual(name, 
                     ax, 
                     voffset=0.0,
                     data_color='lightgrey'):

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
            color=data_color,
            lw=1)

    ax.axhline(0.0, color='black', linestyle=':')


    return None 

def example_spectrum_grid_extreme_oiii():

    # fig = plt.figure(figsize=figsize(1, vscale=1.7))
    fig = plt.figure(figsize=figsize(1, vscale=1.36))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 2, wspace=0.0, hspace=0.15)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    
    # names = ['QSO107',
    #          'QSO335',
    #          'QSO058',
    #          'QSO424',
    #          'QSO007',
    #          'QSO522',
    #          'QSO423',
    #          'QSO602',
    #          'QSO354',
    #          'QSO375']      


    # ylims = [[0.0, 0.9],
    #          [0.0, 1.0],
    #          [0.0, 0.8],
    #          [0.0, 0.5],
    #          [0.0, 0.7],
    #          [0.0, 0.35],
    #          [0.0, 1.0],
    #          [0.0, 0.8],
    #          [0.0, 0.8],
    #          [0.0, 0.8]]

    # titles = ['J001708+813508',
    #           'J005758-264315',
    #           'J011150+140141',
    #           'J024008-230915',
    #           'J040954-041137',
    #           'J084402+050358',
    #           'J110325-264516',
    #           'J110916-115449',
    #           'J112443-170517',
    #           'J120148+120630'] 

    names = ['QSO360',
             'QSO361',
             'QSO615',
             'QSO152',
             'QSO368',
             'QSO053',
             'QSO055',
             'QSO620']      


    ylims = [[0.0, 0.7],
             [0.0, 0.8],
             [0.0, 1.2],
             [0.0, 0.7],
             [0.0, 1.8],
             [0.0, 0.8],
             [0.0, 0.5],
             [0.0, 0.4]]

    titles = ['J133336+164904',
              'J134427-103542',
              'J144424-104542',
              'J144516+095836',
              'J145103-232931',
              'J162549+264659',
              'J163456+301438',
              'J220530-254222'] 

    rebins = np.ones(len(names))
             
    for i in range(len(names)):
 
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        ax = plt.Subplot(fig, inner_grid[:3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], 
                        ax, 
                        rebins[i], 
                        data_color=palettable.colorbrewer.sequential.Greys_9.mpl_colors[4],
                        components_color=palettable.colorbrewer.sequential.Oranges_9.mpl_colors[3])
        ax.set_ylim(ylims[i])
        ax.set_xlim(-5000, 15000)
        ax.set_title(titles[i], size=9, y=0.95)

        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_grid[3])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], ax, data_color=palettable.colorbrewer.sequential.Greys_9.mpl_colors[4])
        ax.set_xlim(-5000, 15000)
        ax.set_ylim(-6, 6)
        
        if (i % 2 == 0):
            ax.set_yticks([-3,3])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])

        if i < 6:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 5000, 10000])
            ax.xaxis.set_ticks_position('bottom')

        fig.add_subplot(ax)

    fig.text(0.5, 0.05, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'Relative $F_{\lambda}$', rotation=90)
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/example_spectrum_grid_extreme_oiii_1.pdf')

    plt.show() 


    
    return None 

def example_spectrum_grid():

    fig = plt.figure(figsize=figsize(1, vscale=1.5))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 2, wspace=0.0, hspace=0.15)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    

    names = ['QSO478',
             'QSO176',
             'QSO113',
             'QSO517',
             'QSO120',
             'QSO034',
             'QSO012',
             'QSO329']      


    ylims = [[0.0, 1.5],
             [0.0, 0.3],
             [0.0, 1.2],
             [0.0, 1],
             [0.0, 2],
             [0.0, 9],
             [0.0, 2.8],
             [0.0, 0.55]]


    titles = ['J002948-095639',
              'J003136+003421',
              'J023146+132255',
              'J025906+001122',
              'J080151+113456',
              'J084158+392121',
              'J101900-005420',
              'J222007-280323'] 

    rebins = np.zeros(len(names))
             
    for i in range(8):
 
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        ax = plt.Subplot(fig, inner_grid[:3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], 
                        ax, 
                        rebins[i], 
                        data_color=palettable.colorbrewer.sequential.Greys_9.mpl_colors[4],
                        components_color=palettable.colorbrewer.sequential.Oranges_9.mpl_colors[3],
                        voffset=df.loc[names[i], 'OIII_FIT_VEL_FULL_OIII_PEAK'])
        ax.set_ylim(ylims[i])
        ax.set_xlim(-5000, 15000)
        ax.set_title(titles[i], size=9, y=0.95)



        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_grid[3])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], ax, data_color=palettable.colorbrewer.sequential.Greys_9.mpl_colors[4])
        ax.set_xlim(-5000, 15000)
        ax.set_ylim(-6, 6)
        
        if (i % 2 == 0):
            ax.set_yticks([-3,3])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])

        if i < 8:
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
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.0, hspace=0.17)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  

    # names = ['QSO560',
    #          'QSO569',
    #          'QSO570',
    #          'QSO587',
    #          'QSO589',
    #          'QSO590',
    #          'QSO591',
    #          'QSO038',
    #          'QSO381',
    #          'QSO015',
    #          'QSO601',
    #          'QSO640']
    
    # titles = ['J005202+010129',
    #           'J012257-334844',
    #           'J012337-323828',
    #           'J025055-361635',
    #           'J025634-401300',
    #           'J030211-314030',
    #           'J032944-233835',
    #           'J092747+290721',
    #           'J102510+045247',
    #           'J104915-011038',
    #           'J105651-114122',
    #           'J115302+215118']

    # ylims = [[0.0, 0.7],
    #          [0.0, 0.9],
    #          [0.0, 0.7],
    #          [0.0, 0.4],
    #          [0.0, 0.4],
    #          [0.0, 1.3],
    #          [0.0, 0.9],
    #          [0.0, 1.2],
    #          [0.0, 1.0],
    #          [0.0, 0.7],
    #          [0.0, 0.6],
    #          [0.0, 0.8]]  


    
    titles = ['J123355+031328',
              'J125141+080718',
              'J134104-073947',
              'J141949+060654',
              'J204010-065403',
              'J212912-153841',
              'J214507-303046',
              'J214950-444405',
              'J215052-315824',
              'J223246-363203',
              'J223820-092106',
              'J232539-065259']

    names = ['QSO537',
             'QSO538',
             'QSO611',
             'QSO540',
             'QSO546',
             'QSO169',
             'QSO307',
             'QSO618',
             'QSO619',
             'QSO624',
             'QSO551',
             'QSO629']

    ylims = [[0.0, 0.5],
             [0.0, 1.1],
             [0.0, 1.0],
             [0.0, 0.9],
             [0.0, 0.9],
             [0.0, 0.6],
             [0.0, 0.3],
             [0.0, 0.45],
             [0.0, 0.45],
             [0.0, 0.7],
             [0.0, 0.15],
             [0.0, 0.4]] 

    rebins = np.ones(len(names))




             
    for i in range(len(names)):
              
        ax = plt.subplot(outer_grid[i])
        ax.set_xticks([])
        ax.set_yticks([])
        example_spectra(names[i], ax, rebins[i], plot_model=False, data_color=cs[-1], smooth=100.0)
        ax.set_ylim(ylims[i])
        ax.set_xlim(-5000, 15000)
        ax.set_title(titles[i], size=10, y=0.95)
        # ax.axvline(df.loc[names[i], 'OIII_FIT_VEL_HB_PEAK'], c=cs[0], linestyle=':')
        v4959 = wave2doppler(4960.295*u.AA, w0=4862.721*u.AA).value 
        v5007 = wave2doppler(5008.239*u.AA, w0=4862.721*u.AA).value 
        ax.axvline(df.loc[names[i], 'OIII_FIT_VEL_HB_PEAK'] + v4959, c=cs[0], linestyle=':')
        ax.axvline(df.loc[names[i], 'OIII_FIT_VEL_HB_PEAK'] + v5007, c=cs[0], linestyle=':')

        if i < 9:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 5000, 10000])
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_ticklabels(['0', '5000', '10000'], rotation=45)

        fig.add_subplot(ax)

    fig.text(0.5, 0.02, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'Relative $F_{\lambda}$', rotation=90)
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/example_spectrum_grid_extreme_fe_2.pdf')

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


def parameter_hists():

    
    fig, axs = plt.subplots(3, 1, figsize=figsize(1, vscale=2)) 

    # ----------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)     
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_BROAD_OFF == False]

    
    x = df.OIII_5007_R_80
    
    x_d = np.linspace(-1, 0.8, 1000)
    
    # bandwidths = 10 ** np.linspace(-2, 0, 100)
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
    #                     {'bandwidth': bandwidths},
    #                     cv=LeaveOneOut(len(x)))
    # grid.fit(x[:, None]);
    
    # print grid.best_params_
    
    kde = KernelDensity(bandwidth=0.0774, kernel='gaussian')
    kde.fit(x[:, None])
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    axs[2].fill_between(x_d, np.exp(logprob), color=cs[1])
    axs[2].plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)

    axs[2].set_ylim(-0.1, None)

    axs[2].grid() 

    axs[2].set_xlabel('Asymmetry R')
    axs[2].set_ylabel('PDF')
    
    axs[2].yaxis.set_major_locator(MaxNLocator(5))

    # -------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EQW_FLAG == 0]
    
    x = df.OIII_5007_W80
    print np.mean(x), np.std(x), np.median(x), np.min(x), np.max(x)
    norm = np.std(x)
    x = x / norm
    
    x_d = np.linspace(0, 6, 1000)

    
    # bandwidths = 10 ** np.linspace(-1, 1, 100)
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
    #                     {'bandwidth': bandwidths},
    #                     cv=LeaveOneOut(len(x)))
    # grid.fit(x[:, None]);
    
    # print grid.best_params_
    
    kde = KernelDensity(bandwidth=0.242, kernel='gaussian')
    kde.fit(x[:, None])
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    axs[1].fill_between(x_d*norm, np.exp(logprob), color=cs[1])
    axs[1].plot(x*norm, np.full_like(x, -0.01), '|k', markeredgewidth=1);

    axs[1].set_xlabel(r'$w_{80}$ [km~$\rm{s}^{-1}$]')
    axs[1].set_ylabel('PDF')

    axs[1].set_ylim(-0.05, None)

    axs[1].grid() 

    axs[1].yaxis.set_major_locator(MaxNLocator(5))

    # -----------------------------------------------------
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    
    x = df.OIII_5007_EQW_3
    x[x < 0.01] = 0.01
    x = np.log10(x)
    
    x_d = np.linspace(-2, 3, 1000)
    
    # bandwidths = 10 ** np.linspace(-1, 1, 100)
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
    #                     {'bandwidth': bandwidths},
    #                     cv=LeaveOneOut(len(x)))
    # grid.fit(x[:, None]);
    
    # print grid.best_params_
    
    kde = KernelDensity(bandwidth=0.138, kernel='gaussian')
    kde.fit(x[:, None])
    
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    
    axs[0].fill_between(x_d, np.exp(logprob), color=cs[1])
    axs[0].plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1);

    axs[0].set_xlabel(r'EQW [\AA]')
    axs[0].set_ylabel('PDF')

    axs[0].set_ylim(-0.05, None)

    axs[0].yaxis.set_major_locator(MaxNLocator(5))

    axs[0].grid() 

    
    axs[0].xaxis.set_ticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$']); 

    # ---------------------------------------------------


    labels = ['(a)', '(b)', '(c)']


    for i, label in enumerate(labels):

        axs[i].text(0.9, 0.93, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = axs[i].transAxes)



    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter04/parameter_hists.pdf')

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
    # df = df[df.OIII_BROAD_OFF == False] # otherwise asymetry always zero

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


def oiii_luminosity_z_w80():

    greys = palettable.colorbrewer.sequential.Greys_4.mpl_colors

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.8))
    
    props = {'vmin': 500.0, 
             'vmax': 3000.0, 
             'cmap': 'YlGnBu', 
             'edgecolor':'None'}

    # Mullaney
    
    t = Table.read('/data/lc585/Mullaney13/ALPAKA_liam.fits')
    t = t[t['AGN_TYPE'] == 1] # Only type 1 AGN 
    
    ax.hexbin(t['Z'],
              np.log10(t['OIII_5007_LUM']),
              C=t['w80'],
              vmin=props['vmin'],
              vmax=props['vmax'],
              cmap=props['cmap'],
              gridsize=(5, 25),
              edgecolor=greys[3])
    
    
    # Our sample ------------------------------------------------
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_EQW_FLAG == 0]
    df = df[df.OIII_SNR_FLAG == 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    
    ax.hexbin(df.z_IR.values, 
              df.OIII_5007_LUM_2, 
              C=df.OIII_5007_W80,
              vmin=props['vmin'],
              vmax=props['vmax'],
              cmap=props['cmap'],
              gridsize=(30, 20),
              edgecolor=greys[1])
    
    # Harrison+16 ----------------------------------------------------------------
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/harrison+16.dat', index_col=0)
    
    df.SA.replace(to_replace='<(.*)', value=np.nan, inplace=True, regex=True)
    df.SB.replace(to_replace='<(.*)', value=np.nan, inplace=True, regex=True)
    
    df[['SA', 'SB']] = df[['SA', 'SB']].apply(pd.to_numeric)
    
    df[['SA', 'SB']] = df[['SA', 'SB']].fillna(value=0.0)
    
    df['SA+SB'] = df['SA'] + df['SB']
    
    df = df[df['SA+SB'] > 0.0] 
    
    flx = df['SA+SB'].values * 1.0e-17 * u.erg / u.s / u.cm / u.cm 
    
    lumdist = cosmoWMAP.luminosity_distance(df['zL'].values).to(u.cm) 
    
    lum_oiii = flx * (1.0 + df['zL'].values) * 4.0 * math.pi * lumdist**2 
    
    im = ax.hexbin(df.zL, 
                   np.log10(lum_oiii.value), 
                   C=df.W80,
                   vmin=props['vmin'],
                   vmax=props['vmax'],
                   cmap=props['cmap'],
                   gridsize=(8, 13),
                   edgecolor=greys[2],
                   mincnt=2)
   
    #---------------------------------------------------------------------------

    cb = fig.colorbar(im) 
    cb.set_label(r'w$_{80}$ [km~$\rm{s}^{-1}$]')

    ax.set_xlim(0, 4)
    ax.set_ylim(41, 45)

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'log $L_{\rm[OIII]}$ [erg/s]')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter04/oiii_luminosity_z_w80.pdf')
    plt.show() 

    return None 

def mfica_oiii_weight():

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.mfica_flag == 1]
    
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
    
    fig, axs = plt.subplots(1, 2, figsize=figsize(1, vscale=0.5))
    
    w_norm = df[col_list[3:6]].sum(axis=1) / df[col_list[:6]].sum(axis=1) # sum positive components 
    w_norm = w_norm[~np.isnan(w_norm) & ~np.isinf(w_norm)]
    
    hist = axs[0].hist(w_norm,
                      normed=True,
                      histtype='step',
                      color=cs[1],
                      bins=np.arange(0, 0.5, 0.02),
                      zorder=1) 
        
    fname = '/data/vault/phewett/ICAtest/DR12exp/Spectra/hbeta_2154_c10.weight'
    t = np.genfromtxt(fname)
    
    w_norm = np.sum(t[:, 3:6], axis=1) / np.sum(t[:, :6], axis=1) # sum positive components 
    
    hist = axs[0].hist(w_norm,
                       normed=True,
                       histtype='step',
                       color=cs[8], 
                       bins=np.arange(0, 0.5, 0.02),
                       zorder=0)

    axs[0].set_yticks([]) 
    axs[0].get_xaxis().tick_bottom()

    axs[0].set_xlabel(r"$\displaystyle {\sum_{i=3}^6 w_i} / {\sum_{i=1}^6 w_i}$")
    axs[0].set_ylabel('Normalised counts')

    #----------------------------------------------------------------


    w_norm = df[col_list[5]] / df[col_list[3:6]].sum(axis=1) # sum positive components 
    w_norm = w_norm[~np.isnan(w_norm) & ~np.isinf(w_norm)]
    
    hist = axs[1].hist(w_norm,
                       normed=True,
                       histtype='step',
                       color=cs[1],
                       bins=np.arange(0, 0.8, 0.02),
                       zorder=1) 
        
    
    w_norm = t[:, 5] / np.sum(t[:, 3:6], axis=1) # sum positive components 
    
    hist = axs[1].hist(w_norm,
                       normed=True,
                       histtype='step',
                       color=cs[8], 
                       bins=np.arange(0, 0.8, 0.02),
                       zorder=0)

    axs[1].set_yticks([])
    axs[1].xaxis.set_ticks_position('bottom')
    axs[1].set_xlabel(r"$\displaystyle w_6 / {\sum_{i=3}^6 w_i}$")

    #----------------------------------------------------------------

    axs[0].text(0.9, 0.9, '(a)',
                horizontalalignment='center',
                verticalalignment='center',
                transform = axs[0].transAxes)

    axs[1].text(0.9, 0.9, '(b)',
                horizontalalignment='center',
                verticalalignment='center',
                transform = axs[1].transAxes)

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter04/mfica_oiii_weight.pdf')
    
    plt.show() 

    return None 

def composite():

    from SpectraTools.get_nir_spec import get_nir_spec
    from SpectraTools.make_composite import make_composite
    from SpectraTools.fit_line import fit_line

    set_plot_properties() # change style  

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors



    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.8))

    # Hb --------------------------------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.OIII_FLAG_2 > 0]
    # df = df[df.OIII_EQW_FLAG == 0]
    # df = df[df.OIII_SNR_FLAG == 0]
    # df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_TEMPLATE== False]
 
    wav_new = np.arange(4700.0, 5100.0, 1.0) 

    flux_array = []
    wav_array = [] 
    z_array = [] 
    name_array = [] 
        
    for idx, row in df.iterrows():

        save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', idx, 'OIII') 

        wav, flux = np.genfromtxt(os.path.join(save_dir, 'spec_cont_sub.txt'), unpack=True)

        wav = wav * (1.0 + row.z_IR_OIII_FIT)

        z = row.OIII_FIT_HB_Z
        if row.OIII_Z_FLAG == 1:
            z = row.OIII_FIT_Z_FULL_OIII_PEAK

        flux_array.append(flux)
        wav_array.append(wav)
        z_array.append(z)
        name_array.append(idx)

    wav_array = np.array(wav_array)
    flux_array = np.array(flux_array)
    z_array = np.array(z_array)


    wav_new, flux, err, ns  = make_composite(wav_new,
                                             wav_array, 
                                             flux_array, 
                                             z_array,
                                             names=name_array,
                                             verbose=False)

    # add power-law background
    from SpectraTools.fit_line import PLModel

    bkgdmod = Model(PLModel, 
                    param_names=['amplitude','exponent'], 
                    independent_vars=['x']) 

    bkgdpars = bkgdmod.make_params()

    bkgdpars['exponent'].value = 1.0
    bkgdpars['amplitude'].value = 1.0 

    wav_bkgd = np.arange(4435.0, 5535.0, 1.0)

    spec_min = np.argmin(np.abs(wav_bkgd - 4700.0))
    spec_max = np.argmin(np.abs(wav_bkgd - 5100.0))

    flux_new = bkgdmod.eval(params=bkgdpars, x=wav_bkgd)

    flux_new[spec_min:spec_max] = flux_new[spec_min:spec_max] + flux 
    



    # np.savetxt('composite.dat', np.array([wav_bkgd, flux_new, 0.1*np.ones_like(wav_bkgd)]).T)

    vdat = wave2doppler(wav_new*u.AA, w0=4862.721*u.AA)


    ax.plot(wav_bkgd, flux_new, color=cs[0])


  
    # ax.set_xlim(-20000, 20000)
    # ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'$F_{\lambda}$ [Arbitrary units]')
    plt.grid()

    fig.tight_layout() 
    
    # fig.savefig('/home/lc585/thesis/figures/chapter04/ha_hb_composite.pdf')

    plt.show() 

    return None 

def eqw_cut(): 


    fig, ax = plt.subplots(1, 1, figsize=figsize(1, 0.75))
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    
    df = df[df.OIII_FLAG_2 > 0]
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]

    # df['Binned'] = np.digitize(df.OIII_5007_EQW_3, bins=[10, 20, 40, 60, 80, 100])
    # grouped = df.groupby(by = 'Binned')
    # grouped[['OIII_5007_EQW_3', 'OIII_5007_V10_ERR']].mean()
    
    ax.plot(df['OIII_5007_EQW_3'],
            df['OIII_5007_V10_ERR'],
            linestyle='', 
            markerfacecolor=cs[1],
            markeredgecolor='None',
            markersize=4,
            marker='o')
    
    
    ax.axvline(8, color='black', linestyle=':')
    
    ax.set_xlim(-1, 100)
    ax.set_ylim(-50, 3500)

    ax.set_xlabel(r'EQW [\AA]')
    ax.set_ylabel(r'$\sigma(v_{10})$ [km~$\rm{s}^{-1}$]')


    fig.tight_layout() 
    
    fig.savefig('/home/lc585/thesis/figures/chapter04/eqw_cut.pdf')

    plt.show() 

    return None 



