import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PlottingTools.plot_setup import figsize, set_plot_properties
import palettable 
from matplotlib.ticker import MaxNLocator
from scipy import optimize
from astropy import constants as const
import astropy.units as u 

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