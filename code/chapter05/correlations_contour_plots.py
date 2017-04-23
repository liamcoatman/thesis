from astropy.table import Table 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 
from PlottingTools.kde_contours import kde_contours
import palettable 
import pandas as pd 
import astropy.units as u 
from matplotlib.ticker import MaxNLocator

cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

def plot():

    set_plot_properties() # change style 

    fig, axs = plt.subplots(3, 2, figsize=figsize(1, vscale=1.5), sharey='col', sharex='row')
    


    t = Table.read('/home/lc585/qsosed/out_bbt_fixed.fits')
    t = t[['NAME', 'BBT', 'IR_UV_RATIO']]
    t.rename_column('NAME', 'SDSS_NAME')
    
    df_fit = t.to_pandas() 
    
    df_bhm = pd.read_csv('/data/lc585/SDSS/civ_bs_basesub.liam', delimiter='|')
    df_bhm.drop('Unnamed: 0', inplace=True, axis=1)
    df_bhm.drop('Unnamed: 18', inplace=True, axis=1)
    
    df = pd.merge(df_fit, df_bhm, how='inner', on='SDSS_NAME')

    Lbol = 3.81 * 10**df['LOGL1350_SDSS']
    Ledd = 3.2e4 * 10**df['CIV_LOGBH_SDSS']
    Lbol = Lbol / (3.846e33*(u.erg/u.s)) # in units of solar luminosity
    EddRatio = Lbol / Ledd
    df['EddRatio_Bias'] = EddRatio
    
    Lbol = 3.81 * 10**df['LOGL1350_SDSS']
    Ledd = 3.2e4 * 10**df['CIV_LOGBH_CORR_HW10']
    Lbol = Lbol / (3.846e33*(u.erg/u.s)) # in units of solar luminosity
    EddRatio = Lbol / Ledd
    df['EddRatio'] = EddRatio


    df = df[~np.isinf(df.EddRatio_Bias)]

    df = df[(df.LOGL1350_SDSS > 45) & (df.LOGL1350_SDSS < 47)]
    df = df[(df.CIV_LOGBH_SDSS > 8) & (df.CIV_LOGBH_SDSS < 10.5)]
    df = df[(np.log10(df.EddRatio_Bias) > -1.5) & (np.log10(df.EddRatio_Bias) < 0.5)]




    kde_contours(df.LOGL1350_SDSS, df.IR_UV_RATIO, axs[0, 0], color='black')
    kde_contours(df.CIV_LOGBH_SDSS, df.IR_UV_RATIO, axs[1, 0], color='black')
    kde_contours(np.log10(df.EddRatio_Bias), df.IR_UV_RATIO, axs[2, 0], color='black')

    kde_contours(df.LOGL1350_SDSS, df.IR_UV_RATIO, axs[0, 1], color='black')
    kde_contours(df.CIV_LOGBH_CORR_HW10, df.IR_UV_RATIO, axs[1, 1], color='black')
    kde_contours(np.log10(df.EddRatio), df.IR_UV_RATIO, axs[2, 1], color='black')
 

    axs[0, 0].set_xlim(45.8, 47)

    axs[0, 0].xaxis.set_major_locator(MaxNLocator(4))

    axs[1, 0].set_xlim(8.25, 10.25)
    axs[2, 0].set_xlim(-1.5, 0.5)
    axs[0, 1].set_xlim(45.8, 47)
    axs[1, 1].set_xlim(8.25, 10.25)
    axs[2, 1].set_xlim(-1.5, 0.5)

    axs[0, 0].set_ylim(0, 0.8)
    axs[0, 1].set_ylim(0, 0.8)
    axs[0, 0].yaxis.set_major_locator(MaxNLocator(4))
    axs[0, 1].yaxis.set_major_locator(MaxNLocator(4))
    
    axs[0, 0].set_ylabel(r'R$_{\rm NIR/UV}$')
    axs[1, 0].set_ylabel(r'R$_{\rm NIR/UV}$')
    axs[2, 0].set_ylabel(r'R$_{\rm NIR/UV}$')



    axs[0, 0].set_xlabel(r'Log L$_{\rm UV}$ [erg~$\rm{s}^{-1}$]')
    axs[1, 0].set_xlabel(r'Log M$_{\rm BH}$ [M$\odot$]')
    axs[2, 0].set_xlabel(r'Log $\lambda_{\rm Edd}$')

    axs[0, 1].set_xlabel(r'Log L$_{\rm UV}$ [erg~$\rm{s}^{-1}$]')
    axs[1, 1].set_xlabel(r'Log M$_{\rm BH}$ [M$\odot$]')
    axs[2, 1].set_xlabel(r'Log $\lambda_{\rm Edd}$')

    

    labels = ['(a)', '(b)', '(b)', '(d)', '(c)', '(e)']

    print len(axs.flatten())

    for i, ax in enumerate(axs.flatten()):

        ax.text(0.1, 0.93, labels[i],
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)

    fig.delaxes(axs[0, 1])



    

    
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    
    fig.savefig('/home/lc585/thesis/figures/chapter05/correlations_contour.pdf')

    plt.show() 

    return None 

