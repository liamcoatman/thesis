from astropy.table import Table 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 
from PlottingTools.kde_contours import kde_contours
import palettable 
cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

def plot():

    set_plot_properties() # change style 

    fig, axs = plt.subplots(3, 2, figsize=figsize(1, vscale=1.5), sharey='col', sharex='row')
    

    tab1 = Table.read('/data/lc585/QSOSED/Results/150224/sample1/out_add.fits')

    tab1 = tab1[ ~np.isnan(tab1['BBT_STDERR'])]
    tab1 = tab1[ tab1['BBT_STDERR'] < 500. ]
    tab1 = tab1[ tab1['BBT_STDERR'] > 5.0 ]
    tab1 = tab1[ (tab1['LUM_IR_SIGMA']*tab1['RATIO_IR_UV']) < 0.4]
    
    tab2 = Table.read('/data/lc585/QSOSED/Results/150224/sample2/out_add.fits')
    
    tab2 = tab2[ ~np.isnan(tab2['BBT_STDERR'])]
    tab2 = tab2[ tab2['BBT_STDERR'] < 500. ]
    tab2 = tab2[ tab2['BBT_STDERR'] > 5.0 ]
    tab2 = tab2[ (tab2['LUM_IR_SIGMA']*tab2['RATIO_IR_UV']) < 0.4]
    
    kde_contours(tab1['LUM_UV'], tab1['RATIO_IR_UV'], axs[0, 0], color=cs[-1])
    kde_contours(tab2['LUM_UV'], tab2['RATIO_IR_UV'], axs[0, 0], color=cs[1])
    kde_contours(tab1['LOGBH'], tab1['RATIO_IR_UV'], axs[1, 0], color=cs[-1])
    kde_contours(tab2['LOGBH'], tab2['RATIO_IR_UV'], axs[1, 0], color=cs[1])
    kde_contours(tab1['LOGEDD_RATIO'][ tab1['LOGEDD_RATIO'] > -2.0] , 
                 tab1['RATIO_IR_UV'][ tab1['LOGEDD_RATIO'] > -2.0], 
                 axs[2, 0], color=cs[-1])
    kde_contours(tab2['LOGEDD_RATIO'][ tab2['LOGEDD_RATIO'] > -2.0], 
                 tab2['RATIO_IR_UV'][ tab2['LOGEDD_RATIO'] > -2.0],
                 axs[2, 0],
                 color=cs[1])


    tab1 = Table.read('/data/lc585/QSOSED/Results/141209/sample1/out_add.fits')
    
    tab1 = tab1[ ~np.isnan(tab1['BBT_STDERR'])]
    tab1 = tab1[ tab1['BBT_STDERR'] < 500. ]
    tab1 = tab1[ tab1['BBT_STDERR'] > 5.0 ]
    tab1 = tab1[ (tab1['LUM_IR_SIGMA']*tab1['RATIO_IR_UV']) < 0.4]
    
    tab2 = Table.read('/data/lc585/QSOSED/Results/150211/sample2/out_add.fits')
    
    tab2 = tab2[ ~np.isnan(tab2['BBT_STDERR'])]
    tab2 = tab2[ tab2['BBT_STDERR'] < 500. ]
    tab2 = tab2[ tab2['BBT_STDERR'] > 5.0 ]
    tab2 = tab2[ (tab2['LUM_IR_SIGMA']*tab2['RATIO_IR_UV']) < 0.4]

    kde_contours(tab1['LUM_UV'], tab1['BBT'], axs[0, 1], color=cs[-1])
    kde_contours(tab2['LUM_UV'], tab2['BBT'], axs[0, 1], color=cs[1])
    kde_contours(tab1['LOGBH'], tab1['BBT'], axs[1, 1], color=cs[-1])
    kde_contours(tab2['LOGBH'], tab2['BBT'], axs[1, 1], color=cs[1])
    kde_contours(tab1['LOGEDD_RATIO'][ tab1['LOGEDD_RATIO'] > -2.0] , 
                 tab1['BBT'][ tab1['LOGEDD_RATIO'] > -2.0], 
                 axs[2, 1], color=cs[-1])
    kde_contours(tab2['LOGEDD_RATIO'][ tab2['LOGEDD_RATIO'] > -2.0], 
                 tab2['BBT'][ tab2['LOGEDD_RATIO'] > -2.0],
                 axs[2, 1],
                 color=cs[1])

    axs[0, 0].set_xlim(45, 47)
    axs[1, 0].set_xlim(8, 10.5)
    axs[2, 0].set_xlim(-1.5, 0.5)

    axs[0, 0].set_ylim(0, 0.8)
    axs[0, 1].set_ylim(800, 1800)
 
    
    
    axs[0, 0].set_ylabel(r'R$_{\rm NIR/UV}$')
    axs[0, 1].set_ylabel(r'T$_{\rm BB}$ [K]')

    axs[1, 0].set_ylabel(r'R$_{\rm NIR/UV}$')
    axs[1, 1].set_ylabel(r'T$_{\rm BB}$ [K]')

    axs[2, 0].set_ylabel(r'R$_{\rm NIR/UV}$')
    axs[2, 1].set_ylabel(r'T$_{\rm BB}$ [K]')


    axs[0, 0].set_xlabel(r'Log L$_{\rm UV}$ [erg~$\rm{s}^{-1}$]')
    axs[1, 0].set_xlabel(r'Log M$_{\rm BH}$ [M$\odot$]')
    axs[2, 0].set_xlabel(r'Log $\lambda_{\rm Edd}$')

    axs[0, 1].set_xlabel(r'Log L$_{\rm UV}$ [erg~$\rm{s}^{-1}$]')
    axs[1, 1].set_xlabel(r'Log M$_{\rm BH}$ [M$\odot$]')
    axs[2, 1].set_xlabel(r'Log $\lambda_{\rm Edd}$')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.25)
    
    fig.savefig('/home/lc585/thesis/figures/chapter05/correlations_contour.pdf')

    plt.show() 

    return None 