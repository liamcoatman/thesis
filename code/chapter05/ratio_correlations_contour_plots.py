from astropy.table import Table 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 
from PlottingTools.kde_contours import kde_contours

def plot():

    set_plot_properties() # change style 

    fig, axs = plt.subplots(2, 3, figsize=figsize(1, vscale=0.8), sharex='col', sharey='row')
    

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
    
    kde_contours(tab1['LUM_UV'], tab1['RATIO_IR_UV'], axs[0, 0])
    kde_contours(tab2['LUM_UV'], tab2['RATIO_IR_UV'], axs[0, 0], color='red')
    kde_contours(tab1['LOGBH'], tab1['RATIO_IR_UV'], axs[0, 1])
    kde_contours(tab2['LOGBH'], tab2['RATIO_IR_UV'], axs[0, 1], color='red')
    kde_contours(tab1['LOGEDD_RATIO'][ tab1['LOGEDD_RATIO'] > -2.0] , 
                 tab1['RATIO_IR_UV'][ tab1['LOGEDD_RATIO'] > -2.0], 
                 axs[0, 2])
    kde_contours(tab2['LOGEDD_RATIO'][ tab2['LOGEDD_RATIO'] > -2.0], 
                 tab2['RATIO_IR_UV'][ tab2['LOGEDD_RATIO'] > -2.0],
                 axs[0, 2],
                 color='red')


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

    kde_contours(tab1['LUM_UV'], tab1['BBT'], axs[1, 0])
    kde_contours(tab2['LUM_UV'], tab2['BBT'], axs[1, 0], color='red')
    kde_contours(tab1['LOGBH'], tab1['BBT'], axs[1, 1])
    kde_contours(tab2['LOGBH'], tab2['BBT'], axs[1, 1], color='red')
    kde_contours(tab1['LOGEDD_RATIO'][ tab1['LOGEDD_RATIO'] > -2.0] , 
                 tab1['BBT'][ tab1['LOGEDD_RATIO'] > -2.0], 
                 axs[1, 2])
    kde_contours(tab2['LOGEDD_RATIO'][ tab2['LOGEDD_RATIO'] > -2.0], 
                 tab2['BBT'][ tab2['LOGEDD_RATIO'] > -2.0],
                 axs[1, 2],
                 color='red')

    axs[0, 0].set_xlim(45, 47)
    axs[0, 1].set_xlim(8, 10.5)
    axs[0, 2].set_xlim(-1.5, 0.5)

    axs[1, 0].set_ylim(800, 1800)
    axs[0, 0].set_ylim(0, 0.8)
 
    
    
    axs[0, 0].set_ylabel(r'$R_{NIR/UV}$')
    axs[1, 0].set_ylabel(r'$T_{BB}$')
    axs[1, 0].set_xlabel('UV Luminosity')
    axs[1, 1].set_xlabel('Black Hole Mass')
    axs[1, 2].set_xlabel('Eddington Ratio')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    
    fig.savefig('/home/lc585/thesis/figures/chapter06/correlations_contour.pdf')

    plt.show() 

    return None 