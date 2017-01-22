from astropy.table import Table 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 

def bbt_correlations():

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
    
    
    fig, axs = plt.subplots(1,3,figsize=(12,4),sharey=True)
    
    m1, m2 = tab1['LUM_UV'], tab1['BBT']
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[0].contour(X,Y,Z, colors='black')
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axs[0].scatter(x, y, c='black', edgecolor='None', s=3 )
    
    m1, m2 = tab2['LUM_UV'], tab2['BBT']
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[0].contour(X,Y,Z, colors='red')
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axs[0].scatter(x, y, c='red', edgecolor='None', s=3 )
    
    m1, m2 = tab1['LOGBH'], tab1['BBT']
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[1].contour(X,Y,Z, colors='black')
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axs[1].scatter(x, y, c='black', edgecolor='None', s=3 )
    
    m1, m2 = tab2['LOGBH'], tab2['BBT']
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[1].contour(X,Y,Z, colors='red')
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axs[1].scatter(x, y, c='red', edgecolor='None', s=3 )
    
    m1, m2 = tab1['LOGEDD_RATIO'][ tab1['LOGEDD_RATIO'] > -2.0] , tab1['BBT'][ tab1['LOGEDD_RATIO'] > -2.0]
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[2].contour(X,Y,Z, colors='black')
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axs[2].scatter(x, y, c='black', edgecolor='None', s=3 )
    
    m1, m2 = tab2['LOGEDD_RATIO'][ tab2['LOGEDD_RATIO'] > -2.0] , tab2['BBT'][ tab2['LOGEDD_RATIO'] > -2.0]
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[2].contour(X,Y,Z, colors='red')
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axs[2].scatter(x, y, c='red', edgecolor='None', s=3 )
    
    axs[0].set_ylim(800,1800)
    axs[0].set_xlim(45.2,47)
    axs[1].set_xlim(8,10.5)
    axs[2].set_xlim(-1.6,0.5)
    
    axs[0].set_xticks([45.4,45.8,46.2,46.6])
    
    axs[0].set_ylabel('Blackbody Temperature',fontsize=14)
    axs[0].set_xlabel('UV Luminosity',fontsize=14)
    axs[1].set_xlabel('Black Hole Mass',fontsize=14)
    axs[2].set_xlabel('Eddington Ratio',fontsize=14)
    
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[2].tick_params(axis='both', which='major', labelsize=12)
    
    fig.tight_layout()
    
    fig.savefig('/home/lc585/thesis/figures/chapter05/bbt_correlations.pdf')

    plt.show() 

    return None 

