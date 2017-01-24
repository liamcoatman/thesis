


def luminosity_z(): 

    from astropy.table import Table, join 
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import cPickle as pickle
    import os
    import time
    import numpy.ma as ma 
    from PlottingTools.plot_setup import figsize, set_plot_properties
    import palettable 
    import matplotlib.patches as patches
    from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
    import pandas as pd 

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.0
        
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # no labels
    nullfmt = NullFormatter() 
    
    fig = plt.figure(figsize=figsize(0.8, vscale=1.0))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    axHistx.set_xticks([])
    axHistx.set_yticks([])
    axHisty.set_xticks([])
    axHisty.set_yticks([])
    
    
    t = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits')
    t = t[t['LOGLBOL'] > 0.0]
    t = t[t['Z_HW'] > 1.0]

    m1, m2 = t['Z_HW'], t['LOGLBOL'] # Richards 2006 Bolometric correction

    xmin = 1.0
    xmax = 5.0
    ymin = 44.0
    ymax = 48.0
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    
    kernel = stats.gaussian_kde(values)
    
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axScatter.contour(X, Y, Z, colors=[cs[-1]], levels=[0.02, 0.05, 0.13, 0.2, 0.4, 0.8, 1.2])
    
    # threshold = CS.levels[0]
    threshold = 0.02
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)
    
    # plot unmasked points
    axScatter.plot(x, 
                   y, 
                   markerfacecolor=cs[-1], 
                   markeredgecolor='None', 
                   linestyle='', 
                   marker='o', 
                   markersize=2, 
                   label='SDSS DR7')
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df.drop_duplicates(subset=['ID'], inplace=True)
    df = df[df.SPEC_NIR != 'None']
    df.dropna(subset=['LogL5100'], inplace=True)
    
    axScatter.scatter(df.z_IR, 
                      np.log10(9.26) + df.LogL5100, 
                      c=cs[1], 
                      edgecolor='None', 
                      label='This work', 
                      zorder=10, 
                      s=20)
    
    axScatter.set_xlabel(r'Redshift $z$')
    axScatter.set_ylabel(r'log $L_{\mathrm{Bol}}$ [erg/s]')
    
    legend = axScatter.legend(frameon=True, scatterpoints=1, numpoints=1, loc='lower right') 
    
    axHistx.hist(m1, 
                 bins=np.arange(0.0, 5.0, 0.25), 
                 facecolor=cs[-1], 
                 edgecolor='None', 
                 alpha=0.4, 
                 normed=True)
    
    axHisty.hist(m2, 
                 bins=np.arange(44, 49, 0.25), 
                 facecolor=cs[-1], 
                 edgecolor='None', 
                 orientation='horizontal', 
                 alpha=0.4, 
                 normed=True)
    
    axHistx.hist(df.z_IR, 
                 bins=np.arange(1, 5.0, 0.25), 
                 histtype='step', 
                 edgecolor=cs[1], 
                 normed=True, 
                 lw=2)
    
    axHisty.hist(np.log10(9.26) + np.array(df.LogL5100), 
                 bins=np.arange(44, 49, 0.25), 
                 histtype='step', 
                 edgecolor=cs[1], 
                 normed=True, 
                 lw=2, 
                 orientation='horizontal')
    
    axScatter.set_xlim(1.0, 4.5)
    axScatter.set_ylim(44, 48.5)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    axHistx.set_ylim(0, 1.5)
    axHisty.set_xlim(0, 1.2)   

    fig.savefig('/home/lc585/thesis/figures/chapter02/luminosity_z.pdf')

    plt.show() 

    return None  

