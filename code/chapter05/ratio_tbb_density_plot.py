def plot():

    from PlottingTools.plot_setup_thesis import figsize, set_plot_properties    
    from astropy.table import Table
    import matplotlib.pyplot as plt
    from scipy import histogram2d
    import numpy as np
    from matplotlib import cm
    from PlottingTools.truncate_colormap import truncate_colormap
    import brewer2mpl
    from PlottingTools import running
    import collections
    from PlottingTools.kde_contours import kde_contours
    from PlottingTools.scatter_hist import scatter_hist
    import palettable 

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    fig, axScatter, axHistx, axHisty = scatter_hist(figsize=figsize(1, 1),
                                                    left=0.12,
                                                    bottom=0.12) 

    set_plot_properties() # change style 

    tab = Table.read('/home/lc585/qsosed/out.fits')

    # tab = tab[ ~np.isnan(tab['BBT_STDERR'])]
    # tab = tab[ tab['BBT_STDERR'] < 500. ]
    # tab = tab[ tab['BBT_STDERR'] > 5.0 ]
    # tab = tab[ (tab['LUM_IR_SIGMA']*tab['RATIO_IR_UV']) < 1.]
    
    
    #data definition
    xdat, ydat = tab['BBT']*1e3, tab['IR_UV_RATIO']


    bad = np.isnan(xdat) | np.isnan(ydat)
    xdat = xdat[~bad]
    ydat = ydat[~bad]

    bad = (xdat > 2000) | (xdat < 500) | (ydat < 0) | (ydat > 1)

    xdat = xdat[~bad]
    ydat = ydat[~bad]

        
    kde_contours(xdat, ydat, axScatter, filled=True, lims=(600, 1800, 0, 0.8))
    
    axScatter.set_xlabel(r'$T_{\rm BB}$')
    axScatter.set_ylabel(r'$R_{{\rm NIR}/{\rm UV}}$')
    
    axScatter.set_ylim(0,0.8)
    axScatter.set_xlim(800,1600)

    axHisty.hist(ydat, 
                 bins=np.arange(0, 0.8, 0.05),
                 facecolor=cs[1], 
                 histtype='stepfilled',
                 edgecolor='black', 
                 orientation='horizontal', 
                 normed=True)
    
    axHistx.hist(xdat, 
                 bins=np.arange(800, 1600, 50),
                 histtype='stepfilled', 
                 edgecolor='black',
                 facecolor=cs[1], 
                 normed=True)

    # Sample from single (T,Norm) with gaussian errors on photometry. 
    # Mock magnitude file made in model.py and then fit in runsingleobjfit.py.
    
    # tab2 = Table.read('/data/lc585/QSOSED/Results/150309/sample5/out_add.fits')
    
    
    
    # axScatter.scatter(tab2['BBT'],
    #                   tab2['RATIO_IR_UV'],
    #                   edgecolor='None',
    #                   color=cs[0], 
    #                   s=8,
    #                   zorder=10)


    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axHistx.set_ylim(0, 5e-3)
    axHisty.set_xlim(0, 4)



    fig.savefig('/home/lc585/thesis/figures/chapter05/ratio_tbb_density.pdf')

    plt.show() 

    return None 



    

    



    

