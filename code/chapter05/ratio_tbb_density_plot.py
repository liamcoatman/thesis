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

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    mycm = truncate_colormap(mycm, 0.0, 0.8)
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors

    set_plot_properties() # change style 

    tab = Table.read('/data/lc585/QSOSED/Results/150309/sample4/out_add.fits')

    tab = tab[ ~np.isnan(tab['BBT_STDERR'])]
    tab = tab[ tab['BBT_STDERR'] < 500. ]
    tab = tab[ tab['BBT_STDERR'] > 5.0 ]
    tab = tab[ (tab['LUM_IR_SIGMA']*tab['RATIO_IR_UV']) < 1.]
    
    fig = plt.figure(figsize=figsize(1, vscale=0.8))
    ax = fig.add_subplot(1,1,1)
    
    #histogram definition
    xyrange = [[0,3000],[0,2]] # data range
    bins = [80,60] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['BBT'],tab['RATIO_IR_UV']
    
    # histogram the data
    hh, locx, locy = histogram2d(xdat, ydat, range=xyrange, bins=bins)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)
    
    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
    
    im = ax.imshow(np.flipud(hh.T),
                   cmap=mycm,
                   extent=np.array(xyrange).flatten(), 
                   interpolation='none',aspect='auto')
    cb = plt.colorbar(im)
    cb.set_label('Number of Objects')
    
    ax.scatter(xdat1, ydat1,color=cset[-1],s=8)
    
    ax.set_xlabel(r'$T_{\rm BB}$')
    ax.set_ylabel(r'$R_{{\rm NIR}/{\rm UV}}$')
    
    ax.set_ylim(0,1)
    ax.set_xlim(600,1900)

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter06/ratio_tbb_density.pdf')

    plt.show() 

    return None 