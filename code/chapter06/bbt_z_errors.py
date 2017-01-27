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

    tab = Table.read('/data/lc585/QSOSED/Results/150205/sample1/out_add.fits')
    
    tab = tab[ tab['Z'] > 0.5 ]
    tab = tab[ tab['BBT_STDERR'] < 1000.]
    tab = tab[ tab['BBFLXNRM_STDERR'] < 1.]
    
    tab = tab[ tab['CHI2_RED'] < 2.0]
    
    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.9))
    
    #histogram definition
    xyrange = [[0,2],[100,5000]] # data range
    bins = [140,50] # number of bins
    thresh = 3  #density threshold
    
    #data definition
    xdat, ydat = tab['Z'],tab['BBT']
    
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
                   interpolation='none',
                   aspect='auto', 
                   )

    ax.scatter(xdat1, ydat1,color=cset[-1],s=8)
    
    #axcb = fig.add_axes([0.13,0.01,0.6,0.02]) 
    clb = fig.colorbar(im,orientation='horizontal') 
    clb.set_label('Number of Objects')
    
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$T_{\mathrm{BB}}$')
    
    
    ax.set_ylim(700,4000)
    ax.set_xlim(0.5,2)
    
    tabtmp = tab
    tabtmp.sort('Z')
    xdat = running.RunningMedian(np.array(tabtmp['Z']),101)
    ydat = running.RunningMedian(np.array(tabtmp['BBT']),101)
    ax.plot(xdat[::100],ydat[::100],color='black',linewidth=2.0)
    
    
    # Generate magnitude file in model.py
    # Fit in runsingleobjfit.py (comment out load dat section)
    
    modtab = Table.read('/data/lc585/QSOSED/Results/150107/sample2/out_lumcalc.fits')
    
    ax.plot(modtab['Z'],modtab['BBT'],color='black',linestyle='--', linewidth=2.0)
    
    ax.grid(which='major',c='k')
    
    #histogram definition
    x_range = [0.5,2.9] # data range
    bins = [12] # number of bins
    dat = np.vstack([tab['Z'],tab['BBT_STDERR']])
    
    # histogram the data
    h, locx = np.histogram(dat[0,:], range=x_range, bins=bins[0])
    posx = np.digitize(dat[0,:], locx)
    
    xindices = collections.defaultdict(set)
    for index, bin in enumerate(posx):
        xindices[bin].add(index)
    
    ys = []
    for i in range(bins[0]):
        ys.append(np.mean(dat[1,list(xindices[i+1])]))
            
    ax.errorbar(locx[:-1] + 0.1,np.repeat(3500,12),yerr=ys,linestyle='',color='black')
    
    ax.axvline(1.0, color='black')
    ax.axvline(1.5, color='black')

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter06/bbt_z_errors.pdf')

    plt.show() 

    return None 