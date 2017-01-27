def plot():

    from astropy.table import Table, join
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    import brewer2mpl
    from PlottingTools.truncate_colormap import truncate_colormap
    from matplotlib import cm
    import matplotlib.gridspec as gridspec
    from scipy import histogram2d
    from PlottingTools.plot_setup import figsize, set_plot_properties

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors
    mycm = truncate_colormap(mycm, 0.0, 0.8)

    set_plot_properties() # change style 

    tab = Table.read('/data/lc585/QSOSED/Results/141124/sample4/out_add_lumcalc.fits')
    tab.sort('Z')
    
    tab = tab[ tab['BBPLSLP_STDERR'] < 0.25]
    
    
    fig = plt.figure(figsize=figsize(1.2,0.4))
    
    gs = gridspec.GridSpec(6, 13)
    
    ax1 = fig.add_subplot(gs[1:5, 0:4])
    ax3 = fig.add_subplot(gs[0,0:4])
    
    ax4 = fig.add_subplot(gs[1:5, 4:8])
    ax6 = fig.add_subplot(gs[0,4:8])
    
    ax7 = fig.add_subplot(gs[1:5, 8:12])
    ax9 = fig.add_subplot(gs[0,8:12])
    
    ax10 = fig.add_subplot(gs[1:5, 12])
    
    fig.subplots_adjust(wspace=0.0)
    fig.subplots_adjust(hspace=0.0)
    
    #histogram definition
    xyrange = [[45.5,48],[-0.5,1.5]] # data range
    bins = [50,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LUM_UV'],tab['BBPLSLP']
    
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
    
    im = ax1.imshow(np.flipud(hh.T),
    	            cmap=mycm,
    	            extent=np.array(xyrange).flatten(), 
    	            interpolation='none',
    	            aspect='auto', 
    	            vmin=thresh, 
    	            vmax=45)

    ax1.scatter(xdat1, ydat1,color=cset[-1])
    
    ax1.set_ylabel(r'$\beta_{\rm NIR}$')
    
    ax1.set_ylim(-0.8,1.5)
    ax1.set_xlim(45.8,47.5)
    
    ax3.hist(tab['LUM_UV'],color=cset[-1],bins=20)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_axis_off()
    
    ax1.get_xaxis().set_ticks([46,46.5,47])
    #ax1.get_yaxis().set_ticks([0.05,0.15,0.25,0.35,0.45])
    
    
    ############################################################################
    
    #histogram definition
    xyrange = [[7.5,10.5],[-0.5,1.5]] # data range
    bins = [30,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LOGBH'],tab['BBPLSLP']
    
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
    
    im = ax4.imshow(np.flipud(hh.T),
    	            cmap=mycm,
    	            extent=np.array(xyrange).flatten(), 
    	            interpolation='none',
    	            aspect='auto', 
    	            vmin=thresh, 
    	            vmax=45)

    ax4.scatter(xdat1, ydat1,color=cset[-1])
    

    ax4.set_ylim(ax1.get_ylim())
    ax4.set_xlim(7.9,10.5)
    ax4.set_xticklabels([7.5,8,8.5,9,9.5,10])
    #ax4.set_yticks([0.05,0.15,0.25,0.35,0.45])
    ax4.set_yticklabels([])
    
    ax6.hist(tab[ tab['LOGBH'] > 6]['LOGBH'],color=cset[-1],bins=20)
    ax6.set_xlim(ax4.get_xlim())
    
    plt.tick_params(axis='both',which='major')
    
    ax4.set_ylim(ax1.get_ylim())
    
    ax6.hist(tab['LOGBH'][tab['LOGBH'] > 6],color=cset[-1],bins=20)
    ax6.set_xlim(ax4.get_xlim())
    ax6.set_axis_off()
    
    ######################################################################################
    
    #histogram definition
    xyrange = [[-2,0.5],[-0.5,1.5]] # data range
    bins = [30,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LOGEDD_RATIO'],tab['BBPLSLP']
    
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
    
    im = ax7.imshow(np.flipud(hh.T),
    	            cmap=mycm,
    	            extent=np.array(xyrange).flatten(), 
    	            interpolation='none',
    	            aspect='auto', 
    	            vmin=thresh, 
    	            vmax=45)

    ax7.scatter(xdat1, ydat1,color=cset[-1])
    

    ax7.set_ylim(ax1.get_ylim())
    ax7.set_xlim(-1.8,0.7)
    
    ax9.hist(tab[ tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'],color=cset[-1],bins=20)
    ax9.set_xlim(ax7.get_xlim())
    ax9.set_axis_off()
    
    plt.tick_params(axis='both',which='major')
    
    #ax7.set_yticks([0.05,0.15,0.25,0.35,0.45])
    ax7.set_yticklabels([])
    #ax7.set_xticklabels([])
    
    ax9.hist(tab['LOGBH'][tab['LOGBH'] > 6],color=cset[-1],bins=20)
    ax9.set_xlim(ax7.get_xlim())
    ax9.set_axis_off()
    
    ax10.hist(tab['BBPLSLP'], orientation='horizontal',color=cset[-1],bins=25)
    ax10.set_ylim(ax1.get_ylim())
    ax10.set_axis_off()
    
    ax4.set_xlabel(r'Log$_{10}$ (Black Hole Mass $M_{\rm BH}$)')
    ax1.set_xlabel(r'Log$_{10} (L_{\rm UV} ({\rm erg/s}))$')
    ax7.set_xlabel(r'Log$_{10}$ (Eddington Ratio $\lambda$)')
    
    axcb = fig.add_axes([0.33,0.1,0.33,0.03]) 
    clb = fig.colorbar(im, cax=axcb,orientation='horizontal') 
    clb.set_label('Number of Objects')
    
    s1 = spearmanr( tab[tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'], tab[tab['LOGEDD_RATIO'] > -2.5]['BBPLSLP'] )[0]
    s2 = spearmanr( tab[ tab['LOGBH'] > 6]['LOGBH'], tab[ tab['LOGBH'] > 6]['BBPLSLP'] )[0]
    s3 = spearmanr( tab['LUM_UV'], tab['BBPLSLP'] )[0]
    
    ax1.text(46.95,1.2,r'$\rho =$ {0:.2f}'.format(s3))
    ax4.text(9.7,1.2,r'$\rho =$ {0:.2f}'.format(s2))
    ax7.text(-0.1,1.2,r'$\rho =$ {0:.2f}'.format(s1))

    fig.savefig('/home/lc585/thesis/figures/chapter06/correlations_highz.pdf')


    plt.show() 

    return None 