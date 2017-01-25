def plot():  

    fig = plt.figure(figsize=(8,12))
    
    gs = gridspec.GridSpec(9, 5)
    
    ax1 = fig.add_subplot(gs[1:5, 0:4])
    ax2 = fig.add_subplot(gs[5:, 0:4])
    ax3 = fig.add_subplot(gs[0,0:4])
    ax4 = fig.add_subplot(gs[1:5,4])
    ax5 = fig.add_subplot(gs[5:,4])
    
    fig.subplots_adjust(wspace=0.0)
    fig.subplots_adjust(hspace=0.0)
    
    #histogram definition
    xyrange = [[-0.4,1.6],[0,2]] # data range
    bins = [120,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['Z'],tab['RATIO_IR_UV']
    
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
    
    
    ax1.set_ylabel(r'$R_{{\rm NIR}/{\rm UV}}$',fontsize=14)
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
    ax1.set_ylim(0.,2.0)
    ax1.set_xlim(1,1.5)
    tabtmp = tab
    tabtmp.sort('Z')
    xdat = running.RunningMedian(np.array(tabtmp['Z']),101)
    ydat = running.RunningMedian(np.array(tabtmp['RATIO_IR_UV']),101)
    ax1.plot(xdat[::100],ydat[::100],color='black',linewidth=2.0)
    ax1.axhline(np.median(tab['RATIO_IR_UV']),color='black',linestyle='--')
    ax1.axhline(np.percentile(tab['RATIO_IR_UV'],70),color='black',linestyle='--')
    ax1.axhline(np.percentile(tab['RATIO_IR_UV'],30),color='black',linestyle='--')
    ax1.get_xaxis().set_visible(False)
    
    ax3.hist(tab['Z'],color=cset[-1],bins=20)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_axis_off()
    
    ax4.hist(tab['RATIO_IR_UV'], orientation='horizontal',color=cset[-1],bins=np.arange(0,2,0.1))
    ax4.set_ylim(ax1.get_ylim())
    ax4.set_axis_off()
    
    #histogram definition
    xyrange = [[-0.4,1.6],[500,2000]] # data range
    bins = [100,50] # number of bins
    thresh = 4  #density threshold
    
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
    
    im = ax2.imshow(np.flipud(hh.T),
    	            cmap=mycm,
    	            extent=np.array(xyrange).flatten(), 
    	            interpolation='none',
    	            aspect='auto', 
    	            vmin=thresh, 
    	            vmax=45)

    ax2.scatter(xdat1, ydat1,color=cset[-1])
    
    axcb = fig.add_axes([0.13,0.05,0.6,0.02]) 
    clb = fig.colorbar(im, cax=axcb,orientation='horizontal') 
    clb.set_label('Number of Objects',fontsize=14)
    
    ax2.set_xlabel(r'Redshift $z$',fontsize=14)
    ax2.set_ylabel(r'$T_{\mathrm{BB}}$',fontsize=14)
    
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
    ax2.set_ylim(500,1900)
    ax2.set_xlim(1,1.5)
    tabtmp = tab
    tabtmp.sort('Z')
    xdat = running.RunningMedian(np.array(tabtmp['Z']),101)
    ydat = running.RunningMedian(np.array(tabtmp['BBT']),101)
    ax2.plot(xdat[::100],ydat[::100],color='black',linewidth=2.0)
    ax2.axhline(1200,color='black',linestyle='--')
    ax2.axhline(1100,color='black',linestyle='--')
    ax2.axhline(1300,color='black',linestyle='--')
    
    ax5.hist(tab['BBT'], orientation='horizontal',color=cset[-1],bins=20)
    ax5.set_ylim(ax2.get_ylim())
    ax5.set_axis_off()
    
    plt.tick_params(axis='both',which='major',labelsize=10)

    return None 