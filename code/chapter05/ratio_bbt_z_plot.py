from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy import histogram2d
from PlottingTools.truncate_colormap import truncate_colormap
from PlottingTools import running 
from PlottingTools.plot_setup import figsize, set_plot_properties

def plot():  

    set_plot_properties() # change style 

    tab = Table.read('/data/lc585/QSOSED/Results/141203/sample1/out_add.fits')
    tab = tab[ tab['BBT_STDERR'] < 200.0 ]
    tab = tab[ tab['BBFLXNRM_STDERR'] < 0.05]
    tab = tab[ tab['CHI2_RED'] < 3.0]
    tab = tab[ tab['LUM_IR'] > 40.0] # just gets rid of one annoying point

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors
    mycm = truncate_colormap(mycm, 0.0, 0.8)

    fig = plt.figure(figsize=figsize(0.8, 1.8))
    
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
    
    
    ax1.set_ylabel(r'$R_{{\rm NIR}/{\rm UV}}$')

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
    clb.set_label('Number of Objects')
    
    ax2.set_xlabel(r'Redshift $z$')
    ax2.set_ylabel(r'$T_{\mathrm{BB}}$')

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

    fig.savefig('/home/lc585/thesis/figures/chapter06/ratio_bbt_z.pdf')
    plt.show() 
    
    return None 