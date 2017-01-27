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

def plot():

    tab = Table.read('/data/lc585/QSOSED/Results/141203/sample1/out_add.fits')
    tab = tab[ tab['BBT_STDERR'] < 200.0 ]
    tab = tab[ tab['BBFLXNRM_STDERR'] < 0.05]
    tab = tab[ tab['CHI2_RED'] < 3.0]
    tab = tab[ tab['LUM_IR'] > 40.0] # just gets rid of one annoying point

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors
    mycm = truncate_colormap(mycm, 0.0, 0.8)

    set_plot_properties() # change style 

    fig = plt.figure(figsize=figsize(1.5, 0.7))
    
    gs = gridspec.GridSpec(9, 13)
    
    ax1 = fig.add_subplot(gs[1:5, 0:4])
    ax2 = fig.add_subplot(gs[5:, 0:4])
    ax3 = fig.add_subplot(gs[0,0:4])
    
    ax4 = fig.add_subplot(gs[1:5, 4:8])
    ax5 = fig.add_subplot(gs[5:, 4:8])
    ax6 = fig.add_subplot(gs[0,4:8])
    
    ax7 = fig.add_subplot(gs[1:5, 8:12])
    ax8 = fig.add_subplot(gs[5:, 8:12])
    ax9 = fig.add_subplot(gs[0,8:12])
    
    ax10 = fig.add_subplot(gs[1:5, 12])
    ax11 = fig.add_subplot(gs[5:, 12])
    
    
    fig.subplots_adjust(wspace=0.0)
    fig.subplots_adjust(hspace=0.0)
    
    #histogram definition
    xyrange = [[44,48],[0,2]] # data range
    bins = [80,40] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LUM_UV'],tab['RATIO_IR_UV']
    
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
    
    ax1.set_ylim(0.,2.)
    ax1.set_xlim(45.25,47)
    
    ax3.hist(tab['LUM_UV'],color=cset[-1],bins=20)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_axis_off()
    
    #histogram definition
    xyrange = [[44,48],[200,2000]] # data range
    bins = [90,35] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LUM_UV'],tab['BBT']
    
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
    
    ax2.set_xlabel(r'Log$_{10} (L_{\rm UV} ({\rm erg/s}))$',fontsize=14)
    ax2.set_ylabel(r'$T_{\mathrm{BB}}$',fontsize=14)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(500,2100)
    
    plt.tick_params(axis='both',which='major',labelsize=10)
    
    ax1.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([600,800,1000,1200,1400,1600,1800,2000])
    ax2.get_xaxis().set_ticks([45.4,45.6,45.8,46.0,46.2,46.4,46.6,46.8])
    
    #############################################################################
    
    #histogram definition
    xyrange = [[7.5,10.5],[0,2]] # data range
    bins = [30,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LOGBH'],tab['RATIO_IR_UV']
    
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
    ax4.set_xlim(7.5,10.5)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    
    ax6.hist(tab[ tab['LOGBH'] > 6]['LOGBH'],color=cset[-1],bins=20)
    ax6.set_xlim(ax4.get_xlim())
    ax6.set_axis_off()
    
    #histogram definition
    xyrange = [[7.5,10.5],[500,2000]] # data range
    bins = [50,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LOGBH'],tab['BBT']
    
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
    
    im = ax5.imshow(np.flipud(hh.T),
                    cmap=mycm,
                    extent=np.array(xyrange).flatten(), 
                    interpolation='none',
                    aspect='auto', 
                    vmin=thresh, 
                    vmax=45)

    ax5.scatter(xdat1, ydat1,color=cset[-1])
    
    ax5.set_xlabel(r'Log$_{10}$ (Black Hole Mass $M_{\rm BH}$)')
    
    ax5.set_yticklabels([]) 
    
    # ax5.hist(tab['BBT'], orientation='horizontal',color=cset[-1],bins=20)
    # ax5.set_ylim(ax2.get_ylim())
    # ax5.set_axis_off()
    
    plt.tick_params(axis='both',which='major')
    
    
    ax4.set_ylim(ax1.get_ylim())
    ax5.set_ylim(ax2.get_ylim())
    ax5.set_xlim(ax4.get_xlim())
    
    ax6.hist(tab['LOGBH'][tab['LOGBH'] > 6],color=cset[-1],bins=20)
    ax6.set_xlim(ax5.get_xlim())
    ax6.set_axis_off()
    
    ######################################################################################
    
    #histogram definition
    xyrange = [[-2,0.5],[0,2]] # data range
    bins = [30,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LOGEDD_RATIO'],tab['RATIO_IR_UV']
    
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
    
    im = ax7.imshow(np.flipud(hh.T),cmap=mycm,extent=np.array(xyrange).flatten(), interpolation='none',aspect='auto', vmin=thresh, vmax=45)
    ax7.scatter(xdat1, ydat1,color=cset[-1])
    
    ax7.set_ylim(ax1.get_ylim())
    ax7.set_xlim(-2,0.5)
    
    ax9.hist(tab[ tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'],color=cset[-1],bins=20)
    ax9.set_xlim(ax7.get_xlim())
    ax9.set_axis_off()
    
    #histogram definition
    xyrange = [[-2,0.5],[500,2000]] # data range
    bins = [50,30] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    xdat, ydat = tab['LOGEDD_RATIO'],tab['BBT']
    
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
    
    im = ax8.imshow(np.flipud(hh.T),cmap=mycm,extent=np.array(xyrange).flatten(), interpolation='none',aspect='auto', vmin=thresh, vmax=45)
    ax8.scatter(xdat1, ydat1,color=cset[-1])
    
    axcb = fig.add_axes([0.33,0.05,0.33,0.02]) 
    clb = fig.colorbar(im, cax=axcb,orientation='horizontal') 
    clb.set_label('Number of Objects')
    
    ax8.set_xlabel(r'Log$_{10}$ (Eddington Ratio $\lambda$)')
    
    ax8.set_ylim(ax2.get_ylim())
    ax8.set_xlim(ax7.get_xlim())
    
    plt.tick_params(axis='both',which='major')
    
    ax8.set_yticklabels([])
    ax8.set_xticklabels(['','-1.5','-1.0','-0.5','0.0','0.5'])
    ax7.set_yticklabels([])
    ax7.set_xticklabels([])
    
    ax9.hist(tab['LOGBH'][tab['LOGBH'] > 6],color=cset[-1],bins=20)
    ax9.set_xlim(ax8.get_xlim())
    ax9.set_axis_off()
    
    ax10.hist(tab['RATIO_IR_UV'], orientation='horizontal',color=cset[-1],bins=np.arange(0,2,0.1))
    ax10.set_ylim(ax1.get_ylim())
    ax10.set_axis_off()
    
    ax11.hist(tab['BBT'], orientation='horizontal',color=cset[-1],bins=20)
    ax11.set_ylim(ax2.get_ylim())
    ax11.set_axis_off()
    
    s1 = spearmanr( tab[tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'], tab[tab['LOGEDD_RATIO'] > -2.5]['RATIO_IR_UV'] )[0]
    s2 = spearmanr( tab[ tab['LOGBH'] > 6]['LOGBH'], tab[ tab['LOGBH'] > 6]['RATIO_IR_UV'] )[0]
    s3 = spearmanr( tab['LUM_UV'], tab['RATIO_IR_UV'] )[0]
    s4 = spearmanr( tab[tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'], tab[tab['LOGEDD_RATIO'] > -2.5]['BBT'] )[0]
    s5 = spearmanr( tab[ tab['LOGBH'] > 6]['LOGBH'], tab[ tab['LOGBH'] > 6]['BBT'] )[0]
    s6 = spearmanr( tab['LUM_UV'], tab['BBT'] )[0]
    
    ax1.text(46.5,0.2,r'$\rho =$ {0:.2f}'.format(s3))
    ax4.text(9.7,0.2,r'$\rho =$ {0:.2f}'.format(s2))
    ax7.text(-0.1,0.2,r'$\rho =$ {0:.2f}'.format(s1))
    ax2.text(46.5,1900,r'$\rho =$ {0:.2f}'.format(s6))
    ax5.text(9.7,1900,r'$\rho =$ {0:.2f}'.format(s5))
    ax8.text(-0.1,1900,r'$\rho =$ {0:.2f}'.format(s4))

    fig.savefig('/home/lc585/thesis/figures/chapter06/correlations.pdf')

    plt.show() 

    return None 