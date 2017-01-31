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

    tab = Table.read('/data/lc585/QSOSED/Results/141209/sample1/out_add.fits')

    #tab = Table.read('/data/lc585/QSOSED/Results/150115/sample2/out_add.fits') # derened before imag cut
    #tab = Table.read('/data/lc585/QSOSED/Results/150115/sample3/out_add.fits') # not deredened before cut
    
    tab = tab[ ~np.isnan(tab['BBT_STDERR'])]
    tab = tab[ tab['BBT_STDERR'] < 500. ]
    tab = tab[ tab['BBT_STDERR'] > 5.0 ]
    tab = tab[ (tab['LUM_IR_SIGMA']*tab['RATIO_IR_UV']) < 0.4]

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    mycm = truncate_colormap(mycm, 0.0, 0.8)
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors

    set_plot_properties() # change style 

    fig = plt.figure(figsize=figsize(1, 0.6))
    
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
    fig.subplots_adjust(bottom=0.2)
    
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
    
    im = ax1.imshow(np.flipud(hh.T),cmap=mycm,extent=np.array(xyrange).flatten(), interpolation='none',aspect='auto', vmin=thresh, vmax=45)
    ax1.scatter(xdat1, ydat1,color=cset[-1])
    
    ax1.set_ylabel(r'$T_{\mathrm{BB}}$')
    
    ax1.set_ylim(500,2000)
    ax1.set_xlim(45,47)
    
    ax3.hist(tab['LUM_UV'],color=cset[-1],bins=20)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_axis_off()
    
    ax1.get_xaxis().set_ticks([45.5,46,46.5])
    #ax1.get_yaxis().set_ticks([0.05,0.15,0.25,0.35,0.45])
    
    
    ############################################################################
    
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
    
    ax10.hist(tab['BBT'], orientation='horizontal',color=cset[-1],bins=25)
    ax10.set_ylim(ax1.get_ylim())
    ax10.set_axis_off()
    
    ax4.set_xlabel(r'Log$_{10}$ (BH Mass $M_{\rm BH}$)')
    ax1.set_xlabel(r'Log$_{10} (L_{\rm UV} ({\rm erg/s}))$')
    ax7.set_xlabel(r'Log$_{10}$ ($\lambda_{\rm Edd}$)')
    
    axcb = fig.add_axes([0.33,0.15,0.33,0.03]) 
    clb = fig.colorbar(im, cax=axcb,orientation='horizontal') 
    clb.set_label('Number of Objects')
    
    s1 = spearmanr( tab[tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'], tab[tab['LOGEDD_RATIO'] > -2.5]['BBT'] )[0]
    s2 = spearmanr( tab[ tab['LOGBH'] > 6]['LOGBH'], tab[ tab['LOGBH'] > 6]['BBT'] )[0]
    s3 = spearmanr( tab['LUM_UV'], tab['BBT'] )[0]
    
    p1 = spearmanr( tab[tab['LOGEDD_RATIO'] > -2.5]['LOGEDD_RATIO'], tab[tab['LOGEDD_RATIO'] > -2.5]['BBT'] )[1]
    p2 = spearmanr( tab[ tab['LOGBH'] > 6]['LOGBH'], tab[ tab['LOGBH'] > 6]['BBT'] )[1]
    p3 = spearmanr( tab['LUM_UV'], tab['BBT'] )[1]
    
    ax1.text(45.5,600,r'$\rho =$ {0:.2f} ({1:.2f})'.format(s3,p3))
    ax4.text(8.6,600,r'$\rho =$ {0:.2f} ({1:.2f})'.format(s2,p2))
    ax7.text(-1.0,600,r'$\rho =$ {0:.2f} ({1:.2f})'.format(s1,p1))
    
    xdat = tab['LUM_UV']
    ydat = tab['RATIO_IR_UV']
    zdat = tab['BBT']
    
    rxy = spearmanr(xdat,ydat)[0]
    rxz = spearmanr(xdat,zdat)[0]
    ryz = spearmanr(ydat,zdat)[0]


    fig.savefig('/home/lc585/thesis/figures/chapter06/tbb_lowz_correlations.pdf')
    
    plt.show() 
 

    return None 