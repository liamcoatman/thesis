from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import sys
import cPickle as pickle
import yaml
import brewer2mpl
from matplotlib import cm
from scipy import histogram2d
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from qsosed.load import load
from qsosed.sedmodel import model 
from PlottingTools.plot_setup import figsize, set_plot_properties

set_plot_properties() # change style 

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot():

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    mycm = truncate_colormap(mycm, 0.0, 0.8)
    
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors
    
    tab = Table.read('/data/lc585/QSOSED/Results/141031/sample1.fits')
    
    fig = plt.figure(figsize=figsize(0.7, vscale=1.5))
    
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    fig.subplots_adjust(wspace=0.0)
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(top=0.99, bottom=0.2)

    #histogram definition
    xyrange = [[0.5,1.5],[-0.4,1.2]] # data range
    bins = [45,45] # number of bins
    thresh = 4  #density threshold
    
    #data definition
    w1mag = tab['W1MPRO_ALLWISE'] + 2.699
    w2mag = tab['W2MPRO_ALLWISE'] + 3.339
    z = tab['Z_HEWETT']
    xdat, ydat = z, w1mag - w2mag
    
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
    
    ax1.set_ylabel(r'$W1-W2$',fontsize=14)
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax1.set_ylim(-0.4,1.2)
    ax1.set_xlim(0.25,1.7)
    
    im = ax2.imshow(np.flipud(hh.T),cmap=mycm,extent=np.array(xyrange).flatten(), interpolation='none',aspect='auto', vmin=thresh, vmax=45)
    ax2.scatter(xdat1, ydat1,color=cset[-1])
    
    axcb = fig.add_axes([0.13,0.1,0.75,0.02])
    clb = fig.colorbar(im, cax=axcb,orientation='horizontal')
    clb.set_label('Number of Objects',fontsize=12)
    clb.ax.tick_params(labelsize=10)
    
    ax2.set_xlabel(r'Redshift $z$',fontsize=14)
    ax2.set_ylabel(r'$W1-W2$',fontsize=14)
    
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(ax1.get_xlim())
    tabtmp = tab
    tabtmp.sort('Z')
    
    
    #plt.tick_params(axis='both',which='major',labelsize=12)
    
    ax1.get_xaxis().set_ticks([])
    ax1.set_yticklabels(['',0.0,0.2,0.4,0.6,0.8,1.0,1.2])
    ax2.set_yticklabels([-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0])
    #plt.scatter(z,w1mag-w2mag,c='grey',alpha=0.3)
    
    #xdat = z
    #ydat = w1mag - w2mag
    #zdat = tab['LOGLBOL']
    #
    #plt.hexbin(xdat,ydat,C=tab['LUM_UV'],gridsize=10,cmap=mycm)
    #cb = plt.colorbar()
    #cb.set_label('UV Luminosity')
    #plt.ylim(-0.2,1.2)
    #plt.xlim(0.4,1.6)
    ##plt.savefig('/data/lc585/QSOSED/Results/141030/figure1.jpg')
    #
    #xyrange = [[0.5,1.5],[0,1.2]] # data range
    #bins = [10,6] # number of bins
    #thresh = 20
    #hh, locx, locy = histogram2d(xdat,
    #                             ydat,
    #                             range=xyrange,
    #                             bins=bins)
    #
    #
    #print locy
    #
    #posx = np.digitize(xdat, locx)
    #posy = np.digitize(ydat, locy)
    #
    #grid = []
    #for i in range(10):
    #    row = []
    #    for j in range(6):
    #        row.append(np.median(zdat[(posx == i+1) & (posy ==j+1)]))
    #    grid.append(row)
    
    #ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    #hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    #xdat1 = xdat[ind][hhsub < thresh] # low density points
    #ydat1 = ydat[ind][hhsub < thresh]
    #hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
    
    #ax.scatter(z[ind],w1mag[ind]-w2mag[ind])
    
    #im = ax.imshow(np.flipud(hh.T),
    #                cmap=mycm,
    #                extent=np.array(xyrange).flatten(),
    #                interpolation='none',
    #                aspect='auto',vmin=10)
    #ax.scatter(xdat1,ydat1)
    
    # loop over bins
    #for i in range(len(locx)):
    #    for j in range(len(locy)):
    #        print np.median(tab['LUM_UV'][ (posx == i) & (posy == j)])
    #ax.hist2d(z,
    #          w1mag-w2mag,
    #          range=xyrange,
    #          bins=bins,
    #          cmap=mycm,
    #          vmin=10.)
    
    
    with open('/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit/input.yml', 'r') as f:
        parfile = yaml.load(f)
    
    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()
    
    with open('/data/lc585/QSOSED/Results/140811/allsample_2/fluxcorr.array','rb') as f:
        flxcorr = pickle.load(f)
    
    plslp1 = 0.46
    plslp2 = 0.03
    plbrk = 2822.
    bbt = 1216.
    bbflxnrm = 0.24
    elscal = 0.71
    scahal = 0.86
    galfra = 0.31
    ebv = 0.0
    imod = 18.0
    
    zs = np.arange(0.5,1.525,0.025)
    
    bbt = 1200.
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    print w1w2_model[0]
    ax1.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbt = 1100.
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    print w1w2_model[0]
    ax1.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbt = 1300.
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    print w1w2_model[0]
    ax1.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbt = 1400.
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    print w1w2_model[0]
    ax1.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbt = 1500.
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    print w1w2_model[0]
    ax1.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbt = 1000.
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    ax1.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    ax1.text(0.33,0.67,'1000K',fontsize=12,color='black')
    ax1.text(0.33,0.52,'1100K',fontsize=12,color='black')
    ax1.text(0.33,0.40,'1200K',fontsize=12,color='black')
    ax1.text(0.33,0.28,'1300K',fontsize=12,color='black')
    ax1.text(0.33,0.20,'1400K',fontsize=12,color='black')
    ax1.text(0.33,0.10,'1500K',fontsize=12)
    
    
    #plt.savefig('/home/lc585/Dropbox/IoA/HotDustPaper/w1w2_temp.pdf')
    
    bbt = 1216.
    bbflxnrm = 0.0
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    ax2.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbflxnrm = 0.1
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    ax2.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbflxnrm = 0.2
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    ax2.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbflxnrm = 0.3
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    ax2.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    bbflxnrm = 0.4
    w1w2_model = []
    for z in zs:
        magtmp, wavlentmp, fluxtmp = model(plslp1,
                                           plslp2,
                                           plbrk,
                                           bbt,
                                           bbflxnrm,
                                           elscal,
                                           scahal,
                                           galfra,
                                           ebv,
                                           imod,
                                           z,
                                           fittingobj,
                                           flxcorr,
                                           parfile)
        w1w2_model.append(magtmp[9] - magtmp[10])
    
    ax2.plot(zs,w1w2_model,linewidth=2.0,color='black')
    
    #plt.xlim(0.2,1.8)
    
    ax2.text(1.55,-0.1,'0.17',fontsize=12,color='black')
    ax2.text(1.55,0.38,'0.28',fontsize=12,color='black')
    ax2.text(1.55,0.60,'0.40',fontsize=12,color='black')
    ax2.text(1.55,0.75,'0.52',fontsize=12,color='black')
    ax2.text(1.55,0.90,'0.64',fontsize=12,color='black')
    
    fig.savefig('/home/lc585/thesis/figures/chapter06/w1w2_versus_redshift.pdf')
    #plt.xlabel(r'$z$',fontsize=12)
    #plt.ylabel('$W1$-$W2$',fontsize=12)
    #plt.tight_layout()
    #plt.tick_params(axis='both',which='major',labelsize=10)
    
    #sns.set_style('ticks')
    #xdat = np.arange(0.1,1.2,0.2)
    #plt.savefig('/data/lc585/QSOSED/Results/141101/figure1.jpg')
    #fig, ax = plt.subplots()
    #for row in grid:
    #    ax.plot(xdat,row)
    #ax.set_xlabel('W1-W2')
    #ax.set_ylabel('LUM BOL')
    #plt.savefig('/data/lc585/QSOSED/Results/141101/figure2.jpg')
    #plt.show()
    #plt.savefig('/home/lc585/Dropbox/IoA/HotDustPaper/w1w2_bbnorm.pdf')

    plt.show() 
    

    return None 





