import yaml
import numpy as np
from qsosed.sedmodel import model
from qsosed.load import load
import cPickle as pickle
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy import histogram2d
import brewer2mpl
from matplotlib import cm
from PlottingTools.truncate_colormap import truncate_colormap
# import matplotlib.colors as colors
from qsosed.bb import bb
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties


def plot():

    set_plot_properties() # change style 

    with open('input.yml', 'r') as f:
        parfile = yaml.load(f)

    fittingobj = load(parfile)

    flxcorr_file = '/data/lc585/QSOSED/Results/140811/allsample_2/fluxcorr.array'

    with open(flxcorr_file,'rb') as f:
        flxcorr = pickle.load(f)

    plslp1 = parfile['quasar']['pl']['slp1']
    plslp2 = parfile['quasar']['pl']['slp2']
    plbrk = parfile['quasar']['pl']['brk']
    bbt = parfile['quasar']['bb']['t']
    bbflxnrm = parfile['quasar']['bb']['flxnrm']
    elscal = parfile['quasar']['el']['scal']
    scahal = parfile['quasar']['el']['scahal']
    galfra = parfile['gal']['fra']
    ebv = parfile['ext']['EBV']
    imod = parfile['quasar']['imod']

    tab = Table.read('/data/lc585/QSOSED/Results/150118/sample6/tabout.fits')

    zs = np.linspace(0.25,3.0,100)
    wavlen = fittingobj.get_wavlen()

    uvintmin = np.argmin(np.abs(wavlen - 2000.))
    uvintmax = np.argmin(np.abs(wavlen - 9000.))
    irintmin = np.argmin(np.abs(wavlen - 10000.))
    irintmax = np.argmin(np.abs(wavlen - 30000.))


    def ratiogoal(rg):

        ratio = 0.0
        bbflxnrm = 0.0
        z = 2.0

        while ratio < rg:

            bbflux = bb(wavlen,bbt,bbflxnrm,20000.0)
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

            ratio = np.sum(bbflux[irintmin:irintmax]) / np.sum(fluxtmp[uvintmin:uvintmax])
            bbflxnrm = bbflxnrm + 0.001

        cols = []

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

            cols.append(magtmp[9] - magtmp[10])

        return cols



    fig, ax = plt.subplots(figsize=figsize(1.0, vscale=0.7))

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    mycm = truncate_colormap(mycm, 0.0, 0.8)
    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors

    #histogram definition
    xyrange = [[0,3],[0,1.2]] # data range
    bins = [50,50] # number of bins
    thresh = 4  #density threshold

    #data definition
    w1mag = tab['W1MPRO_ALLWISE'] + 2.699
    w2mag = tab['W2MPRO_ALLWISE'] + 3.339
    xdat, ydat = tab['Z_HEWETT'], w1mag - w2mag

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
                  vmin=thresh, 
                  vmax=45)

    cb = plt.colorbar(im)
    cb.set_label('Number of Objects')

    ax.scatter(xdat1, ydat1,color=cset[-1],s=5)

    cset = brewer2mpl.get_map('YlGnBu', 'sequential', 8).mpl_colors

    ax.plot(zs,ratiogoal(0.5),c=cset[7],linewidth=2.0,label=r'0.5')
    ax.plot(zs,ratiogoal(0.4),c=cset[6],linewidth=2.0,label=r'0.4')
    ax.plot(zs,ratiogoal(0.3),c=cset[5],linewidth=2.0,label=r'0.3')
    ax.plot(zs,ratiogoal(0.2),c=cset[4],linewidth=2.0,label=r'0.2')
    ax.plot(zs,ratiogoal(0.1),c=cset[3],linewidth=2.0,label=r'0.1')
    ax.plot(zs,ratiogoal(0.0),c=cset[2],linewidth=2.0,label=r'0.0')

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$W1-W2$')

    ax.set_xlim(-0.25, 4.3)
    ax.set_ylim(-0.3,1.2)

    plt.legend(frameon=False,loc=(0.75,0.15))

    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/w1w2_versus_redshift_ratio.pdf')

    plt.show()

    return None

