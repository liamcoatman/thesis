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
from qsosed.get_data import get_data 
import astropy.units as u 
from qsosed.qsrmod import wav2num
from qsosed.pl import pl

def plot():

    set_plot_properties() # change style 

    with open('/home/lc585/qsosed/input.yml', 'r') as f:
        parfile = yaml.load(f)

    fittingobj = load(parfile)

    df = get_data() 

    zs = np.linspace(1, 3.0, 100)
    wavlen = fittingobj.get_wavlen()

    uvintmin = np.argmin(np.abs(wavlen - 2000.))
    uvintmax = np.argmin(np.abs(wavlen - 9000.))
    irintmin = np.argmin(np.abs(wavlen - 10000.))
    irintmax = np.argmin(np.abs(wavlen - 30000.))


    def ratiogoal(rg):

        ratio = 0.0
        bbflxnrm = 0.0
        z = 2.0
        bbt = 1306
        plslp1 = -0.478 
        plslp2 = -0.199
        plbrk = 2402.0
        flxnrm = 1.0
        wavnrm = 5500.0
        bbwavnrm = 20000.0

        # while ratio < rg:

            
        #     flux = np.zeros(len(wavlen), dtype=np.float)  

        #     # Define normalisation constant to ensure continuity at wavbrk 
        #     const2 = flxnrm / (wavnrm**(-plslp2)) 
        #     const1 = const2 * ((plbrk**(-plslp2)) / (plbrk**(-plslp1)))
        
        #     wavnumbrk = wav2num(wavlen, plbrk) 
        
        #     flux[:wavnumbrk] = flux[:wavnumbrk] + pl(wavlen[:wavnumbrk], plslp1, const1)
        #     flux[wavnumbrk:] = flux[wavnumbrk:] + pl(wavlen[wavnumbrk:], plslp2, const2)
        
        
        #     # Hot blackbody ---------------------------------------------------
       
        #     bbflux = bb(wavlen*u.AA,
        #                 bbt*u.K,
        #                 bbflxnrm,
        #                 bbwavnrm*u.AA,
        #                 units='freq')

        #     flux = flux*(u.erg / u.s / u.cm**2 / u.Hz)
        #     flux = flux.to(u.erg / u.s / u.cm**2 / u.AA, 
        #                    equivalencies=u.spectral_density(wavlen * u.AA))

        #     bbflux = bbflux*(u.erg / u.s / u.cm**2 / u.Hz)
        #     bbflux = bbflux.to(u.erg / u.s / u.cm**2 / u.AA, 
        #                        equivalencies=u.spectral_density(wavlen * u.AA))


        #     ratio = np.sum(bbflux[irintmin:irintmax]) / np.sum(flux[uvintmin:uvintmax])
        #     bbflxnrm = bbflxnrm + 0.01

        # print ratio, bbflxnrm
        # parfile['quasar']['bb']['flxnrm'] = bbflxnrm
        
        print rg 
        if rg == 0.5: parfile['quasar']['bb']['flxnrm'] = 4.47
        elif rg == 0.4: parfile['quasar']['bb']['flxnrm'] = 3.58
        elif rg == 0.3: parfile['quasar']['bb']['flxnrm'] = 2.69
        elif rg == 0.2: parfile['quasar']['bb']['flxnrm'] = 1.8
        elif rg == 0.1: parfile['quasar']['bb']['flxnrm'] = 0.91
        elif rg == 0.0: parfile['quasar']['bb']['flxnrm'] = 0.0

        cols = []


        for z in zs:

            magtmp, wavlentmp, fluxtmp = model(redshift=z,
                                               parfile=parfile)

            cols.append(magtmp[9] - magtmp[10])

        return cols



    fig, ax = plt.subplots(figsize=figsize(1.0, vscale=0.7))

    mycm = cm.get_cmap('Blues_r')
    mycm.set_under('w')
    mycm = truncate_colormap(mycm, 0.0, 0.8)
    cset = brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colors

    #histogram definition
    xyrange = [[1,3],[0,2]] # data range
    bins = [50,50] # number of bins
    thresh = 4  #density threshold

    #data definition
    xdat, ydat = df.z_HW, df.W1VEGA - df.W2VEGA

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

    ax.scatter(xdat1, ydat1, color=cset[-1],s=3)

    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 8).mpl_colors

    ax.plot(zs,ratiogoal(0.5),c=cset[7],linewidth=2.0,label=r'0.5')
    ax.plot(zs,ratiogoal(0.4),c=cset[6],linewidth=2.0,label=r'0.4')
    ax.plot(zs,ratiogoal(0.3),c=cset[5],linewidth=2.0,label=r'0.3')
    ax.plot(zs,ratiogoal(0.2),c=cset[4],linewidth=2.0,label=r'0.2')
    ax.plot(zs,ratiogoal(0.1),c=cset[3],linewidth=2.0,label=r'0.1')
    ax.plot(zs,ratiogoal(0.0),c=cset[2],linewidth=2.0,label=r'0.0')

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$W1-W2$')

    ax.set_xlim(1, 4)
    ax.set_ylim(0.2,2)

    ax.text(0.72, 0.7, r'${\mathrm R_{\mathrm NIR/UV}}$', transform = ax.transAxes)
    plt.legend(frameon=False,loc=(0.7,0.15))

    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter05/w1w2_versus_redshift_ratio.pdf')

    plt.show()

    return None

