import yaml 
from qsosed.load import load
import cPickle as pickle 
from qsosed.loaddatraw import loaddatraw
import numpy as np 
import numpy.ma as ma 
from qsosed.sedmodel import model 
from scipy.interpolate import interp1d 
from matplotlib import cm
import brewer2mpl
from PlottingTools.truncate_colormap import truncate_colormap
import matplotlib.colors as colors
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import histogram2d
from PlottingTools.plot_setup import figsize, set_plot_properties
set_plot_properties() # change style 

def plot():

    # Load parameter file
    with open('./input.yml', 'r') as f:
        parfile = yaml.load(f)

    ebvmin = parfile['extlist']['ebvmin']
    ebvmax =  parfile['extlist']['ebvmax']

    zmin = 0.5
    zmax = 3.0

    plslp1 = parfile['quasar']['pl']['slp1']
    plslp2 = parfile['quasar']['pl']['slp2']
    plbrk = parfile['quasar']['pl']['brk']
    bbt = parfile['quasar']['bb']['t']
    bbflxnrm = parfile['quasar']['bb']['flxnrm']
    elscal = parfile['quasar']['el']['scal']
    scahal = parfile['quasar']['el']['scahal']
    galfra = parfile['gal']['fra']
    ebv = 0.0
    imod = parfile['quasar']['imod']

    # Load stuff
    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()

    if parfile['extlist']['flxcorr_file'] == 'None':
        flxcorr = np.array( [1.0] * len(wavlen) )
        with open(parfile['extlist']['outdir']+'/extlist_params.dat','a') as f:
            f.write('No flux correction applied \n')
    else:
        with open(parfile['extlist']['flxcorr_file'],'rb') as f:
            flxcorr = pickle.load(f)
        with open(parfile['extlist']['outdir']+'/extlist_params.dat','a') as f:
            value = parfile['extlist']['flxcorr_file']
            f.write('Flux correction to be applied using {} \n'.format(value))

    # Load data
    datmag,  sigma, datz, name, snr = loaddatraw(parfile['extlist']['datset'],
                                                 parfile['extlist']['cat'],
                                                 parfile['extlist']['balflg_on'],
                                                 parfile['extlist']['balflg'],
                                                 parfile['extlist']['imin'],
                                                 parfile['extlist']['imax'],
                                                 parfile['extlist']['snrmin'],
                                                 nrmflg=True)


    # Ignore objects with masked out magnitudes and uncertainties.
    ftrlst = fittingobj.get_ftrlst()
    requiredmags = (np.where(np.array(
    [parfile['extlist']['requiremag'][ftrlst[i]] for i in range(len(ftrlst))]
    ) == 'y' )[0])

    wmsk = ma.getmask(datz)
    for i in requiredmags:
        wmsk = ((wmsk) |
                (ma.getmask(datmag[:,i])) |
                (ma.getmask(sigma[:,i])) |
                (ma.getmask(snr[:,i])))

    datz = datz[~wmsk]
    name = name[~wmsk]
    datmag = datmag[~wmsk]
    sigma = sigma[~wmsk]
    snr = snr[~wmsk]


    # Redshift cut
    ind = np.where( (datz >= zmin) & (datz <= zmax) )[0]
    datz = datz[ind]
    name = name[ind]
    datmag = datmag[ind]
    sigma = sigma[ind]
    snr = snr[ind]

    print len(datmag)

    if parfile['extlist']['ebvcut'] is True:

        colmin, colmax = [], []

        zs = np.arange(zmin,zmax+0.025,0.025)
        for z in zs:
            print z

            magtmp, wavlentmp, fluxtmp  = model(plslp1,
                                                plslp2,
                                                plbrk,
                                                bbt,
                                                bbflxnrm,
                                                elscal,
                                                scahal,
                                                galfra,
                                                ebvmin,
                                                imod,
                                                z,
                                                fittingobj,
                                                flxcorr,
                                                parfile=parfile)

            colmin.append(magtmp[3] - magtmp[8])

            magtmp, wavlentmp, fluxtmp  = model(plslp1,
                                                plslp2,
                                                plbrk,
                                                bbt,
                                                bbflxnrm,
                                                elscal,
                                                scahal,
                                                galfra,
                                                ebvmax,
                                                imod,
                                                z,
                                                fittingobj,
                                                flxcorr,
                                                parfile=parfile)

            colmax.append(magtmp[3] - magtmp[8])

        fmin = interp1d(zs,colmin,bounds_error=False,fill_value=0.0)
        fmax = interp1d(zs,colmax,bounds_error=False,fill_value=0.0)

        mycm = cm.get_cmap('YlOrRd_r')
        mycm.set_under('w')
        cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors
        mycm = truncate_colormap(mycm, 0.0, 0.8)

        fig = plt.figure(figsize=(figsize(0.8)))
        ax = fig.add_subplot(1,1,1)
        plt.tight_layout()

        #histogram definition
        xyrange = [[0.,3.0],[-2,3]] # data range
        bins = [100,60] # number of bins
        thresh = 7  #density threshold

        #data definition
        xdat, ydat = datz, datmag[:,3] - datmag[:,8]

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

        im = ax.imshow(np.flipud(hh.T),cmap=mycm,extent=np.array(xyrange).flatten(), interpolation='none',aspect='auto', vmin=thresh, vmax=45)
        clb = fig.colorbar(im)
        clb.set_label('Number of Objects',fontsize=12)
        clb.ax.tick_params(labelsize=10)
        ax.scatter(xdat1, ydat1,color=cset[-1],s=2)

        plt.ylim(-2,3)

        col = []
        for z in zs:
            magtmp, wavlentmp, fluxtmp  = model(plslp1,
                                                plslp2,
                                                plbrk,
                                                bbt,
                                                bbflxnrm,
                                                elscal,
                                                scahal,
                                                galfra,
                                                0.075,
                                                imod,
                                                z,
                                                fittingobj,
                                                flxcorr,
                                                parfile=parfile)

            col.append(magtmp[3] - magtmp[8])

        plt.plot(zs,col,label='E(B-V) = 0.075',color='black',linewidth=1.0)
        upper_bound_1 = col[np.argmin(np.abs(zs-1.0)):np.argmin(np.abs(zs-1.5))]
        upper_bound_2 = col[np.argmin(np.abs(zs-2.0)):np.argmin(np.abs(zs-2.7))]


        col = []
        for z in zs:
            magtmp, wavlentmp, fluxtmp  = model(plslp1,
                                                plslp2,                                                                   plbrk,
                                                bbt,
                                                bbflxnrm,
                                                elscal,
                                                scahal,
                                                galfra,
                                                -0.075,
                                                imod,
                                                z,
                                                fittingobj,
                                                flxcorr,
                                                parfile=parfile)
            col.append(magtmp[3] - magtmp[8])

        plt.plot(zs,col,label='E(B-V) = -0.075',color='black',linewidth=1.0)
        lower_bound_1 = col[np.argmin(np.abs(zs-1.0)):np.argmin(np.abs(zs-1.5))]
        lower_bound_2 = col[np.argmin(np.abs(zs-2.0)):np.argmin(np.abs(zs-2.7))]


        plt.fill_between(zs[np.argmin(np.abs(zs-1.0)):np.argmin(np.abs(zs-1.5))],
                         lower_bound_1,
                         upper_bound_1,
                         facecolor='None',
                         edgecolor='black',
                         linewidth=3.0)


        plt.fill_between(zs[np.argmin(np.abs(zs-2.0)):np.argmin(np.abs(zs-2.7))],
                         lower_bound_2,
                         upper_bound_2,
                         facecolor='None',
                         edgecolor='black',
                         linewidth=3.0)

        col = []
        for z in zs:
            magtmp, wavlentmp, fluxtmp  = model(plslp1,
                                                plslp2,                                                                   plbrk,
                                                bbt,
                                                bbflxnrm,
                                                elscal,
                                                scahal,
                                                galfra,
                                                0.0,
                                                imod,
                                                z,
                                                fittingobj,
                                                flxcorr,
                                                parfile=parfile)
            col.append(magtmp[3] - magtmp[8])

        plt.plot(zs,col,label='E(B-V) = 0.0',color='black',linewidth=1.0)


        plt.xlim(0.5,3.5)

        plt.text(1.25,
                 -1,
                 r'Low-$z$',
                 fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center')

        plt.text(2.35,
                 -1,
                 r'High-$z$',
                 fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center')


        plt.text(3.33821,
                 1.0695,
                 'E(B-V)=',
                 fontsize=12,
                 horizontalalignment='right',
                 verticalalignment='center',
                 color='black')

        plt.text(3.09608,
                 0.629896,
                 '0.075',
                 fontsize=12,
                 horizontalalignment='left',
                 verticalalignment='center',
                 color='black')


        plt.text(3.09608,
                 -0.469225,
                 '-0.075',
                 fontsize=12,
                 horizontalalignment='left',
                 verticalalignment='center',
                 color='black')


        plt.text(3.09608,
                 0.1,
                 '0.0',
                 fontsize=12,
                 horizontalalignment='left',
                 verticalalignment='center',
                 color='black')

        plt.xlabel(r'Redshift $z$')
        plt.ylabel(r'$i$-$K$')
        plt.tight_layout()

        fig.savefig('/home/lc585/thesis/figures/chapter06/ik_versus_z_low_ext.pdf')
        plt.show()

        return None
