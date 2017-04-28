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
import palettable
import matplotlib.pyplot as plt 
from scipy import histogram2d
from qsosed.get_data import get_data, extcut
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
set_plot_properties() # change style 

def plot():

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    # Load parameter file
    with open('/home/lc585/qsosed/input.yml', 'r') as f:
        parfile = yaml.load(f)

    ebvmin = -0.075
    ebvmax = 0.075

    zmin = 1 
    zmax = 3.0

    # Load stuff
    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()

    # Load data
    df = get_data() 


    df = df[(df.z_HW >= zmin) & (df.z_HW <= zmax)]

    colmin, colmax = [], []

    zs = np.arange(zmin, zmax+0.025, 0.025)

    mycm = cm.get_cmap('Blues_r')
    mycm.set_under('w')
    cset = brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colors
    mycm = truncate_colormap(mycm, 0.0, 0.8)

    fig = plt.figure(figsize=(figsize(1, vscale=0.8)))
    ax = fig.add_subplot(1,1,1)
    plt.tight_layout()

    #histogram definition
    xyrange = [[0.,3.0],[0,3]] # data range
    bins = [100,60] # number of bins
    thresh = 7  #density threshold

    #data definition

    xdat, ydat = df.z_HW, df.iVEGA - df.KVEGA

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

    clb = fig.colorbar(im)
    clb.set_label('Number of Objects')
    ax.scatter(xdat1, ydat1,color=cset[-1],s=2)

    plt.ylim(0, 3.5)

    col = []
    for z in zs:

        parfile['ext']['EBV'] = ebvmax  

        magtmp, wavlentmp, fluxtmp  = model(redshift=z,
                                            parfile=parfile) 

        col.append(magtmp[3] - magtmp[8])

    plt.plot(zs,col,label='E(B-V) = 0.075',color=cs[0],linewidth=1.0)
    upper_bound_1 = col[np.argmin(np.abs(zs-1.0)):np.argmin(np.abs(zs-1.5))]
    upper_bound_2 = col[np.argmin(np.abs(zs-2.0)):np.argmin(np.abs(zs-2.7))]


    col = []
    for z in zs:

        parfile['ext']['EBV'] = ebvmin   

        magtmp, wavlentmp, fluxtmp  = model(redshift=z,
                                        parfile=parfile) 

        col.append(magtmp[3] - magtmp[8])

    plt.plot(zs,col,label='E(B-V) = -0.075',color=cs[0],linewidth=1.0)
    lower_bound_1 = col[np.argmin(np.abs(zs-1.0)):np.argmin(np.abs(zs-1.5))]
    lower_bound_2 = col[np.argmin(np.abs(zs-2.0)):np.argmin(np.abs(zs-2.7))]


    plt.fill_between(zs[np.argmin(np.abs(zs-2.0)):np.argmin(np.abs(zs-2.7))],
                     lower_bound_2,
                     upper_bound_2,
                     facecolor='None',
                     edgecolor=cs[0],
                     linewidth=3.0)

    col = []
    for z in zs:
        
        parfile['ext']['EBV'] = 0.0   

        magtmp, wavlentmp, fluxtmp  = model(redshift=z,
                                            parfile=parfile) 


        col.append(magtmp[3] - magtmp[8])

    plt.plot(zs,col,label='E(B-V) = 0.0',color=cs[0],linewidth=1.0)


    plt.xlim(0.75,3.5)


    plt.text(3.33821,
             2.5,
             'E(B-V)=',
             horizontalalignment='right',
             verticalalignment='center',
             color='black')

    plt.text(3.09608,
             2.2,
             '0.075',
             horizontalalignment='left',
             verticalalignment='center',
             color='black')


    plt.text(3.09608,
             1.2,
             '-0.075',
             horizontalalignment='left',
             verticalalignment='center',
             color='black')


    plt.text(3.09608,
             1.7,
             '0.0',
             horizontalalignment='left',
             verticalalignment='center',
             color='black')

    plt.xlabel(r'Redshift $z$')
    plt.ylabel(r'$i$-$K$')
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter05/ik_versus_z_low_ext.pdf')
    plt.show()

    return None
