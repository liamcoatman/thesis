import cosmolopy.distance as cd
from qsosed.residual import residual
import numpy.ma as ma
import numpy as np
import qsosed.readdat as rd
import matplotlib.pyplot as plt
import os
import scipy
from matplotlib import cm
import matplotlib.colors as colors
import brewer2mpl
from qsosed.load import load
import yaml 
from lmfit import Parameters
import cPickle as pickle 
from qsosed.loaddat import loaddat
from PlottingTools.plot_setup import figsize, set_plot_properties
from qsosed.sedmodel import model 
from qsosed.loaddatraw import loaddatraw

set_plot_properties() # change style 

def plot():
        
    plslp1 = 0.71
    plslp2 = 0.05
    plbrk = 2679.92
    bbt = 1158.99
    bbflxnrm = 0.25
    galfra = 0.89
    elscal = 0.72
    imod = 18.0 
    scahal = 1.0
    
    with open('input.yml', 'r') as f:
        parfile = yaml.load(f)
    
    # Load stuff
    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()
    
    flxcorr = np.array([1.0] * len(wavlen))

    ik0, ik1, ik2 = [], [], []

    zs = np.arange(2,4.01,0.01)
    
    for z in zs:

        print z 

        magtmp, wavlentmp1, fluxtmp1 = model(plslp1,
                                             plslp2,
                                             plbrk,
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
                                             parfile)

        ik0.append( magtmp[3] - magtmp[8] )

        magtmp, wavlentmp1, fluxtmp1 = model(plslp1,
                                             plslp2,
                                             plbrk,
                                             bbt,
                                             bbflxnrm,
                                             elscal,
                                             scahal,
                                             galfra,
                                             0.1,
                                             imod,
                                             z,
                                             fittingobj,
                                             flxcorr,
                                             parfile)

        ik1.append( magtmp[3] - magtmp[8] )

        magtmp, wavlentmp1, fluxtmp1 = model(plslp1,
                                             plslp2,
                                             plbrk,
                                             bbt,
                                             bbflxnrm,
                                             elscal,
                                             scahal,
                                             galfra,
                                             0.2,
                                             imod,
                                             z,
                                             fittingobj,
                                             flxcorr,
                                             parfile)

        ik2.append( magtmp[3] - magtmp[8] )                           
 

    datmag, sigma, datz, name, snr = loaddatraw('DR10',
                                                '/data/lc585/SDSS/DR10QSO_AllWISE_matched.v2.fits',
                                                True,
                                                0,
                                                23.0,
                                                15.0,
                                                False,
                                                0.1)

    

    fig = plt.figure(figsize=figsize(0.7))
    ax = fig.add_subplot(111)
    ax.plot(zs,ik0,color='black',linewidth=2, zorder=1)
    ax.plot(zs,ik1,color='black',linewidth=2, zorder=1)
    ax.plot(zs,ik2,color='black',linewidth=2, zorder=1)

    ax.plot(datz,
            datmag[:,3] - datmag[:,8],
            marker='o',
            markersize=2,
            alpha=0.5,
            markeredgecolor='none',
            markerfacecolor='gray',
            linestyle='',
            zorder=0)
    
    textprops = dict(fontsize=10,ha='left',va='center', zorder=1, color='red')

    ax.text(3.5,0.646,'E(B-V) = 0.0', **textprops)
    ax.text(3.5,1.455,'E(B-V) = 0.1', **textprops)
    ax.text(3.5,2.285,'E(B-V) = 0.2', **textprops)

    ax.set_xlim(2,4)
    ax.set_ylim(-2,4)

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$i - K$')
    
    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/ik_versus_z_high_ext.pdf')
    plt.show() 

    return None