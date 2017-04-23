# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:44:37 2014

@author: lc585

Make plots of fit to whole sample
"""
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
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
from PlottingTools.truncate_colormap import truncate_colormap
import palettable 
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
from qsosed.get_data import get_data


set_plot_properties() # change style 

def plot():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_8.mpl_colors

    with open('/home/lc585/qsosed/input.yml', 'r') as f:
        parfile = yaml.load(f)

    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()
    lin = fittingobj.get_lin()
    galspc = fittingobj.get_galspc()
    ext = fittingobj.get_ext()
    galcnt = fittingobj.get_galcnt()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    ztran = fittingobj.get_ztran()
    lyatmp = fittingobj.get_lyatmp()
    lybtmp = fittingobj.get_lybtmp()
    lyctmp = fittingobj.get_lyctmp()
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    qsomag = fittingobj.get_qsomag()
    flxcorr = fittingobj.get_flxcorr()
    cosmo = fittingobj.get_cosmo() 

    params = Parameters()
    params.add('plslp1', value = -0.478)
    params.add('plslp2', value = -0.199)
    params.add('plbrk', value = 2.40250)
    params.add('bbt', value = 1.30626)
    params.add('bbflxnrm', value = 2.673)
    params.add('elscal', value = 1.240)
    params.add('scahal',value = 0.713)
    params.add('galfra',value = 0.0)
    params.add('bcnrm',value = 0.135)
    params.add('ebv',value = 0.0)
    params.add('imod',value = 18.0)
    
    # Load median magnitudes 
    with open('/home/lc585/qsosed/sdss_ukidss_wise_medmag_ext.dat') as f:
        datz = np.loadtxt(f, usecols=(0,))

    datz = datz[:-5]

    # Load filters
    ftrlst = fittingobj.get_ftrlst()[2:-2] 
    lameff = fittingobj.get_lameff()[2:-2]
    bp = fittingobj.get_bp()[2:-2] # these are in ab and data is in vega 
    dlam = fittingobj.get_bp()[2:-2]
    zromag = fittingobj.get_zromag()[2:-2]

    with open('ftgd_dr7.dat') as f:
        ftgd = np.loadtxt(f, skiprows=1, usecols=(1,2,3,4,5,6,7,8,9))

    modarr = residual(params,
                      parfile,
                      wavlen,
                      datz,
                      lin,
                      bp,
                      dlam,
                      zromag,
                      galspc,
                      ext,
                      galcnt,
                      ignmin,
                      ignmax,
                      ztran,
                      lyatmp,
                      lybtmp,
                      lyctmp,
                      ftrlst,
                      whmin,
                      whmax,
                      cosmo,
                      flxcorr,
                      qsomag,
                      ftgd)
    
    fname = '/home/lc585/qsosed/sdss_ukidss_wise_medmag_ext.dat'
    datarr = np.genfromtxt(fname, usecols=(5,7,9,11,13,15,17,19,21)) 
    datarr[datarr < 0.0] = np.nan 

    datarr = datarr[:-5, :]




    # remove less than lyman break

    col1 = np.arange(8)
    col2 = col1 + 1 
    
    col_label = ['$r$ - $i$',
                 '$i$ - $z$',
                 '$z$ - $Y$',
                 '$Y$ - $J$',
                 '$J$ - $H$',
                 '$H$ - $K$',
                 '$K$ - $W1$',
                 '$W1$ - $W2$']

    df = get_data() 
    df = df[(df.z_HW > 1) & (df.z_HW < 3)]

    colstr1 = ['rVEGA',
               'iVEGA',
               'zVEGA',
               'YVEGA',
               'JVEGA',
               'HVEGA',
               'KVEGA',
               'W1VEGA']
    
    colstr2 = ['iVEGA',
               'zVEGA',
               'YVEGA',
               'JVEGA',
               'HVEGA',
               'KVEGA',
               'W1VEGA',
               'W2VEGA']

    ylims = [[0, 0.6], 
             [-0.1, 0.5], 
             [-0.1, 0.5],
             [-0.1, 0.5],
             [0.2, 0.9],
             [0.2, 0.9],
             [0.5, 1.6],
             [0.8, 1.5]]


    fig, axs = plt.subplots(4, 2, figsize=figsize(1, vscale=2), sharex=True) 
    
    for i, ax in enumerate(axs.flatten()):

        #data definition
        ydat = datarr[:, col1[i]] - datarr[:, col2[i]]

        ax.scatter(datz, 
                   ydat, 
                   color='black', 
                   s=5,
                   label='Data')

        ax.plot(datz,
                modarr[:,col1[i]] - modarr[:, col2[i]],
                color=cs[1], 
                label='Model')

        # ax.scatter(df.z_HW, df[colstr1[i]] - df[colstr2[i]], s=1, alpha=0.1) 


        ax.set_title(col_label[i], size=10)
    
        ax.set_ylim(ylims[i])
        ax.set_xlim(0.75, 3.25)

    

    axs[0, 0].legend(bbox_to_anchor=(0.7, 0.99), 
                     bbox_transform=plt.gcf().transFigure,
                     fancybox=True, 
                     shadow=True,
                     scatterpoints=1,
                     ncol=2) 



    axs[3, 0].set_xlabel(r'Redshift $z$')
    axs[3, 1].set_xlabel(r'Redshift $z$')

    fig.tight_layout()

    fig.subplots_adjust(wspace=0.2, hspace=0.15, top=0.93)

    fig.savefig('/home/lc585/thesis/figures/chapter05/sed_color_plot.pdf')


    plt.show() 

    return None


if __name__ == '__main__':
    plot() 