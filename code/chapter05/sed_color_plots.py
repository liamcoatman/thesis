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

set_plot_properties() # change style 

def plot():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_8.mpl_colors

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
    params.add('plslp1', value = parfile['quasar']['pl']['slp1'])
    params.add('plslp2', value = parfile['quasar']['pl']['slp2'])
    params.add('plbrk', value = parfile['quasar']['pl']['brk'])
    params.add('bbt', value = parfile['quasar']['bb']['t'])
    params.add('bbflxnrm', value = parfile['quasar']['bb']['flxnrm'])
    params.add('elscal', value = parfile['quasar']['el']['scal'])
    params.add('galfra',value = parfile['gal']['fra'])
    params.add('ebv',value = parfile['ext']['EBV'])
    params.add('imod',value = parfile['quasar']['imod'])
    params.add('scahal',value=parfile['quasar']['el']['scahal'])
    
    # Load median magnitudes 
    with open('/home/lc585/qsosed/sdss_ukidss_wise_medmag_ext.dat') as f:
        datz = np.loadtxt(f, usecols=(0,))

    # Load filters
    ftrlst = fittingobj.get_ftrlst()[:-2] 
    lameff = fittingobj.get_lameff()[:-2]
    bp = fittingobj.get_bp()[:-2] # these are in ab and data is in vega 
    dlam = fittingobj.get_bp()[:-2]
    zromag = fittingobj.get_zromag()[:-2]
  
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
                      qsomag)
    
    fname = '/home/lc585/qsosed/sdss_ukidss_wise_medmag_ext.dat'
    datarr = np.genfromtxt(fname, usecols=(1,3,5,7,9,11,13,15,17,19,21)) 
    datarr[datarr < 0.0] = np.nan 

    col1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8]
    col2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10]
    
    col_label = ['$u$ - $g$',
                 '$g$ - $r$',
                 '$r$ - $i$',
                 '$i$ - $z$',
                 '$z$ - $y$',
                 '$Y$ - $J$',
                 '$J$ - $H$',
                 '$H$ - $K$',
                 '$K$ - $W1$',
                 '$W1$ - $W2$',
                 '$H - W1$',
                 '$K - W2$']



    fig1, axs1 = plt.subplots(3, 2, figsize=figsize(1.2, vscale=1), sharex=True) 
    fig2, axs2 = plt.subplots(3, 2, figsize=figsize(1.2, vscale=1), sharex=True) 

    for i, ax in enumerate(axs1.flatten()):

        #data definition
        ydat = datarr[:, col1[i]] - datarr[:, col2[i]]

        ax.scatter(datz, 
                   ydat, 
                   color='black', 
                   s=5)

        ax.plot(datz,
                modarr[:,col1[i]] - modarr[:, col2[i]],
                color=cs[0])


        ax.set_ylabel(col_label[i])
    
        ax.set_ylim(np.nanmin(ydat)-0.2, np.nanmax(ydat)+0.2)


    for i, ax in enumerate(axs2.flatten()):
    
        i += 6 

        #data definition
        ydat = datarr[:, col1[i]] - datarr[:, col2[i]]

        ax.scatter(datz, 
                   ydat, 
                   color='black', 
                   s=5)

        ax.plot(datz,
                modarr[:,col1[i]] - modarr[:, col2[i]],
                color=cs[0])


        ax.set_ylabel(col_label[i])
    
        ax.set_ylim(np.nanmin(ydat)-0.2, np.nanmax(ydat)+0.2)

    # Now with flux correction 

    mydir = '/home/lc585/qsosed'
    flxcorr = np.genfromtxt(os.path.join(mydir, 'flxcorr.dat'))
    f = interp1d(flxcorr[:, 0], flxcorr[:, 1], bounds_error=False, fill_value=1.0)
    flxcorr = f(wavlen) 

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
                      qsomag)

    for i, ax in enumerate(axs1.flatten()):

        ax.plot(datz,
                modarr[:,col1[i]] - modarr[:, col2[i]],
                color=cs[1])


    for i, ax in enumerate(axs2.flatten()):
    
        i += 6 

        ax.plot(datz,
                modarr[:,col1[i]] - modarr[:, col2[i]],
                color=cs[1])


    axs1[2, 0].set_xlabel(r'Redshift $z$')
    axs2[2, 0].set_xlabel(r'Redshift $z$')
    axs1[2, 1].set_xlabel(r'Redshift $z$')
    axs2[2, 1].set_xlabel(r'Redshift $z$')

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig('/home/lc585/thesis/figures/chapter05/sed_color_plot_1.pdf')
    fig2.savefig('/home/lc585/thesis/figures/chapter05/sed_color_plot_2.pdf')


    plt.show() 

    return None


if __name__ == '__main__':
    plot() 