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
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties


def plot(): 

    """
    Generates residual plot using model parameters in input.yml
    """

    set_plot_properties() # change style 

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

    fname = '/home/lc585/qsosed/sdss_ukidss_wise_medmag_ext.dat'
    datarr = np.genfromtxt(fname, usecols=(5,7,9,11,13,15,17,19,21)) 
    datarr[datarr < 0.0] = np.nan 

    datarr = datarr[:-5, :]


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
    
    lameff = lameff.reshape( len(lameff), 1)
    lameff = np.repeat(lameff,len(datz),axis=1)
    
    datz = datz.reshape(1, len(datz) )
    datz = np.repeat(datz,len(lameff),axis=0)
    lam = lameff / (1.0 + datz)
    res = np.ndarray.transpose(modarr - datarr)
    

    
    fig = plt.figure(figsize=figsize(1, 0.8))
    ax = fig.add_subplot(1,1,1)
    colormap = plt.cm.Paired
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.8, 9)])
    labels = ['r','i','z','Y','J','H','K','W1','W2']
    for i in range(9):
        ax.plot(lam[i,:], res[i,:], label=labels[i])
    
    ax.grid() 
        
    ax.set_xlim(1000,30000)
    ax.set_ylim(-0.3,0.3)
    ax.set_xlabel(r'Rest Frame Wavelength [${\rm \AA}$]')
    ax.set_ylabel(r'$m_{\rm mod} - m_{\rm dat}$')
    plt.legend(prop={'size':10})
    plt.tick_params(axis='both',which='major')
    plt.tight_layout()

    plt.savefig('/home/lc585/thesis/figures/chapter05/model_residuals.pdf')

    plt.show() 

    return None 
