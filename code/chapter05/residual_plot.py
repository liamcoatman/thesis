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

    # Load filters
    ftrlst = fittingobj.get_ftrlst()[:-2] 
    lameff = fittingobj.get_lameff()[:-2]
    bp = fittingobj.get_bp()[:-2] # these are in ab and data is in vega 
    dlam = fittingobj.get_bp()[:-2]
    zromag = fittingobj.get_zromag()[:-2]

    # load data 
    fname = '/home/lc585/qsosed/sdss_ukidss_wise_medmag_ext.dat'
    datarr = np.genfromtxt(fname, usecols=(1,3,5,7,9,11,13,15,17,19,21)) 
    datarr[datarr < 0.0] = np.nan 

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
    
    lameff = fittingobj.get_lameff()
    lameff = lameff.reshape( len(lameff), 1)
    lameff = np.repeat(lameff,len(datz),axis=1)
    datz = datz.reshape(1, len(datz) )
    datz = np.repeat(datz,len(lameff),axis=0)
    lam = lameff / (1.0 + datz)
    res = np.ndarray.transpose(modarr - datarr)
    
    # Not used in fit
    res[0,10:] = 0.0
    res[1,17:] = 0.0
    res[2,30:] = 0.0
    
    fig = plt.figure(figsize=figsize(1, 0.8))
    ax = fig.add_subplot(1,1,1)
    colormap = plt.cm.Paired
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 11)])
    labels = ['u','g','r','i','z','Y','J','H','K','W1','W2']
    for i in range(11):
        ax.semilogx(lam[i,:],res[i,:],label=labels[i])
    
    ax.axhline(0,color='black')
    # ax.axvline(x = 1216.0, color='r', linestyle='--') # Ly_alpha / NV
    # plt.axvline(x = 1400.0, color='r', linestyle='--') # SiIV / OIV
    # plt.axvline(x = 1549.0, color='r', linestyle='--') # CIV
    # ax.axvline(x = 1909.0, color='r', linestyle='--') # CIII]
    # plt.axvline(x = 2326.0, color='r', linestyle='--') # CII]
    # plt.axvline(x = 2798.0, color='r', linestyle='--') # MgII
    # plt.axvline(x = 3426.0, color='r', linestyle='--') # [NeV]
    # plt.axvline(x = 3727.0, color='r', linestyle='--') # [OII]
    # plt.axvline(x = 3869.0, color='r', linestyle='--') # [NeIII]
    # plt.axvline(x = 4102.0, color='r', linestyle='--') # H_delta
    # plt.axvline(x = 4340.0, color='r', linestyle='--') # H_gamma
    # ax.axvline(x = 4861.0, color='r', linestyle='--') # H_beta
    # ax.axvline(x = 4983.0, color='r', linestyle='--') # [OIII]
    # ax.axvline(x = 6563.0, color='black', linestyle='--') # H_alpha
    # ax.text(7000,-0.35,r'H$\alpha$',horizontalalignment='left',verticalalignment='center')
    # plt.axvline(x = 18700.0, color='r', linestyle='--') # Pa_alpha
    
    ax.set_xlim(1000,50000)
    ax.set_ylim(-0.3,0.3)
    ax.set_xlabel(r'Rest Frame Wavelength [${\rm \AA}$]',fontsize=12)
    ax.set_ylabel(r'$m_{\rm mod} - m_{\rm dat}$',fontsize=12)
    plt.legend(prop={'size':10})
    plt.tick_params(axis='both',which='major',labelsize=10)
    plt.tight_layout()

    plt.savefig('/home/lc585/thesis/figures/chapter05/model_residuals.pdf')

    plt.show() 

    return None 
