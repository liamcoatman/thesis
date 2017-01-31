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

def plot(): 

    """
    Generates residual plot using model parameters in input.yml
    This is not the same as the uncorrected version in the thesis
    """

    with open('input.yml', 'r') as f:
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
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()

    with open(parfile['run']['ftgd_file']) as f:
        modz = np.loadtxt(f, skiprows=1, usecols=(0,))

    # Load filters
    ftrlst = []
    lameff = []

    if (parfile['run']['ftrlst']['u'] == 'y'):
        ftrlst.append('u')
        lameff.append(3546.0)
    if parfile['run']['ftrlst']['g'] == 'y':
        ftrlst.append('g')
        lameff.append(4670.0)
    if parfile['run']['ftrlst']['r'] == 'y':
        ftrlst.append('r')
        lameff.append(6156.0)
    if parfile['run']['ftrlst']['i'] == 'y':
        ftrlst.append('i')
        lameff.append(7471.0)
    if parfile['run']['ftrlst']['z'] == 'y':
        ftrlst.append('z')
        lameff.append(8918.0)
    if parfile['run']['ftrlst']['Y'] == 'y':
        ftrlst.append('Y')
        lameff.append(10305.0)
    if parfile['run']['ftrlst']['J'] == 'y':
        ftrlst.append('J')
        lameff.append(12483.0)
    if parfile['run']['ftrlst']['H'] == 'y':
        ftrlst.append('H')
        lameff.append(16313.0)
    if parfile['run']['ftrlst']['K'] == 'y':
        ftrlst.append('K')
        lameff.append(22010.0)
    if parfile['run']['ftrlst']['W1'] == 'y':
        ftrlst.append('W1')
        lameff.append(33680.0)
    if parfile['run']['ftrlst']['W2'] == 'y':
        ftrlst.append('W2')
        lameff.append(46180.0)
    if parfile['run']['ftrlst']['W3'] == 'y':
        ftrlst.append('W3')
        lameff.append(120000.0)
    if parfile['run']['ftrlst']['W4'] == 'y':
        ftrlst.append('W4')
        lameff.append(220000.0)

    ftrlst = np.array(ftrlst)
    nftr = len(ftrlst)
    bp = np.empty(nftr,dtype='object')
    dlam = np.zeros(nftr)

    for nf in range(nftr):
        with open('/home/lc585/Dropbox/IoA/QSOSED/Model/Filter_Response/'+ftrlst[nf]+'.response','r') as f:
            wavtmp, rsptmp = np.loadtxt(f,unpack=True)
        dlam[nf] = (wavtmp[1] - wavtmp[0])
        bptmp = np.ndarray(shape=(2,len(wavtmp)), dtype=float)
        bptmp[0,:], bptmp[1,:] = wavtmp, rsptmp
        bp[nf] = bptmp

    zromag = np.zeros(len(bp))

    for ftr in range(len(bp)):
        sum1 = np.sum( bp[ftr][1] * (1.0/(bp[ftr][0]**2)) * bp[ftr][0] * dlam[ftr])
        sum2 = np.sum( bp[ftr][1] * bp[ftr][0] * dlam[ftr])
        flxlam = sum1 / sum2
        zromag[ftr] = -2.5 * np.log10(flxlam)

    # Load ftgd
    cptftrlst = np.array(['u','g','r','i','z','Y','J','H','K','W1','W2','W3','W4'])
    
    with open(parfile['run']['ftgd_file']) as f:
        modz = np.loadtxt(f, skiprows=1, usecols=(0,))

    with open(parfile['run']['ftgd_file']) as f:
        ftgd = np.zeros((len(modz),nftr))
        ftgdcpt = np.loadtxt(f,skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
    
    for nf in range(nftr):
        i = np.where( cptftrlst == ftrlst[nf] )[0][0]
        ftgd[:,nf] = ftgdcpt[:,i]


    if parfile['run']['flxcorr_file'] == 'None':
        flxcorr = np.array( [1.0] * len(wavlen) )
        
    else:
        with open(parfile['run']['flxcorr_file'],'rb') as f:
            flxcorr = pickle.load(f)


    # load data 
    datarr, bincount = loaddat(fittingobj, ftrlst, modz, parfile)

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
                      modz,
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
                      ftgd,
                      ftrlst,
                      whmin,
                      whmax,
                      cosmo,
                      flxcorr)
    
    lameff = fittingobj.get_lameff()
    lameff = lameff.reshape( len(lameff), 1)
    lameff = np.repeat(lameff,len(modz),axis=1)
    modz = modz.reshape( 1, len(modz) )
    modz = np.repeat(modz,len(lameff),axis=0)
    lam = lameff / (1.0 + modz)
    res = np.ndarray.transpose(modarr - datarr)
    
    # Not used in fit
    res[0,10:] = 0.0
    res[1,17:] = 0.0
    res[2,30:] = 0.0
    
    fig = plt.figure(figsize=(7,4))
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
    ax.set_ylim(-0.4,0.4)
    ax.set_xlabel(r'Rest Frame Wavelength (${\rm \AA}$)',fontsize=12)
    ax.set_ylabel(r'$m_{\rm mod} - m_{\rm dat}$',fontsize=12)
    plt.legend(prop={'size':10})
    plt.tick_params(axis='both',which='major',labelsize=10)
    plt.tight_layout()

    plt.savefig('/home/lc585/thesis/figures/chapter06/model_residuals_with_correction.pdf')

    plt.show() 

    return None 
