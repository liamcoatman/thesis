# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:44:37 2014

@author: lc585

Make plots of fit to whole sample
"""
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

    np.seterr(all='ignore')

    tmparr,sigmatmp,balflg,name, snr = rd.loadmagdat(parfile['run']['datset'],
                                                     parfile['run']['cat'],
                                                     True,
                                                     parfile['run']['imin'],
                                                     parfile['run']['imax'],
                                                     parfile['run']['snrmin'] )
    nftr = len(ftrlst)
    cptftrlst = np.array(['rs','u','g','r','i','z','Y','J','H','K','W1','W2','W3','W4'])

    ind = []
    for f in range(nftr):
        ind.append(np.where( cptftrlst == ftrlst[f] )[0][0])

    if (parfile['run']['balflg_on'] is True):

        tmparr = tmparr[balflg == parfile['run']['balflg']]
        sigma = sigmatmp[balflg == parfile['run']['balflg']]
        snr = snr[balflg == parfile['run']['balflg']]

        datz = tmparr[:,0]
        datmag = tmparr[:,ind]
        sigma = sigma[:,ind]
        snr = snr[:,ind]

    elif (parfile['run']['balflg_on'] is False):

        datz = tmparr[:,0]
        datmag = tmparr[:,ind]
        sigma = sigmatmp[:,ind]
        snr = snr[:,ind]

    magmask = (ma.getmask(datmag)) | (ma.getmask(sigma)) | (ma.getmask(snr))
    datmag = ma.masked_array(datmag,mask=magmask)

    # load data 
    datarr, bincount = loaddat(fittingobj, ftrlst, modz, parfile)

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    mycm = cm.get_cmap('YlOrRd_r')
    mycm.set_under('w')
    mycm = truncate_colormap(mycm, 0.0, 0.8)

    cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors

    col1 = [0,1,2,3,4,5,6,7,8,9,7,8]
    col2 = [1,2,3,4,5,6,7,8,9,10,9,10]
    y_low_lim = [-0.5,-0.4,-0.3,-0.2,-0.4,-0.4,-0.3,-0.3,-0.4,-0.2,-0.2,-0.5]
    y_up_lim = [1.5,0.8,0.5,0.5,0.5,0.8,0.4,0.6,1.0,1.0,1.6,2.0]
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

    lammid = [np.mean( [lameff[0],lameff[1]] ),
              np.mean( [lameff[1],lameff[2]] ),
              np.mean( [lameff[2],lameff[3]] ),
              np.mean( [lameff[3],lameff[4]] ),
              np.mean( [lameff[4],lameff[5]] ),
              np.mean( [lameff[5],lameff[6]] ),
              np.mean( [lameff[6],lameff[7]] ),
              np.mean( [lameff[7],lameff[8]] ),
              np.mean( [lameff[8],lameff[9]] ),
              np.mean( [lameff[9],lameff[10]] ),
              np.mean( [lameff[7],lameff[9]] ),
              np.mean( [lameff[8],lameff[10]] ) ]

    outfile = ['ug','gr','ri','iz','zy','yj','jh','hk','kw1','w1w2','hw1','kw2']

    for i in range(12):

        fig, ax = plt.subplots(figsize=(5,4))

        #histogram definition
        xyrange = [[0.5,3],[-0.5,1]] # data range
        bins = [80,50] # number of bins
        thresh = 3  #density threshold

        #data definition
        xdat, ydat = datz, datmag[:,col1[i]] - datmag[:,col2[i]]

        # histogram the data
        hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)
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
                       )
        ax.scatter(xdat1, ydat1,color=cset[-1],s=5)

        ax.plot(modz,
                modarr[:,col1[i]] - modarr[:,col2[i]],
                color='k',
                linewidth=2.0)

        ax.plot(modz,
                datarr[:,col1[i]] - datarr[:,col2[i]],
                markerfacecolor='k',
                linestyle='',
                marker='o',
                markersize=5.0)

        ax.set_ylim(y_low_lim[i],y_up_lim[i])
        ax.set_xlim(0.5,3.0)
        ax.set_ylabel(col_label[i],fontsize=14)
        ax.set_xlabel(r'$z$',fontsize=14)
        ax.tick_params(axis='both',which='major',labelsize=10)

        plt.tight_layout()
        plt.savefig('/home/lc585/thesis/figures/chapter06/sed_color_plots/'+ outfile[i] + '.jpg',format='jpg')
        plt.clf()
        


    return None


if __name__ == '__main__':
    plot() 