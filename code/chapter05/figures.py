import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, join
import brewer2mpl
import palettable 
cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
from scipy import stats 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
set_plot_properties() # change style 

def bbt_vs_z_with_correction():

    tab = Table.read('/data/lc585/QSOSED/Results/140530/out1.fits')
    
    newtab = tab[ (tab['BBT'] > 600.) & (tab['BBT'] < 1800.) ]
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot( newtab['Z'], newtab['BBT'], linestyle='', marker='o', markersize=2, markerfacecolor='grey', markeredgecolor='None', alpha=0.5)
    
    yrun = running.RunningMedian(tab['BBT'],101)[0::50]
    xrun = running.RunningMedian(tab['Z'],101)[0::50]
    
    ax.plot(xrun,yrun,linewidth=2,color='black')
    
    tab = Table.read('/data/lc585/QSOSED/Results/140530/out2.fits')
    
    yrun = running.RunningMedian(tab['BBT'],101)[0::50]
    xrun = running.RunningMedian(tab['Z'],101)[0::50]
    
    ax.plot(xrun,yrun,linewidth=2,color='blue')
    
    
    ax.set_xlabel(r'$z$',fontsize=12)
    ax.set_ylabel(r'T$_{\rm BB}$',fontsize=12)
    
    plt.tick_params(axis='both',which='major',labelsize=10)
    plt.tight_layout()
    plt.savefig('bbt_z.pdf')
    
    plt.show() 

    return None 


def shang_sed():

    """
    Shang SED model
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.constants import c
    
    rltab = np.genfromtxt('rlmsedMR.txt')
    rqtab = np.genfromtxt('rqmsedMR.txt')
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    
    ax.plot( 1.0e10 * c / (10**rltab[:,0]), 10**rltab[:,1], color='black', label='Radio-loud' )
    ax.plot( 1.0e10 * c / (10**rqtab[:,0]), 10**rqtab[:,1], color='green', label='Radio-quiet' )
    
    ax.axvline(912,color='black',linestyle='--')
    ax.axvline(10**4,color='black',linestyle='--')
    ax.axvline(6.e5,color='black',linestyle='--')
    
    ax.set_xlabel(r'Wavelength $\AA$',fontsize=10)
    ax.set_ylabel( r'log(${\lambda}f_{\lambda}$) (Arbitrary Units)',fontsize=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_ylim(1e-3,12)
    ax.set_xlim(10,10**8)
    
    plt.legend( loc='lower left', prop={'size':10} )
    
    ax.text(1.3e3, 2e-1, 'Big Blue \n Bump', fontsize=12)
    ax.text(2e4, 2e-1, 'Near-IR Bump', fontsize=12)
    
    plt.tick_params(axis='both',which='major',labelsize=10)
    plt.tight_layout()
    
    plt.savefig('shangsed.pdf')

    return None 


def ntt_proposal_figure2():

    """
    Plot from NTT telsecope proposal
    """

    import yaml 
    from load import load
    import cPickle as pickle
    import numpy as np
    from astropy.table import Table 
    from model import model 
    from pl import pl
    from scipy.interpolate import interp1d 
    import matplotlib.pyplot as plt 

    with open('input.yml', 'r') as f:
        parfile = yaml.load(f)

    fittingobj = load(parfile)

    wavlen = fittingobj.get_wavlen()
    with open('/data/lc585/QSOSED/Results/140811/allsample_2/fluxcorr.array','rb') as f:
        flxcorr = pickle.load(f)

    flxcorr = np.ones(len(wavlen))
    zromag = fittingobj.get_zromag()
    bp = fittingobj.get_bp()
    dlam = fittingobj.get_dlam()

    tab1 = Table.read('/data/lc585/QSOSED/Results/140912/lowbetatab_3.fits')
    tab2 = Table.read('/data/lc585/QSOSED/Results/140912/highbetatab_3.fits')

    magcolumns = ['SDSS_UMAG',
                  'SDSS_GMAG',
                  'SDSS_RMAG',
                  'SDSS_IMAG',
                  'SDSS_ZMAG',
                  'UKIDSS_YMAG',
                  'UKIDSS_JMAG',
                  'UKIDSS_HMAG',
                  'UKIDSS_KMAG',
                  'ALLWISE_W1MAG',
                  'ALLWISE_W2MAG',
                  'ALLWISE_W3MAG',
                  'ALLWISE_W4MAG']

    datmag1 = np.array([np.array(tab1[i]) for i in magcolumns])
    datmag1 = datmag1 - datmag1[3, :] + 18.0

    datmag2 = np.array([np.array(tab2[i]) for i in magcolumns])
    datmag2 = datmag2 - datmag2[3, :] + 18.0

    plslp1 = 0.46
    plslp2 = 0.03
    plbrk = 2822.50
    bbt = 1216.32
    bbflxnrm = 0.24
    galfra = 0.31
    elscal = 0.71
    scahal = 0.86
    ebv = 0.0

    magtmp, wavlentmp, fluxtmp = model(plslp1,
                                       plslp2,
                                       plbrk,
                                       bbt,
                                       bbflxnrm,
                                       elscal,
                                       scahal,
                                       galfra,
                                       ebv,
                                       18.0,
                                       2.,
                                       fittingobj,
                                       flxcorr,
                                       parfile)

    wavnumjoin =  (np.abs(wavlen - parfile['runhotdustplfit']['wavmin_bbpl'] )).argmin()
    wavnummax = (np.abs(wavlen - parfile['runhotdustplfit']['wavmax_bbpl'] )).argmin()


    slope = 1.852337650258
    nrm = fluxtmp[wavnumjoin] / (wavlen[wavnumjoin]**(slope - 2.0))

    plmodel = pl(wavlen,slope,nrm)

    newflux = np.zeros(len(wavlen))
    newflux[:wavnumjoin] = fluxtmp[:wavnumjoin]
    newflux[wavnumjoin:] = plmodel[wavnumjoin:]

    # Calculate normalised model flux
    spc = interp1d(wavlentmp,newflux,bounds_error=False,fill_value=0.0)
    sum1 = np.sum( bp[3][1] * spc(bp[3][0]) * bp[3][0] * dlam[3])
    sum2 = np.sum( bp[3][1] * bp[3][0] * dlam[3])
    flxlam = sum1 / sum2
    flxlam = flxlam + 1e-200
    imag = (-2.5 * np.log10(flxlam)) - zromag[3]
    delta_m = 18.0 - imag # what i must add to model magnitude to match data
    fnew = newflux * 10**(-0.4 * delta_m) # this is normalised flux in erg/cm^2/s/A

    ### Calculate model with beta = 0 for comparison.
    slope = 0.9119355988796
    nrm = fluxtmp[wavnumjoin] / (wavlen[wavnumjoin]**(slope - 2.0))
    plmodel2 = pl(wavlen, slope, nrm)
    newflux2 = np.zeros(len(wavlen))
    newflux2[:wavnumjoin] = fluxtmp[:wavnumjoin]
    newflux2[wavnumjoin:] = plmodel2[wavnumjoin:]

    # Calculate normalised model flux
    spc = interp1d(wavlentmp,newflux2,bounds_error=False,fill_value=0.0)
    sum1 = np.sum( bp[3][1] * spc(bp[3][0]) * bp[3][0] * dlam[3])
    sum2 = np.sum( bp[3][1] * bp[3][0] * dlam[3])
    flxlam = sum1 / sum2
    flxlam = flxlam + 1e-200
    imag = (-2.5 * np.log10(flxlam)) - zromag[3]
    delta_m = 18.0 - imag # what i must add to model magnitude to match data
    fnew2 = newflux2 * 10**(-0.4 * delta_m) # this is normalised flux in erg/cm^2/s/A

    flam, lameff = np.zeros((len(tab1),13)), np.zeros((len(tab1),13))

    for obj in range(len(tab1)):

        lameff[obj,:] = fittingobj.get_lameff() / (1.0 + tab1[obj]['Z_1'])

        datmagtmp = datmag1[:,obj]

        # Calculate data fluxes from magnitudes
        f_0 = np.zeros(len(bp)) # flux zero points
        for ftr in range(len(bp)):
            sum1 = np.sum( bp[ftr][1] * (0.10893/(bp[ftr][0]**2)) * bp[ftr][0] * dlam[ftr])
            sum2 = np.sum( bp[ftr][1] * bp[ftr][0] * dlam[ftr])
            f_0[ftr] = sum1 / sum2
        flam[obj,:] = f_0 * 10.0**( -0.4 * datmagtmp ) # data fluxes in erg/cm^2/s/A

    flam_2, lameff_2 = np.zeros((len(tab1),13)), np.zeros((len(tab1),13))

    for obj in range(len(tab2)):

        lameff_2[obj,:] = fittingobj.get_lameff() / (1.0 + tab2[obj]['Z_1'])

        datmagtmp = datmag2[:,obj]

        # Calculate data fluxes from magnitudes
        f_0 = np.zeros(len(bp)) # flux zero points
        for ftr in range(len(bp)):
            sum1 = np.sum( bp[ftr][1] * (0.10893/(bp[ftr][0]**2)) * bp[ftr][0] * dlam[ftr])
            sum2 = np.sum( bp[ftr][1] * bp[ftr][0] * dlam[ftr])
            f_0[ftr] = sum1 / sum2
        flam_2[obj,:] = f_0 * 10.0**( -0.4 * datmagtmp ) # data fluxes in erg/cm^2/s/A


    # Manda's Very Red Quasars
    redcat = np.genfromtxt('/data/lc585/QSOSED/Results/140920/Red_Quasar_photom.cat')

    flam_3, lameff_3 = np.zeros((len(redcat),3)), np.zeros((len(redcat),3))

    for obj in range(len(redcat)):

        lameff_3[obj,0] =  33680.0 / (1.0 + redcat[obj,12])
        lameff_3[obj,1] =  46180.0 / (1.0 + redcat[obj,12])
        lameff_3[obj,2] =  120000.0 / (1.0 + redcat[obj,12])


        # Calculate data fluxes from magnitudes
        f_0 = np.zeros(len(bp)) # flux zero points
        for ftr in range(len(bp)):
            sum1 = np.sum( bp[ftr][1] * (0.10893/(bp[ftr][0]**2)) * bp[ftr][0] * dlam[ftr])
            sum2 = np.sum( bp[ftr][1] * bp[ftr][0] * dlam[ftr])
            f_0[ftr] = sum1 / sum2

        flam_3[obj,0] = f_0[9] * 10.0**( -0.4 * redcat[obj,2] )
        flam_3[obj,1] = f_0[10] * 10.0**( -0.4 * redcat[obj,3] )
        flam_3[obj,2] = f_0[11] * 10.0**( -0.4 * redcat[obj,4] )


    nrm = fnew[520] * wavlen[520]

    flam = flam / nrm
    flam_2 = flam_2 / nrm
    flam_3 = flam_3 / nrm

    w1med = np.median(lameff[:,9]*flam[:,9])
    w2med = np.median(lameff[:,10]*flam[:,10])
    w3med = np.median(lameff[:,11]*flam[:,11])
    w1med_2 = np.median(lameff_2[:,9]*flam_2[:,9])
    w2med_2 = np.median(lameff_2[:,10]*flam_2[:,10])
    w3med_2 = np.median(lameff_2[:,11]*flam_2[:,11])

    w1err = np.std(lameff[:,9]*flam[:,9])
    w2err = np.std(lameff[:,10]*flam[:,10])
    w3err = np.std(lameff[:,11]*flam[:,11])
    w1err_2 = np.std(lameff_2[:,9]*flam_2[:,9])
    w2err_2 = np.std(lameff_2[:,10]*flam_2[:,10])
    w3err_2 = np.std(lameff_2[:,11]*flam_2[:,11])

    w1lam = np.median(lameff[:,9])
    w2lam = np.median(lameff[:,10])
    w3lam = np.median(lameff[:,11])
    w1lam_2 = np.median(lameff_2[:,9])
    w2lam_2 = np.median(lameff_2[:,10])
    w3lam_2 = np.median(lameff_2[:,11])

#    print w1med, w2med, w3med, w1med_2, w2med_2, w3med_2, w1lam, w2lam, w3lam, w1lam_2, w2lam_2, w3lam_2

    import matplotlib
    import prettyplotlib as ppl

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1,1,1)

    fnew = fnew / nrm
    fnew2 = fnew2 / nrm
    ax.plot(wavlen,wavlen*fnew,color='black')
    ax.plot(wavlen[1900:],wavlen[1900:]*fnew2[1900:],color='black')
    ax.errorbar(w1lam,w1med,yerr=w1err,color='blue')
    ax.errorbar(w2lam,w2med,yerr=w2err,color='blue')
    ax.errorbar(w3lam,w3med,yerr=w3err,color='blue')
    ax.errorbar(w1lam_2,w1med_2,yerr=w1err_2,color='red')
    ax.errorbar(w2lam_2,w2med_2,yerr=w2err_2,color='red')
    ax.errorbar(w3lam_2,w3med_2,yerr=w3err_2,color='red')


#    ppl.scatter(lameff_3[:,0] , lameff_3[:,0] * flam_3[:,0], color='grey', alpha=0.5)
#    ppl.scatter(lameff_3[:,1] , lameff_3[:,1] * flam_3[:,1], color='grey', alpha=0.5)
#    ppl.scatter(lameff_3[:,2] , lameff_3[:,2] * flam_3[:,2], color='grey', alpha=0.5)

    # Or plot lines
    for i in range(len(lameff_3)):
        xdat = np.array([lameff_3[i,0],lameff_3[i,1],lameff_3[i,2]])
        ydat = np.array([lameff_3[i,0] * flam_3[i,0],
                lameff_3[i,1] * flam_3[i,1],
                lameff_3[i,2] * flam_3[i,2]])
        nrmind = np.argmin( np.abs(wavlen - xdat[0]) )
        ydat = ydat * (wavlen[nrmind] * fnew[nrmind]) / ydat[0]
        ppl.scatter(xdat,ydat, color='grey', alpha=0.5)
        ppl.plot(xdat,ydat,color='grey',alpha=0.2)


    ax.set_xlim(1200,50000 )
    ax.set_ylim(0,3.5)

    ax.set_ylabel(r'Relative Flux $\lambda F_{\lambda}(\lambda)$',fontsize=10)
    ax.set_xlabel(r'Rest-frame Wavelength ($\AA$)',fontsize=10)
    plt.text(17699,0.21607,r'$\beta_{\rm NIR}=-0.09$',fontsize=8,horizontalalignment='left',verticalalignment='top')
    plt.text(17000,0.6,r'$\beta_{\rm NIR}=0.85$',fontsize=8,rotation=20.0,horizontalalignment='left',verticalalignment='bottom')
    plt.text(6604,2.23966,r'H$\alpha$',fontsize=8,rotation=90.0,horizontalalignment='center',verticalalignment='bottom')
    plt.text(4902,1.21462,r'H$\beta$ & OIII',fontsize=8,rotation=90.0,horizontalalignment='center',verticalalignment='bottom')
    plt.text(1558,2.3328,r'CIV',fontsize=8,rotation=90.0,horizontalalignment='center',verticalalignment='bottom')

    plt.tick_params(axis='both',which='major',labelsize=8)
    ax.set_xscale('log')
    ax.set_xticks([2000,5000,10000,20000,40000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    # plt.savefig('/home/lc585/Dropbox/IoA/NTT_Proposal_95A/esoform-95A/figure2_v2.pdf')
    plt.show()

    return None

def dr7_completeness():

    

    # Manda matched DR7Q to ALLWISE 
    wisedat = Table.read('/data/sdss/DR7/AllWISE/DR7QSO_AllWISE_matched.fits')

    # Main DR7Q
    dr7dat = Table.read('/data/sdss/DR7/dr7qso.fit')

    imin = np.arange(18.0,21.0,0.1)

    w1frac, w2frac, w3frac, w4frac = [], [], [], [] 
    for i in range(len(imin)):
        objsall = wisedat[ (wisedat['IMAG'] < imin[i]) ]
        objs = wisedat[ (wisedat['IMAG'] < imin[i]) & (wisedat['W1SNR_ALLWISE'] > 5.0) ]
        w1frac.append(float(len(objs)) / float(len(objsall)) )
        objs = wisedat[ (wisedat['IMAG'] < imin[i]) & (wisedat['W2SNR_ALLWISE'] > 5.0) ]
        w2frac.append(float(len(objs)) / float(len(objsall)))
        objs = wisedat[ (wisedat['IMAG'] < imin[i]) & (wisedat['W3SNR_ALLWISE'] > 5.0) ]
        w3frac.append(float(len(objs)) / float(len(objsall))) 
        objs = wisedat[ (wisedat['IMAG'] < imin[i]) & (wisedat['W4SNR_ALLWISE'] > 5.0) ]
        w4frac.append(float(len(objs)) / float(len(objsall)))
    
    ulasdat = np.genfromtxt('/data/mbanerji/Projects/QSO/DR7QSO/DR7QSO_ULASDR9_ABmags.cat')
    
    YSNR = 1.0 / (ulasdat[:,18] * 0.4 * np.log(10) )
    JSNR = 1.0 / (ulasdat[:,19] * 0.4 * np.log(10) )
    HSNR =  1.0 / (ulasdat[:,20] * 0.4 * np.log(10) )
    KSNR =  1.0 / (ulasdat[:,21] * 0.4 * np.log(10) )
    
    Yfrac, Jfrac, Hfrac, Kfrac, totalnum = [], [], [], [], [] 
    for i in range(len(imin)):
        objsall = ulasdat[ (ulasdat[:,7] < imin[i]) ]
        objs = ulasdat[ (ulasdat[:,7] < imin[i]) & (YSNR > 5.0) ]
        Yfrac.append(float(len(objs)) / float(len(objsall)) )
        objs = ulasdat[ (ulasdat[:,7] < imin[i]) & (JSNR > 5.0)]
        Jfrac.append(float(len(objs)) / float(len(objsall)) )
        objs = ulasdat[ (ulasdat[:,7] < imin[i]) & (HSNR > 5.0) ]
        Hfrac.append(float(len(objs)) / float(len(objsall)) )
        objs = ulasdat[ (ulasdat[:,7] < imin[i]) & (KSNR > 5.0) ]
        Kfrac.append(float(len(objs)) / float(len(objsall)) )
    
    totalnum = []
    for i in range(len(imin)):
        objsall = dr7dat[ (dr7dat['IMAG'] < imin[i]) ]
        totalnum.append(len(objsall))
    
    fig = plt.figure(figsize=figsize(0.7))
    ax = fig.add_subplot(111)
    
    set2 = brewer2mpl.get_map('Set2', 'qualitative', 7).mpl_colors
    ax.plot(imin,Yfrac,label='Y',color=set2[0])
    ax.plot(imin,Jfrac,label='J',color=set2[1])
    ax.plot(imin,Hfrac,label='H',color=set2[2])
    ax.plot(imin,Kfrac,label='K',color=set2[3])
    ax.plot(imin,w1frac,label='W1',color=set2[4])
    ax.plot(imin,w2frac,label='W2',color=set2[5])
    ax.plot(imin,w3frac,label='W3',color=set2[6])
    
    plt.legend(loc='lower left',prop={'size':10})
    
    ax2 = ax.twinx()
    ax2.plot(imin,totalnum,color='black',linestyle='--')
    
    ax2.axvline(19.1,color='grey')
    ax.set_ylim(0.6,1.05)
    ax2.set_ylim(0,110000)
    
    ax.set_xlabel(r'$i_{\rm min}$')
    ax.set_ylabel('Completeness')
    ax2.set_ylabel('Number of Objects')
    
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/dr7completeness.pdf')
    
    plt.show() 

    return None 

def dr10_completeness():

    wisedat = Table.read('/data/lc585/SDSS/DR10QSO_AllWISE_matched.fits')
    
    dat = Table.read('/data/sdss/DR10/DR10Q_v2.fits')
    gmag_all = dat['PSFMAG'][:,1] - dat['EXTINCTION_RECAL'][:,1] + 0.03
    
    gmag_wise = wisedat['PSFMAG'][:,1] - wisedat['EXTINCTION_RECAL'][:,1] + 0.03

    
    ysnr = dat['YFLUX'] / dat['YFLUX_ERR']
    jsnr = dat['JFLUX'] / dat['JFLUX_ERR']
    hsnr = dat['HFLUX'] / dat['HFLUX_ERR']
    ksnr = dat['KFLUX'] / dat['KFLUX_ERR']
    
    gmin = np.arange(18.0,22.5,0.1)
    
    yfrac, jfrac, hfrac, kfrac, w1frac, w2frac, w3frac, w4frac, totalnum = [], [], [], [], [], [], [], [], []
    
    for i in range(len(gmin)):
        objsall = wisedat[ (gmag_wise < gmin[i]) ]
        objs = wisedat[ (gmag_wise < gmin[i]) & (wisedat['W1SNR_ALLWISE'] > 5.0) ]
        w1frac.append(float(len(objs)) / float(len(objsall)) )
        objs = wisedat[ (gmag_wise < gmin[i]) & (wisedat['W2SNR_ALLWISE'] > 5.0) ]
        w2frac.append(float(len(objs)) / float(len(objsall)))
        objs = wisedat[ (gmag_wise < gmin[i]) & (wisedat['W3SNR_ALLWISE'] > 5.0) ]
        w3frac.append(float(len(objs)) / float(len(objsall))) 
        objs = wisedat[ (gmag_wise < gmin[i]) & (wisedat['W4SNR_ALLWISE'] > 5.0) ]
        w4frac.append(float(len(objs)) / float(len(objsall)))
        objsall = dat[ (dat['UKIDSS_MATCHED'] == 1) & (gmag_all < gmin[i]) ]
        objs = dat[ (dat['UKIDSS_MATCHED'] == 1) & (gmag_all < gmin[i])  & (ysnr > 5.0) ]
        yfrac.append(float(len(objs)) / float(len(objsall)) )
        objs = dat[(dat['UKIDSS_MATCHED'] == 1) & (gmag_all < gmin[i]) & (jsnr > 5.0) ]
        jfrac.append(float(len(objs)) / float(len(objsall)) )
        objs = dat[(dat['UKIDSS_MATCHED'] == 1) & (gmag_all < gmin[i]) & (hsnr > 5.0) ]
        hfrac.append(float(len(objs)) / float(len(objsall)) )
        objs = dat[(dat['UKIDSS_MATCHED'] == 1) & (gmag_all < gmin[i]) & (ksnr > 5.0) ]
        kfrac.append(float(len(objs)) / float(len(objsall)) )
    
    for i in range(len(gmin)):
        objsall = dat[ (gmag_all < gmin[i]) ]
        totalnum.append(len(objsall))
    
    fig = plt.figure(figsize=figsize(0.7))
    ax = fig.add_subplot(111)
    
    set2 = brewer2mpl.get_map('Set2', 'qualitative', 7).mpl_colors

    ax.plot(gmin,yfrac,label='Y',color=set2[0])
    ax.plot(gmin,jfrac,label='J',color=set2[1])
    ax.plot(gmin,hfrac,label='H',color=set2[2])
    ax.plot(gmin,kfrac,label='K',color=set2[3])
    ax.plot(gmin,w1frac,label='W1',color=set2[4])
    ax.plot(gmin,w2frac,label='W2',color=set2[5])
    ax.plot(gmin,w3frac,label='W3',color=set2[6])
    # ax.plot(gmin,w4frac,label='W4')
    
    ax.axvline(22.0,color='grey')
    
    plt.legend(loc='lower left', prop={'size':10})
    
    ax2 = ax.twinx()
    ax2.plot(gmin,totalnum,color='black',linestyle='--')
    ax2.set_ylim(0,170000)
    
    ax.set_xlabel(r'$g_{\rm min}$')
    ax.set_ylabel('Completeness')
    ax2.set_ylabel('Number of Objects')

    plt.tight_layout()
    
    plt.savefig('/home/lc585/thesis/figures/chapter06/dr10completeness.pdf')

    plt.show() 

    return None 


def red_spectra():


    set_plot_properties() # change style 
    
    from SpectraTools.get_spectra import read_boss_dr12_spec

    fig, ax = plt.subplots(figsize=figsize(0.7))

    wav, dw, flux, err = read_boss_dr12_spec('spec-4240-55455-0626.fits')
    ax.plot(wav, flux, color=cs[-1], label='SDSSJ0240+0103')

    wav, dw, flux, err = read_boss_dr12_spec('spec-5481-55983-0346.fits')
    ax.plot(wav, flux + 10, color=cs[0], label='SDSSJ1500+0826')

    plt.legend(frameon=False)

    ax.set_xlim(4000, 9000)
    ax.set_ylim(-5, 20)

    ax.set_xlabel(r'Wavelength $\lambda$ [\AA]')
    ax.set_ylabel('Flux F$_{\lambda}$ [Arbitary Units]')

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter06/red_spectra.pdf')

    plt.show() 


    return None 

def ebv_and_elscal_hist(): 

    tab = Table.read('/data/lc585/QSOSED/Results/Red_Obj_Cat/RedObjCatExt_MCMC.fits')
    
    fig = plt.figure(figsize=(8,3))
    
    ax = fig.add_subplot(1,2,1)
    ax.hist(tab['EBV_FIT'],histtype='step',color='black',bins=np.arange(0,0.5,0.05),log=True)
    plt.tick_params(axis='both',which='major',labelsize=10)
    ax.set_xlabel('E(B-V)',fontsize=12)
    ax.set_ylabel('Number of Quasars')
    plt.tight_layout()
    
    ax = fig.add_subplot(1,2,2)
    ax.hist(tab['ELSCAL_FIT'],histtype='step',color='black',bins=np.arange(0,10,1),log=True)
    ax.set_xlabel('Emission Line Scaling',fontsize=12)
    plt.tick_params(axis='both',which='major',labelsize=10)
    
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/ebv_and_elscal_hist.pdf')

    plt.show() 

    return None 

def zhist_elscal():

    tab = Table.read('/data/lc585/QSOSED/Results/Red_Obj_Cat/RedObjCatExt_MCMC.fits')
       
    fig = plt.figure(figsize=figsize(0.7))
    ax = fig.add_subplot(1,1,1)
    
    npbins = np.arange(2,4.1,0.1)
    
    bins, edges = np.histogram( tab['Z'][ tab['ELSCAL_FIT'] < 0.01 ], bins=npbins)
    bins_all, edges_all = np.histogram( tab['Z'], bins=npbins)
    
    bins = np.array(bins,dtype='float') / np.array(bins_all,dtype='float')
    
    left,right = edges[:-1],edges[1:]
    
    X = np.array([left,right]).T.flatten()
    Y = np.array([bins,bins]).T.flatten()
    
    ax.plot(X,Y,color='black')
  
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('Fraction of weak emission line objects')
    ax.set_xlim(2,4)
    ax.set_ylim(0,0.7)
    plt.tick_params(axis='both',which='major')

    plt.tight_layout()
    plt.savefig('/home/lc585/thesis/figures/chapter06/zhist_elscal.pdf') 
    
    plt.show() 

    return None 

def posteriors():

    import pymc
    import sys
    sys.path.insert(0,'/home/lc585/Dropbox/IoA/QSOSED/Model/MCMC')
    from idlplot import plothist, tvhist2d

    fig = plt.figure(figsize=(9.7,6.2))

    f = '/data/lc585/QSOSED/Results/140324_MCMC_CHAINS/0019.pickle'
    pymcMC = pymc.database.pickle.load(f) 

    # plot the diagram with posterior distributions
    keys = ['ebv','elscal','imod']
    plt.clf() # clear plot
    counter = 0
    for k1 in keys:
        var1 = pymcMC.trace(k1)[:]
        #chain for variable k1
        for k2 in keys:
            var2 = pymcMC.trace(k2)[:]
            #chain for variable k1  
            ax = fig.add_subplot(3, 3, counter+1)

            if k1!=k2:
                tvhist2d(var2,var1,noerase=True,bins=[30,30])
                
            else:
                plothist(var1,noerase=True,nbins=30)

            if counter == 0:
                ax.set_xlabel('E(B-V)')
    
            if counter == 3:
                ax.set_xlabel('E(B-V)')
                ax.set_ylabel('EW Scaling')
    
            if counter == 4:
                ax.set_xlabel('EW Scaling')
    
            if counter == 6:
                ax.set_xlabel('E(B-V)')
                ax.set_ylabel('Normalisation')
    
            if counter == 7:
                ax.set_xlabel('EW Scaling')
                ax.set_ylabel('Normalisation')
    
            
            if counter == 8:
                ax.set_xlabel('Normalisation')           
            
            
            if (counter == 1) | (counter == 2) | (counter == 5): 
                plt.delaxes(ax)

            counter +=1

    fig.tight_layout() 

    plt.savefig('/home/lc585/thesis/figures/chapter06/posteriors.pdf')

    return None 


def ratio_tbb_beta(density=False):

    from astropy.table import join

    tab = Table.read('/data/lc585/QSOSED/Results/141203/sample1/out_add.fits')
    
    tab = tab[ tab['BBT_STDERR'] < 200.0 ]
    tab = tab[ tab['BBFLXNRM_STDERR'] < 0.05]
    tab = tab[ tab['CHI2_RED'] < 3.0]
    
    tab1 = Table.read('/data/lc585/QSOSED/Results/141124/sample2/out.fits') # 9000 - 23500 fit
    tab2 = Table.read('/data/lc585/QSOSED/Results/141124/sample1/out.fits') # 10000 - 23500 fit
    
    tab1['BBPLSLP'] = tab1['BBPLSLP'] - 1.0
    tab2['BBPLSLP'] = tab2['BBPLSLP'] - 1.0
    
    goodnames = Table()
    goodnames['NAME'] = tab['NAME']
    
    tab1 = join( tab1, goodnames, join_type='right', keys='NAME')
    tab2 = join( tab2, goodnames, join_type='right', keys='NAME')
    
    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.8))

    if not density: 

        im = ax.hexbin(tab['BBT'],
                       tab['RATIO_IR_UV'],
                       C = tab1['BBPLSLP'],
                       gridsize=(80,80))
        
        cb = plt.colorbar(im)
        cb.set_label(r'$\beta_{\rm NIR}$')
        
        # Digitize beta
        nbins = 20
        bins = np.linspace(-1.5,1.5,nbins+1)
        ind = np.digitize(tab1['BBPLSLP'],bins)
        bbt_locus = [np.median(tab['BBT'][ind == j]) for j in range(1, nbins)]
        ratio_locus = [np.median(tab['RATIO_IR_UV'][ind == j]) for j in range(1, nbins)]
        ax.plot(bbt_locus[7:-2],ratio_locus[7:-2],color='black',linewidth=2.0)
    
    
        ax.set_ylabel(r'$R_{{\rm NIR}/{\rm UV}}$')
        ax.set_xlabel(r'$T_{\rm BB}$')
    
        ax.set_ylim(0,2)
        ax.set_xlim(700,1900)
    
        fig.tight_layout()

        fig.savefig('/home/lc585/thesis/figures/chapter06/ratio_tbb_beta.pdf')
        
    else:

        from matplotlib import cm
        from PlottingTools.truncate_colormap import truncate_colormap
        mycm = cm.get_cmap('YlOrRd_r')
        mycm.set_under('w')
        cset = brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colors
        mycm = truncate_colormap(mycm, 0.0, 0.8)

        im = ax.hexbin(tab['BBT'],
                       tab['RATIO_IR_UV'],
                       gridsize=(80,80),
                       cmap = mycm,
                       mincnt=1)
        
        
        cb = plt.colorbar(im)
        cb.set_label(r'Number of Objects')
        
        ax.set_ylabel(r'$R_{{\rm NIR}/{\rm UV}}$')
        ax.set_xlabel(r'$T_{\rm BB}$')
    
        ax.set_ylim(0,2)
        ax.set_xlim(700,1900)
    
        fig.tight_layout()

        fig.savefig('/home/lc585/thesis/figures/chapter06/ratio_tbb_beta_density.pdf')

    plt.show() 

    return None 




def ratio_tbb_contours(): 


    tab = Table.read('/data/lc585/QSOSED/Results/150309/sample4/out_add.fits')
    
    tab = tab[ ~np.isnan(tab['BBT_STDERR'])]
    tab = tab[ tab['BBT_STDERR'] < 500. ]
    tab = tab[ tab['BBT_STDERR'] > 5.0 ]
    tab = tab[ (tab['LUM_IR_SIGMA']*tab['RATIO_IR_UV']) < 1.]
    
    tab = tab[ (tab['RATIO_IR_UV'] < 2.0) & (tab['RATIO_IR_UV'] > 0.0)]
    tab = tab[ (tab['BBT'] > 600.0) & (tab['BBT'] < 2000.0)]
    
    m1, m2 = tab['BBT'], tab['RATIO_IR_UV']
    
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    # Sample from single (T,Norm) with gaussian errors on photometry. 
    # Mock magnitude file made in model.py and then fit in runsingleobjfit.py.
    
    tab2 = Table.read('/data/lc585/QSOSED/Results/150309/sample5/out_add.fits')
    
    fig, ax = plt.subplots(figsize=figsize(1.0))
    
    CS = ax.contour(X,Y,Z, colors='grey', levels=[0.0015,0.003,0.0045,0.006,0.0075,0.009,0.0105])
    
    ax.scatter(tab2['BBT'],
               tab2['RATIO_IR_UV'],
               edgecolor='None',
               color='black', 
               s=8)
    
    ax.set_ylim(0,0.8)
    ax.set_xlim(600,2000)
    
    ax.set_xlabel(r'$T_{BB}$')
    ax.set_ylabel('$R_{NIR/UV}$')

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter06/ratio_tbb_contours.pdf')
    
    plt.show() 

    return None 


def civ_hot_dust_beta():

    tab = Table.read('/data/lc585/QSOSED/Results/140905/fit2/out_add.fits')
    
    
    tab = tab[ tab['W3SNR'] > 3.0]
    
    tab = tab[ tab['BAL_FLAG'] == 0]
    tab = tab[ tab['CIV_BLUESHIFT_PAUL'] < 10000.0]
    tab = tab[ ~np.isnan(tab['CIV_EW_PAUL'])]
    tab = tab[ tab['CIV_EW_PAUL'] > 10**1.2]
    
    
    xdat = tab['CIV_BLUESHIFT_PAUL']
    ydat = np.log10(tab['CIV_EW_PAUL'])
    C = tab['BBPLSLP']
    
    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.9))
    
    from LiamUtils import colormaps as cmaps
    plt.register_cmap(name='inferno_r', cmap=cmaps.inferno_r)
    plt.set_cmap(cmaps.inferno_r)

    im = plt.hexbin(xdat,
                    ydat,
                    C=C,
                    gridsize=(50, 15),
                    mincnt=3,
                    reduce_C_function=np.median,
                    vmin=0.1,
                    vmax=0.7,
                    cmap='RdBu_r',
                    edgecolor='black',
                    linewidth=0.5)
    
    cb = fig.colorbar(im,ax=ax)
    cb.set_label(r'Hot Dust Abundance')
    cb.set_ticks(np.linspace(0.1, 0.7, 5))
    cb.set_ticklabels(np.linspace(0, 1, 5))

    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'Log C\,{\sc iv} EQW [\AA]')
    
    plt.xlim(-1000,4000)
    plt.ylim(1, 2.2)
    plt.tick_params(axis='both',which='major')
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/hot_dust_beta.pdf')
    plt.show()

    return None 

def civ_hot_dust_ratio():

    tab = Table.read('/data/lc585/QSOSED/Results/150211/sample2/out_add.fits')
    
    tab = tab[ ~np.isnan(tab['BBT_STDERR'])]
    tab = tab[ tab['BBT_STDERR'] < 500. ]
    tab = tab[ tab['BBT_STDERR'] > 5.0 ]
    tab = tab[ (tab['LUM_IR_SIGMA']*tab['RATIO_IR_UV']) < 0.4] 
    
    civtab = Table.read('/data/lc585/QSOSED/Results/140827/civtab.fits')
    
    newtab = Table()
        
    newtab['NAME'] = civtab['NAME']
    newtab['CIV_BLUESHIFT_PAUL'] = civtab['BLUESHIFT']
    newtab['CIV_EW_PAUL'] = civtab['EW']
    
    tab = join( tab, newtab, keys='NAME', join_type= 'left')
    
    xdat = tab['CIV_BLUESHIFT_PAUL']
    ydat = np.log10(tab['CIV_EW_PAUL'])
    C = tab['RATIO_IR_UV']
    
    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.9))

    im = ax.hexbin(xdat,
                   ydat,
                   C=C,
                   gridsize=(55,35),
                   mincnt=2,
                   reduce_C_function=np.median,
                   cmap='jet',
                   vmax=1.2,
                   vmin=0.4)
    
    cb = fig.colorbar(im,ax=ax)
    cb.set_label(r'$R_{NIR/UV}$')
    
    ax.set_xlabel(r'C$\,$IV Blueshift (km/s)')
    ax.set_ylabel(r'Log$_{10}$(C$\,$IV REW ($\AA$))')
    
    ax.set_xlim(-500,3000)
    ax.set_ylim(1.1,1.9)
    
    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/hot_dust_ratio.pdf')
    plt.show()

    return None 

def shang_sed():

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.constants import c
    
    rltab = np.genfromtxt('/home/lc585/thesis/code/chapter01/rlmsedMR.txt')
    
    fig = plt.figure(figsize=figsize(1, 0.8))
    ax = fig.add_subplot(111)
    
    ax.plot( 1.0e10 * c / (10**rltab[:,0]), 10**rltab[:,1], color='black', label='Radio-loud' )
       
    ax.set_xlabel(r'Wavelength [${\rm \AA}$]')
    ax.set_ylabel( r'log ${\lambda}f_{\lambda}$ [Arbitrary Units]')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_ylim(1e-2,12)
    ax.set_xlim(10,10**8)

    
    plt.tight_layout()
    
    plt.savefig('/home/lc585/thesis/figures/chapter05/shangsed.pdf')

    plt.show() 
    
    return None 

def lum_z():

    fig, ax = plt.subplots(figsize=figsize(1, 0.8)) 

    # Created by DefiningSample.ipynb 
    t = Table.read('/data/lc585/SDSS/matched_catalogue.fits')

    t = t[t['LOGLBOL'] > 20]
    
    ax.plot(t['Z_HEWETT'], 
            t['LOGLBOL'], 
            linestyle='', 
            marker='o',
            markersize=1,
            markerfacecolor=cs[1],
            markeredgecolor='None')

    ax.set_ylim(44.5, None)

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Log L$_{\rm Bol}$ [erg~$\rm{s}^{-1}$]')

    plt.tight_layout()
    
    plt.savefig('/home/lc585/thesis/figures/chapter05/lum_z.pdf')

    plt.show() 