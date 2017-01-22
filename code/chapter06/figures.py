import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import running 

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



def posterior_plot():

    import matplotlib.pyplot as plt 
    import pymc
    import sys
    sys.path.insert(0,'/home/lc585/Dropbox/IoA/QSOSED/Model/MCMC')
    import sedfit_pymc_v2 as fit
    from idlplot import plot,ploterror,oplot,tvhist2d,plothist

    f = '/data/lc585/QSOSED/Results/140324_MCMC_CHAINS/0019.pickle'
    pymcMC = pymc.database.pickle.load(f) 
    plot_posteriors(pymcMC)

    keys = ['ebv','elscal','imod']
    
        fig = plt.figure(figsize=(9.7,6.2))
                    
        counter = 0
    for k1 in keys:
        var1 = pymcMC.trace(k1)[:]
        #chain for variable k1
                
        for k2 in keys:
            var2 = pymcMC.trace(k2)[:]
            #chain for variable k1  
            
                        ax = fig.add_subplot(3,3,counter+1)
                        if k1 != k2:
                            tvhist2d(var2,var1,noerase=True,bins=[30,30])
                
            else:
                            plothist(var1,noerase=True,nbins=30)
            
                        if counter == 0:
                            ax.set_xlabel('E(B-V)',fontsize=10)
                            ax.set_ylabel('#',fontsize=10)

                        if counter == 3:
                            ax.set_xlabel('E(B-V)',fontsize=10)
                            ax.set_ylabel('EW Scaling',fontsize=10)

                        if counter == 4:
                            ax.set_xlabel('EW Scaling',fontsize=10)
                            ax.set_ylabel('#',fontsize=10)

                        if counter == 6:
                            ax.set_xlabel('E(B-V)',fontsize=10)
                            ax.set_ylabel('Normalisation',fontsize=10)

                        if counter == 7:
                            ax.set_xlabel('EW Scaling',fontsize=10)
                            ax.set_ylabel('Normalisation',fontsize=10)

                        
                        if counter == 8:
                            ax.set_xlabel('Normalisation',fontsize=10)
                            ax.set_ylabel('#',fontsize=10)                    
            
                         
                        if (counter == 1) | (counter == 2) | (counter == 5): 
                            plt.delaxes(ax)
                        
                      
                        counter +=1
                    
                   

                        plt.tick_params(axis='both',which='major',labelsize=8) 

      
        plt.tight_layout()
    plt.savefig('test.pdf') 


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

