# Return SED for given set of parameters 


def plot():

    import yaml 
    from qsosed.load import load
    import numpy as np
    from qsosed.sedmodel import model 
    import matplotlib.pyplot as plt 
    from qsosed.pl import pl 
    from qsosed.bb import bb
    import cosmolopy.distance as cd
    from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
    import palettable 
    from lmfit import Parameters
    from qsosed.qsrmod import qsrmod
    from qsosed.flx2mag import flx2mag

    set_plot_properties() # change style 
    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
  

    plslp1 = 0.508
    plslp2 = 0.068
    plbrk = 2944.99
    bbt = 1174.0
    bbflxnrm = 0.208
    galfra = 0.313
    elscal = 0.624
    imod = 18.0 
    ebv = 0.0 
    redshift = 2.0
    scahal = 0.8 
    
    with open('/home/lc585/qsosed/input.yml', 'r') as f:
        parfile = yaml.load(f)
    
    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()
    flxcorr = np.array([1.0] * len(wavlen))
     
    params = Parameters()
    params.add('plslp1', value = plslp1)
    params.add('plslp2', value = plslp2)
    params.add('plbrk', value = plbrk)
    params.add('bbt', value = bbt)
    params.add('bbflxnrm', value = 2.0)
    params.add('elscal', value = elscal)
    params.add('galfra', value = galfra)
    params.add('ebv', value = ebv)
    params.add('imod', value = imod)
    params.add('scahal', value = scahal)


    lin = fittingobj.get_lin()
    galspc = fittingobj.get_galspc()
    ext = fittingobj.get_ext()
    galcnt = fittingobj.get_galcnt()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    bp = fittingobj.get_bp() 
    dlam = fittingobj.get_dlam()
    zromag = fittingobj.get_zromag()
    ftrlst = fittingobj.get_ftrlst()
    ztran = fittingobj.get_ztran()
    lyatmp = fittingobj.get_lyatmp()
    lybtmp = fittingobj.get_lybtmp()
    lyctmp = fittingobj.get_lyctmp()
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    qsomag = fittingobj.get_qsomag()
    cosmo = fittingobj.get_cosmo()

    nftr = len(bp)
    
    redshift = 0.5

    wavlentmp, fluxtmp = qsrmod(params,
                                parfile,
                                wavlen,
                                redshift,
                                lin,
                                galspc,
                                ext,
                                galcnt,
                                ignmin,
                                ignmax,
                                ztran,
                                lyatmp,
                                lybtmp,
                                lyctmp,
                                whmin,
                                whmax,
                                cosmo,
                                flxcorr,
                                qsomag)
           
    magtmp = flx2mag(params,wavlentmp,fluxtmp,bp,dlam,zromag,ftrlst)

    fig = plt.figure(figsize=figsize(1, vscale=0.8))
    ax1 = fig.add_subplot(111)
    ax1.loglog(wavlentmp,0.5*fluxtmp,color='black',label=r'$z=0.5$')

    redshift = 2.0
    wavlentmp, fluxtmp = qsrmod(params,
                                parfile,
                                wavlen,
                                redshift,
                                lin,
                                galspc,
                                ext,
                                galcnt,
                                ignmin,
                                ignmax,
                                ztran,
                                lyatmp,
                                lybtmp,
                                lyctmp,
                                whmin,
                                whmax,
                                cosmo,
                                flxcorr,
                                qsomag)
           
    magtmp = flx2mag(params,wavlentmp,fluxtmp,bp,dlam,zromag,ftrlst)
    ax1.loglog(wavlentmp,5*fluxtmp,color='black',label=r'$z=2.0$')
    
    redshift = 3.5
    wavlentmp, fluxtmp = qsrmod(params,
                                parfile,
                                wavlen,
                                redshift,
                                lin,
                                galspc,
                                ext,
                                galcnt,
                                ignmin,
                                ignmax,
                                ztran,
                                lyatmp,
                                lybtmp,
                                lyctmp,
                                whmin,
                                whmax,
                                cosmo,
                                flxcorr,
                                qsomag)
           
    magtmp = flx2mag(params,wavlentmp,fluxtmp,bp,dlam,zromag,ftrlst)
    ax1.loglog(wavlentmp,50*fluxtmp,color='black',label=r'$z=3.5$')

    ax1.set_xlabel(r'log $\lambda$ [${\rm \AA}$]')
    ax1.set_ylabel(r'log $F_{\lambda}$ [Arbitary Units]')
    # ax1.loglog(wavlentmp,1.e10/wavlentmp**2,linestyle='--')
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    labs = ['u','g','r','i','z','Y','J','H','K','W1','W2','W3']
    colormap = plt.cm.jet 
    xpos = fittingobj.get_lameff()
    xpos[:5] = xpos[:5] - 200.0
    xpos[5] = 10405
    xpos[6] = 12505
    xpos[7] = 16411
    xpos[8] = 21942
    xpos[9] = 33500
    xpos[10] = 46027
    xpos[11] = 112684

    ax2.text(19225,3.923,r'$z=3.5$',ha='right')
    ax2.text(11674,3.099,r'$z=2.0$',ha='right')
    ax2.text(6735,2.135,r'$z=0.5$',ha='right')

    
    color_idx = np.linspace(0, 1, 12)
    
    from palettable.colorbrewer.diverging import Spectral_11

    for i in range(len(bp[:-1])):
        wavtmp = ( bp[i][0,:] )  
        flxtmp = bp[i][1,:] / np.max(bp[i][1,:])
        ax2.plot(wavtmp,flxtmp,color=Spectral_11.mpl_colormap(color_idx[i]))
        ax2.fill_between(wavtmp, flxtmp, alpha=0.2, facecolor=Spectral_11.mpl_colormap(color_idx[i]))
        ax2.text(xpos[i],0.2,r'${}$'.format(labs[i]), ha='center')
    
    ax2.set_ylim(0,5)
    ax1.set_ylim(1e-3,200)
    ax1.set_xlim(2800,190000)
    ax2.set_xlim(ax1.get_xlim())
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter05/throughput.pdf')
    plt.show() 

    return None 

