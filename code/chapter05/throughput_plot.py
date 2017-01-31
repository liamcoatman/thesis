# Return SED for given set of parameters 


def plot():

    import yaml 
    from qsosed.load import load
    import numpy as np
    from qsosed.SEDModel import model 
    import matplotlib.pyplot as plt 
    from qsosed.pl import pl 
    from qsosed.bb import bb
    import cosmolopy.distance as cd
    from PlottingTools.plot_setup import figsize, set_plot_properties
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
    
    with open('input.yml', 'r') as f:
        parfile = yaml.load(f)
    
    fittingobj = load(parfile)
    wavlen = fittingobj.get_wavlen()
    flxcorr = np.array([1.0] * len(wavlen))
     
    params = Parameters()
    params.add('plslp1', value = plslp1)
    params.add('plslp2', value = plslp2)
    params.add('plbrk', value = plbrk)
    params.add('bbt', value = bbt)
    params.add('bbflxnrm', value = bbflxnrm)
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
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)

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
                                flxcorr)
           
    magtmp = flx2mag(params,wavlentmp,fluxtmp,bp,dlam,zromag,ftrlst)

    fig = plt.figure(figsize=figsize(0.8))
    ax1 = fig.add_subplot(111)
    ax1.loglog(wavlentmp,0.1*fluxtmp,color='black',label=r'$z=0.5$')
    ax1.text(5272,0.3,r'$z=0.5$')

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
                                flxcorr)
           
    magtmp = flx2mag(params,wavlentmp,fluxtmp,bp,dlam,zromag,ftrlst)
    ax1.loglog(wavlentmp,0.5*fluxtmp,color='black',label=r'$z=2.0$')
    ax1.text(10927,1.5,r'$z=2.0$')
    
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
                                flxcorr)
           
    magtmp = flx2mag(params,wavlentmp,fluxtmp,bp,dlam,zromag,ftrlst)
    ax1.loglog(wavlentmp,2.0*fluxtmp,color='black',label=r'$z=3.5$')
    ax1.text(16766,6,r'$z=3.5$')

    ax1.set_xlabel(r'log($\lambda$) (${\rm \AA}$)')
    ax1.set_ylabel(r'log($F_{\lambda}$) (Arbitary Units)')
    ax1.loglog(wavlentmp,1.e10/wavlentmp**2,linestyle='--')
    ax2 = ax1.twinx()
    labs = ['u','g','r','i','z','Y','J','H','K','W1','W2','W3']
    colormap = plt.cm.jet 
    plt.gca().set_color_cycle([colormap(k) for k in np.linspace(0, 1.0, len(fittingobj.get_lameff())-1)])
    xpos = fittingobj.get_lameff()
    xpos[:5] = xpos[:5] - 200.0
    xpos[5] = 9945
    xpos[6] = 11960
    xpos[7] = 15483
    xpos[8] = 21183
    xpos[9] = 30632
    xpos[10] = 43486

    for i in range(len(bp[:-1])):
        wavtmp = ( bp[i][0,:] )  
        flxtmp = bp[i][1,:] / np.max(bp[i][1,:])
        ax2.plot(wavtmp,flxtmp)
        ax2.text(xpos[i],0.2,r'${}$'.format(labs[i]), ha='center')
    
    ax2.set_ylim(0,3)
    ax1.set_ylim(1e-3,100)
    ax1.set_xlim(2800,190000)
    ax2.set_xlim(ax1.get_xlim())
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/throughput.pdf')
    plt.show() 

    return None 

