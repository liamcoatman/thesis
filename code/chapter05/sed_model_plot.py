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

set_plot_properties() # change style 
cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

def plot():
        
    plslp1 = 0.508
    plslp2 = 0.068
    plbrk = 2944.99
    bbt = 1174.0
    bbflxnrm = 0.208
    galfra = 0.313
    elscal = 0.624
    imod = 18.0 
    ebv = 0.0 
    scahal = 0.8
    
    with open('input.yml', 'r') as f: parfile = yaml.load(f)
    
    # Load stuff
    fittingobj = load(parfile)

    wavlen = fittingobj.get_wavlen()
    flxcorr = np.array( [1.0] * len(wavlen) ) 


    magtmp, wavlentmp, fluxtmp = model(plslp1,
                                       plslp2,
                                       plbrk,
                                       bbt,
                                       bbflxnrm,
                                       elscal,
                                       scahal,
                                       galfra,
                                       ebv,
                                       imod,
                                       1.0,
                                       fittingobj,
                                       flxcorr,
                                       parfile)
    
    

    fig, ax = plt.subplots(figsize=figsize(0.8, vscale=0.9))
        
    ax.plot(wavlen,wavlen*fluxtmp,color='black')

    import lineid_plot

    line_wave = [1216,
                 1549,
                 1909,
                 2798,
                 4861,
                 6563,
                 18744]

    line_names = [r'Ly$\alpha$', 
                  r'C\,{\sc iv}', 
                  r'C\,{\sc iii}]', 
                  r'Mg\,{\sc ii}', 
                  r'H$\beta$', 
                  r'H$\alpha$', 
                  r'Pa$\alpha$']

    lineid_plot.plot_line_ids(wavlen, wavlen*fluxtmp, line_wave, line_names, ax=ax, arrow_tip=25000)

    numwavbin = len(wavlen)
    flux = np.zeros(numwavbin) 
    plflxnrm = parfile['quasar']['pl']['flxnrm']
    plwavnrm = parfile['quasar']['pl']['wavnrm']
    const2 = plflxnrm / (plwavnrm**(plslp2-2.0))
    const1 = const2 * ( (plbrk**(plslp2-2.0)) / (plbrk**(plslp1-2.0)) )
    i = 0
    while wavlen[i] < plbrk: i += 1
    wavnumbrk = i
    flux[:wavnumbrk] = flux[:wavnumbrk] + pl(wavlen[:wavnumbrk],plslp1,const1)
    ax.plot(wavlen[:wavnumbrk],wavlen[:wavnumbrk]*flux[:wavnumbrk],color=cs[1],label='Blue Power-Law')
    flux[wavnumbrk:] = flux[wavnumbrk:] + pl(wavlen[wavnumbrk:],plslp2,const2)
    ax.plot(wavlen[wavnumbrk:],wavlen[wavnumbrk:]*flux[wavnumbrk:],color=cs[0],label='Red Power-Law')
        
    bbwavnrm = parfile['quasar']['bb']['wavnrm']
    flux = bb(wavlen,bbt,bbflxnrm,bbwavnrm)
    ax.plot(wavlen,wavlen*flux,color=cs[3],label='Hot Blackbody')

    flux = np.zeros(numwavbin) 
    plflxnrm = parfile['quasar']['pl']['flxnrm']
    plwavnrm = parfile['quasar']['pl']['wavnrm']
    const2 = plflxnrm / (plwavnrm**(plslp2-2.0))
    const1 = const2 * ( (plbrk**(plslp2-2.0)) / (plbrk**(plslp1-2.0)) )
    i = 0
    while wavlen[i] < plbrk: i += 1
    wavnumbrk = i
    flux[:wavnumbrk] = flux[:wavnumbrk] + pl(wavlen[:wavnumbrk],plslp1,const1)
    flux[wavnumbrk:] = flux[wavnumbrk:] + pl(wavlen[wavnumbrk:],plslp2,const2)
    bbwavnrm = parfile['quasar']['bb']['wavnrm']
    flux = flux + bb(wavlen,bbt,bbflxnrm,bbwavnrm)
    
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax() 
    galcnt = fittingobj.get_galcnt()
    galspc = fittingobj.get_galspc()
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)


    qsocnt = np.sum(flux[ignmin:ignmax]) 

    lumdist_nrm = cd.luminosity_distance(parfile['gal']['znrm'],**cosmo) * (10.0**6)
    lumdist_qso = cd.luminosity_distance(1.0, **cosmo) * (10.0**6)

    vallum = 10**( np.log10( (lumdist_qso / lumdist_nrm)**2 ) )                
            
    cscale = ( qsocnt / galcnt )   

    galplind = parfile['gal']['plind']

    scaval = vallum**(galplind - 1.0 )
    scagal = (galfra / (1.0 - galfra) ) * scaval 
            
    flux = cscale * scagal * galspc
    ax.plot(wavlen,wavlen*flux,color=cs[2],label='Host Galaxy')

    ax.set_xlim(800,30000)
    ax.set_ylim(0,25000)
    ax.set_ylabel(r'${\lambda}F_{\lambda}$ (Arbitary Units)')
    ax.set_xlabel(r'Wavelength $\lambda$ (${\rm \AA}$)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.savefig('/home/lc585/thesis/figures/chapter06/sed_model.pdf')

    plt.show() 

    return None 
