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
from qsosed.qsrmod import wav2num
import astropy.units as u
from qsosed.bc import bc 
import astropy.constants as const
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d

set_plot_properties() # change style 
cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

def plot():

    redshift = 1
   
    with open('/home/lc585/qsosed/input.yml', 'r') as f: 
        parfile = yaml.load(f)
    
    # Load stuff
    fittingobj = load(parfile)
    lin = fittingobj.get_lin()
    qsomag = fittingobj.get_qsomag()  
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    galcnt = fittingobj.get_galcnt()
    galspc = fittingobj.get_galspc()

    wavlen = fittingobj.get_wavlen()
    flxcorr = np.array( [1.0] * len(wavlen) ) 


    magtmp, wavlentmp, fluxtmp = model(redshift=redshift,
                                       parfile=parfile)
    
    

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.9))
        
    # ax.plot(wavlen,wavlen*fluxtmp,color='black')

    # import lineid_plot

    # line_wave = [1216,
    #              1549,
    #              1909,
    #              2798,
    #              4861,
    #              6563,
    #              18744]

    # line_names = [r'Ly$\alpha$', 
    #               r'C\,{\sc iv}', 
    #               r'C\,{\sc iii}]', 
    #               r'Mg\,{\sc ii}', 
    #               r'H$\beta$', 
    #               r'H$\alpha$', 
    #               r'Pa$\alpha$']

    # lineid_plot.plot_line_ids(wavlen, wavlen*fluxtmp, line_wave, line_names, ax=ax, arrow_tip=10000)

    plslp1 = parfile['quasar']['pl']['slp1']
    plslp2 = parfile['quasar']['pl']['slp2']
    plbrk = parfile['quasar']['pl']['brk']
    bbt = parfile['quasar']['bb']['t']
    bbflxnrm = parfile['quasar']['bb']['flxnrm']
    elscal = parfile['quasar']['el']['scal']
    galfra = parfile['gal']['fra']
    ebv = parfile['ext']['EBV']
    scahal = parfile['quasar']['el']['scahal']

    flux = np.zeros(len(wavlen), dtype=np.float) 

    flxnrm = parfile['quasar']['flxnrm']
    wavnrm = parfile['quasar']['wavnrm']

    # Define normalisation constant to ensure continuity at wavbrk 
    const2 = flxnrm / (wavnrm**(-plslp2)) 
    const1 = const2 * ((plbrk**(-plslp2)) / (plbrk**(-plslp1)))

    wavnumbrk = wav2num(wavlen, plbrk) 

    flux[:wavnumbrk] = flux[:wavnumbrk] + pl(wavlen[:wavnumbrk], plslp1, const1)
    flux[wavnumbrk:] = flux[wavnumbrk:] + pl(wavlen[wavnumbrk:], plslp2, const2)

    # Now add steeper power-law component for sub-Lyman-alpha wavelengths

    # Define normalisation constant to ensure continuity at wavbrk
    plbrk_tmp = parfile['quasar']['pl_steep']['brk']
    plslp_tmp = plslp1 + parfile['quasar']['pl_steep']['step']
    plbrknum_tmp = wav2num(wavlen, plbrk_tmp)

    const_tmp = flux[plbrknum_tmp] / (plbrk_tmp**-plslp_tmp)

    flux[:plbrknum_tmp] = pl(wavlen[:plbrknum_tmp], plslp_tmp, const_tmp)

    flux_pl = flux * 1.0

    # Hot blackbody ---------------------------------------------------

    bbwavnrm = parfile['quasar']['bb']['wavnrm']

    flux = flux + bb(wavlen*u.AA,
                     bbt*u.K,
                     bbflxnrm,
                     bbwavnrm*u.AA,
                     units='freq')

    flux_bb = bb(wavlen*u.AA,
                 bbt*u.K,
                 bbflxnrm,
                 bbwavnrm*u.AA,
                 units='freq')


    # Balmer continuum --------------------------------------------------

    # Add Balmer Continuum and blur to simulate effect of bulk-velocity
    # shifts comparable to those present in emission lines

    # Determine height of power-law continuum at wavelength wbcnrm to allow
    # correct scaling of Balmer continuum contribution

    wbcnrm = parfile['quasar']['balmercont']['wbcnrm']
    wbedge = parfile['quasar']['balmercont']['wbedge']
    bcnrm = parfile['quasar']['balmercont']['bcnrm']
    tbc = parfile['quasar']['balmercont']['tbc']
    taube = parfile['quasar']['balmercont']['taube']
    vfwhm = parfile['quasar']['balmercont']['vfwhm']
   
    # mean wavelength increment
    winc = np.diff(wavlen).mean()

    cfact = flux[wav2num(wavlen, wbcnrm)]
    flux = flux / cfact 

    flux_bc = bc(wavlen=wavlen*u.AA,
                 tbb=tbc*u.K,
                 fnorm=bcnrm,
                 taube=taube,
                 wavbe=wbedge*u.AA,
                 wnorm=wbcnrm*u.AA)

    vsigma = vfwhm*(u.km/u.s) / 2.35
    wsigma = wbedge*u.AA * vsigma / const.c
    wsigma = wsigma.to(u.AA)
    psigma = wsigma / (winc*u.AA)

    # Performs a simple Gaussian smooth with dispersion psigma pixels 
    gauss = Gaussian1DKernel(stddev=psigma)
    flux_bc = convolve(flux_bc, gauss)

    flux = flux + flux_bc



    #-----------------------------------------------------------------

    # Now convert to flux per unit wavelength
    # Presumably the emission line spectrum and galaxy spectrum 
    # are already in flux per unit wavelength. 

    # c / lambda^2 conversion 
    flux = flux*(u.erg / u.s / u.cm**2 / u.Hz)
    flux = flux.to(u.erg / u.s / u.cm**2 / u.AA, 
                   equivalencies=u.spectral_density(wavlen * u.AA))
    scale = flux[wav2num(wavlen, wavnrm)] 
    flux = flxnrm * flux / scale 

    flux_pl = flux_pl*(u.erg / u.s / u.cm**2 / u.Hz)
    flux_pl = flux_pl.to(u.erg / u.s / u.cm**2 / u.AA, 
                         equivalencies=u.spectral_density(wavlen * u.AA))
    flux_pl = flxnrm * flux_pl / scale 

    flux_bb = flux_bb*(u.erg / u.s / u.cm**2 / u.Hz)
    flux_bb = flux_bb.to(u.erg / u.s / u.cm**2 / u.AA, 
                         equivalencies=u.spectral_density(wavlen * u.AA))
    flux_bb = flxnrm * flux_bb / scale 

    flux_bc = flux_bc*(u.erg / u.s / u.cm**2 / u.Hz)
    flux_bc = flux_bc.to(u.erg / u.s / u.cm**2 / u.AA, 
                         equivalencies=u.spectral_density(wavlen * u.AA))
    flux_bc = flxnrm * flux_bc / scale 




    # Emission lines -------------------------------------------------

    linwav, linval, conval = lin[:,0], lin[:,1], lin[:,2]

    # Normalise such that continuum flux at wavnrm equal to that
    # of the reference continuum at wavnrm
    inorm = wav2num(wavlen, wavnrm)
    scale = flux[inorm]
    flux = conval[inorm] * flux / scale 
    flux_pl = conval[inorm] * flux_pl / scale 
    flux_bb = conval[inorm] * flux_bb / scale 
    flux_bc = conval[inorm] * flux_bc / scale 





    # Calculate Baldwin Effect Scaling for Halpha
    zbenrm = parfile['quasar']['el']['zbenrm']
    beslp = parfile['quasar']['el']['beslp']

    # Line added to stop enormous BE evolution at low z 
    zval = np.max([redshift, zbenrm]) 

    # I think this is the absolute magnitude of the SDSS sample as 
    # a function of redshift, which is not the same as how the 
    # absolute magnitude of a object of a given flux changes 
    # as a function of redshift 
    qsomag_itp = interp1d(qsomag[:, 0], qsomag[:, 1])

    # Absolute magnitude at redshift z minus
    # normalisation absolute magnitude
    vallum = qsomag_itp(zval) - qsomag_itp(zbenrm) 

    # Convert to luminosity 
    vallum = 10.0**(-0.4*vallum)

    scabe = vallum**(-beslp) 
     
    flux_el = np.zeros_like(flux)
    flux_el[:whmin] = linval[:whmin] * np.abs(elscal) * flux[:whmin] / conval[:whmin]
    flux_el[whmax:] = linval[whmax:] * np.abs(elscal) * flux[whmax:] / conval[whmax:]

     
    flux[:whmin] = flux[:whmin] + linval[:whmin] * np.abs(elscal) * flux[:whmin] / conval[:whmin]
    flux[whmax:] = flux[whmax:] + linval[whmax:] * np.abs(elscal) * flux[whmax:] / conval[whmax:]

    # Scaling for Ha with Baldwin effect 
    scatmp = elscal * scahal / scabe
    flux_el[whmin:whmax] = linval[whmin:whmax] * np.abs(scatmp) * flux[whmin:whmax] / conval[whmin:whmax] 
    flux[whmin:whmax] = flux[whmin:whmax] + \
                        linval[whmin:whmax] * np.abs(scatmp) * flux[whmin:whmax] / conval[whmin:whmax] 





    gznorm = parfile['gal']['znrm']
    gplind = parfile['gal']['plind']

    # Determine fraction of galaxy sed to add to unreddened quasar SED
    qsocnt = np.sum(flux[ignmin:ignmax]) 
    
    # Factor cscale just to bring input galaxy and quasar flux zero-points equal
    cscale = qsocnt / galcnt 
    
    # Find absolute magnitude of quasar at redshift z
    vallum = qsomag_itp(redshift)
    
    # Find absolute magnitude of quasar at redshift gznorm
    galnrm = qsomag_itp(gznorm)
    
    # Subtract supplied normalisation absolute magnitude
    vallum = vallum - galnrm
    
    # Convert to a luminosity 
    vallum = 10.0**(-0.4*vallum)
    
    # Luminosity scaling 
    scaval = vallum**(gplind - 1.0)
    
    scagal = (galfra / (1.0 - galfra)) * scaval

    flux_gal = cscale * scagal * galspc 
    flux = flux + cscale * scagal * galspc 


    ax.plot(wavlen, wavlen*flux, color='black')
    ax.plot(wavlen[:wavnumbrk], wavlen[:wavnumbrk]*flux_pl[:wavnumbrk], color=cs[1], label='Accretion Disc')
    ax.fill_between(wavlen[:wavnumbrk], wavlen[:wavnumbrk]*flux_pl[:wavnumbrk], facecolor=cs[1], alpha=0.2)
    ax.plot(wavlen[wavnumbrk:], wavlen[wavnumbrk:]*flux_pl[wavnumbrk:], color=cs[0], label='Accretion Disc')
    ax.fill_between(wavlen[wavnumbrk:], wavlen[wavnumbrk:]*flux_pl[wavnumbrk:], facecolor=cs[0], alpha=0.2)
    ax.plot(wavlen, wavlen*(flux_bc), color=cs[2], label='Balmer Continuum')
    ax.fill_between(wavlen, wavlen*(flux_bc), facecolor=cs[2], alpha=0.2)
    ax.plot(wavlen, wavlen*(flux_bb), color=cs[4], label='Hot Dust')
    ax.fill_between(wavlen, wavlen*(flux_bb), facecolor=cs[4], alpha=0.2)
    ax.plot(wavlen, wavlen*(flux_gal), color=cs[3], label='Galaxy')
    ax.fill_between(wavlen, wavlen*(flux_gal), facecolor=cs[3], alpha=0.2)

    ax.legend() 

    ax.set_xlim(1216,20000)
    ax.set_ylim(0,10000)
    ax.set_ylabel(r'${\lambda}F_{\lambda}$ (Arbitary Units)')
    ax.set_xlabel(r'Wavelength $\lambda$ (${\rm \AA}$)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.savefig('/home/lc585/thesis/figures/chapter05/sed_model.pdf')

    plt.show() 

    return None 
