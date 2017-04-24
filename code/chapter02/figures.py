from astropy.table import Table, join 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import cPickle as pickle
import os
import time
import numpy.ma as ma 
import palettable 
import matplotlib.patches as patches
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
import pandas as pd 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties 
from PlottingTools.kde_contours import kde_contours
import yaml
from lmfit import minimize, Parameters
from functools import partial
from scipy.interpolate import interp1d

set_plot_properties() # change style 

def luminosity_z(): 

   

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    # definitions for the axes
    left, width = 0.13, 0.65
    bottom, height = 0.13, 0.65
    bottom_h = left_h = left + width + 0.0
        
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # no labels
    nullfmt = NullFormatter() 
    
    fig = plt.figure(figsize=figsize(1, vscale=1.0))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    axHistx.set_xticks([])
    axHistx.set_yticks([])
    axHisty.set_xticks([])
    axHisty.set_yticks([])
    
    t = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits')
    t = t[t['LOGLBOL'] > 0.0]
    t = t[t['Z_HW'] > 1.0]

    kde_contours(t['Z_HW'], 
                 t['LOGLBOL'], 
                 axScatter, 
                 {'levels':[0.02, 0.05, 0.13, 0.2, 0.4, 0.8, 1.2]}, 
                 {'label':'SDSS DR7'}, 
                 color=cs[-1], 
                 lims=(1.0, 5.0, 44.0, 48.0))

    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df.drop_duplicates(subset=['ID'], inplace=True)
    df = df[df.SPEC_NIR != 'None']
    df.dropna(subset=['LogL5100'], inplace=True)
    
    axScatter.scatter(df.z_IR, 
                      np.log10(9.26) + df.LogL5100, 
                      c=cs[1], 
                      edgecolor='None', 
                      label='This work', 
                      zorder=10, 
                      s=10)
    
    axScatter.set_xlabel(r'Redshift $z$')
    axScatter.set_ylabel(r'log $L_{\mathrm{Bol}}$ [erg/s]')
    
    legend = axScatter.legend(frameon=True, 
                              scatterpoints=1, 
                              numpoints=1, 
                              loc='lower right') 
    
    axHistx.hist(t['Z_HW'], 
                 bins=np.arange(0.0, 5.0, 0.25), 
                 facecolor=cs[-1], 
                 edgecolor='None', 
                 alpha=0.4, 
                 normed=True)
    
    axHisty.hist(t['LOGLBOL'], 
                 bins=np.arange(44, 49, 0.25), 
                 facecolor=cs[-1], 
                 edgecolor='None', 
                 orientation='horizontal', 
                 alpha=0.4, 
                 normed=True)
    
    axHistx.hist(df.z_IR, 
                 bins=np.arange(1, 5.0, 0.25), 
                 histtype='step', 
                 edgecolor=cs[1], 
                 normed=True, 
                 lw=2)
    
    axHisty.hist(np.log10(9.26) + np.array(df.LogL5100), 
                 bins=np.arange(44, 49, 0.25), 
                 histtype='step', 
                 edgecolor=cs[1], 
                 normed=True, 
                 lw=2, 
                 orientation='horizontal')
    
    axScatter.set_xlim(1.0, 4.5)
    axScatter.set_ylim(44, 48.5)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    axHistx.set_ylim(0, 1.5)
    axHisty.set_xlim(0, 1.2)   

    fig.savefig('/home/lc585/thesis/figures/chapter02/luminosity_z.pdf')

    plt.show() 

    return None  


def normalise_to_sdss():

    import sys
    sys.path.insert(0, '/home/lc585/Dropbox/IoA/nirspec/python_code')
    from get_nir_spec import get_nir_spec
    from get_sdss_spec import get_sdss_spec
    
    sys.path.insert(0, '/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit')
    from qsrmod import qsrmod
    from load import load
    import cosmolopy.distance as cd
    from get_mono_lum import resid_mag_fit


    fig, axs = plt.subplots(2, 1, figsize=figsize(1, 1.4))

    cs = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    cs_light = palettable.colorbrewer.qualitative.Pastel1_6.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    row = df.ix['QSO125']

    wav_nir, dw_nir, flux_nir, err_nir = get_nir_spec(row.NIR_PATH, row.INSTR)      
    
    wav_nir = wav_nir / (1.0 + row.z_IR)

    axs[0].plot(wav_nir, flux_nir*1e15, color=cs_light[0], label='Near-IR')
    

    if (row.SPEC_OPT == 'SDSS') | (row.SPEC_OPT == 'BOSS+SDSS'):
    
        wav_opt, dw_opt, flux_opt, err_opt = get_sdss_spec('SDSS', row.DR7_PATH)
    
    elif (row.SPEC_OPT == 'BOSS') :
    
        wav_opt, dw_opt, flux_opt, err_opt = get_sdss_spec('BOSS', row.DR12_PATH)
  
    wav_opt = wav_opt / (1.0 + row.z_IR)
    
    axs[0].plot(wav_opt, flux_opt*1e15, color=cs_light[1], label='SDSS')

    
    
    
    # Normalise SED model to SDSS spectra ----------------------------------------------------
    
    """
    SDSS spectra in Shen & Liu emission line free windows 
    """
    
    fit_region = [[1350,1360], [1445,1465], [1700,1705], [2155,2400], [2480,2675], [2925,3500], [4200,4230], [4435,4700], [5100,5535], [6000,6250], [6800,7000]]
    
    fit_mask = np.zeros(len(wav_opt), dtype=bool)
    
    for r in fit_region:
         fit_mask[(wav_opt > r[0]) & (wav_opt < r[1])] = True
    
    tmp = ma.array(flux_opt)
    tmp[~fit_mask] = ma.masked 

    for item in ma.extras.flatnotmasked_contiguous(tmp):
        axs[0].plot(wav_opt[item], flux_opt[item]*1e15, color=cs[1])
    
    # ax.plot(wav_opt[fit_mask], flux_opt[fit_mask], color=cs[0])

    plslp1 = 0.46
    plslp2 = 0.03
    plbrk = 2822.0
    bbt = 1216.0
    bbflxnrm = 0.24
    elscal = 0.71
    scahal = 0.86
    galfra = 0.31
    ebv = 0.0
    imod = 18.0
    
    with open('/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit/input.yml', 'r') as f:
        parfile = yaml.load(f)
    
    fittingobj = load(parfile)
    
    lin = fittingobj.get_lin()
    galspc = fittingobj.get_galspc()
    ext = fittingobj.get_ext()
    galcnt = fittingobj.get_galcnt()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    wavlen_rest = fittingobj.get_wavlen()
    ztran = fittingobj.get_ztran()
    lyatmp = fittingobj.get_lyatmp()
    lybtmp = fittingobj.get_lybtmp()
    lyctmp = fittingobj.get_lyctmp()
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)
    flxcorr = np.array( [1.0] * len(wavlen_rest) )
    
    params = Parameters()
    params.add('plslp1', value = plslp1, vary=False)
    params.add('plslp2', value = plslp2, vary=False)
    params.add('plbrk', value = plbrk, vary=False)
    params.add('bbt', value = bbt, vary=False)
    params.add('bbflxnrm', value = bbflxnrm, vary=False)
    params.add('elscal', value = elscal, vary=False)
    params.add('scahal', value = scahal, vary=False)
    params.add('galfra', value = galfra, vary=False)
    params.add('ebv', value = ebv, vary=True)
    params.add('imod', value = imod, vary=False)
    params.add('norm', value = 1e-17, vary=True)
    
    def resid(params,
              wav_opt,
              flux_opt):
    
        wav_sed, flux_sed = qsrmod(params,
                                   parfile,
                                   wavlen_rest,
                                   row.z_IR,
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
    
        wav_sed = wav_sed / (1.0 + row.z_IR) 
    
        spc = interp1d(wav_sed, flux_sed, bounds_error=True, fill_value=0.0)
        flux_sed_fit = spc(wav_opt)
    
        return flux_opt - params['norm'].value * flux_sed_fit
      
    
    resid_p = partial(resid,
                      wav_opt = wav_opt[fit_mask],
                      flux_opt = flux_opt[fit_mask])
    
    
    result = minimize(resid_p, params, method='leastsq')
    
    
    # ---------------------------------------------------------------------------------------
    
    xs = np.arange(np.nanmin(wav_opt), np.nanmax(wav_nir), 1)


    wav_sed, flux_sed = qsrmod(result.params,
                               parfile,
                               wavlen_rest,
                               row.z_IR,
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
    
    wav_sed = wav_sed / (1.0 + row.z_IR) 
    spc = interp1d(wav_sed, flux_sed * result.params['norm'].value, bounds_error=True, fill_value=0.0)
    
    # do error weighted fit of spectra to SED model
    # Hewett et al. 1985 

    # mask out regions between bandpasses 

    wav_nir_obs = wav_nir * (1.0 + row.z_IR)
    goodinds = ((wav_nir_obs > 11800.0) & (wav_nir_obs < 13100.0))\
               | ((wav_nir_obs > 15000.0) & (wav_nir_obs < 17500.0))\
               | ((wav_nir_obs > 19500.0) & (wav_nir_obs < 23500.0))

    wav_nir = wav_nir[goodinds]
    flux_nir = flux_nir[goodinds]
    err_nir = err_nir[goodinds]

    goodinds = err_nir > 0.0 

    wav_nir = wav_nir[goodinds]
    flux_nir = flux_nir[goodinds]
    err_nir = err_nir[goodinds]

    k = np.nansum((flux_nir * spc(wav_nir)) / err_nir**2) / np.nansum((spc(wav_nir) / err_nir)**2)
    

    inds = np.argsort(np.diff(wav_nir))[-2:]
    wav_nir[inds] = np.nan
    flux_nir[inds] = np.nan

    axs[0].plot(wav_nir, flux_nir*1e15 / k, color=cs[0], alpha=1.0)


    axs[0].plot(xs, spc(xs)*1e15, color='black', lw=1, label='Model')

    axs[0].legend(loc='upper right')
        
    axs[0].set_xlim(1300,7300)
    axs[0].set_ylim(0, 1)

    # axs[0].set_xlabel(r'Rest-frame wavelength [${\mathrm \AA}$]')
    axs[0].set_ylabel(r'F$_{\lambda}$ [Arbitary units]')

    # --------------------------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    row = df.ix['QSO010']

    wav_nir, dw_nir, flux_nir, err_nir = get_nir_spec(row.NIR_PATH, row.INSTR)      
    
    wav_nir = wav_nir / (1.0 + row.z_IR)

    
    axs[1].plot(wav_nir[wav_nir > 2800.0], flux_nir[wav_nir > 2800.0]*1e16, color=cs_light[0])

    ftrlst, maglst, errlst, lameff = [], [], [], []  

    # 1250 condition so we don't go near the lyman break
    if (~np.isnan(row.psfMag_u)) & ((3546.0 / (1.0 + row.z_IR)) > 1250.0) & (~np.isnan(row.psfMagErr_u)):
        ftrlst.append('u.response')
        maglst.append(row.psfMag_u - 0.91) 
        errlst.append(row.psfMagErr_u)
        lameff.append(3546.0) 
    if (~np.isnan(row.psfMag_g)) & ((4670.0 / (1.0 + row.z_IR)) > 1250.0) & (~np.isnan(row.psfMagErr_g)):
        ftrlst.append('g.response')
        maglst.append(row.psfMag_g + 0.08)
        errlst.append(row.psfMagErr_g)
        lameff.append(4670.0) 
    if (~np.isnan(row.psfMag_r)) & ((6156.0 / (1.0 + row.z_IR)) > 1250.0) & (~np.isnan(row.psfMagErr_r)):
        ftrlst.append('r.response')
        maglst.append(row.psfMag_r - 0.16)
        errlst.append(row.psfMagErr_r)
        lameff.append(6156.0) 
    if (~np.isnan(row.psfMag_i)) & ((7471.0 / (1.0 + row.z_IR)) > 1250.0) & (~np.isnan(row.psfMagErr_i)):
        ftrlst.append('i.response')
        maglst.append(row.psfMag_i - 0.37)
        errlst.append(row.psfMagErr_i)
        lameff.append(7471.0) 
    if (~np.isnan(row.psfMag_z)) & ((8918.0 / (1.0 + row.z_IR)) > 1250.0) & (~np.isnan(row.psfMagErr_z)):
        ftrlst.append('z.response')
        maglst.append(row.psfMag_z - 0.54)
        errlst.append(row.psfMagErr_z)
        lameff.append(8918.0) 

    if (~np.isnan(row.VHS_YAperMag3)) & (~np.isnan(row.VHS_YAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_Y2.txt')
        maglst.append(row.VHS_YAperMag3)
        errlst.append(row.VHS_YAperMag3Err)
        lameff.append(10210.0) 
    elif (~np.isnan(row.Viking_YAperMag3)) & (~np.isnan(row.Viking_YAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_Y2.txt')
        maglst.append(row.Viking_YAperMag3)
        errlst.append(row.Viking_YAperMag3Err)
        lameff.append(10210.0) 
    elif (~np.isnan(row.UKIDSS_YAperMag3)) & (~np.isnan(row.UKIDSS_YAperMag3Err)): 
        ftrlst.append('Y.response')  
        maglst.append(row.UKIDSS_YAperMag3)
        errlst.append(row.UKIDSS_YAperMag3Err)
        lameff.append(10305.0) 

    if (~np.isnan(row.VHS_JAperMag3)) & (~np.isnan(row.VHS_JAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_J2.txt')
        maglst.append(row.VHS_JAperMag3)
        errlst.append(row.VHS_JAperMag3Err)
        lameff.append(12540.0) 
    elif (~np.isnan(row.Viking_JAperMag3)) & (~np.isnan(row.Viking_JAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_J2.txt')
        maglst.append(row.Viking_JAperMag3)
        errlst.append(row.Viking_JAperMag3Err)
        lameff.append(12540.0) 
    elif (~np.isnan(row.UKIDSS_J_1AperMag3)) & (~np.isnan(row.UKIDSS_J_1AperMag3Err)):  
        ftrlst.append('J.response')  
        maglst.append(row.UKIDSS_J_1AperMag3)
        errlst.append(row.UKIDSS_J_1AperMag3Err)
        lameff.append(12483.0) 
    elif (~np.isnan(row['2massMag_j'])) & (~np.isnan(row['2massMagErr_j'])): 
        ftrlst.append('J2MASS.response')  
        maglst.append(row['2massMag_j'])
        errlst.append(row['2massMagErr_j'])
        lameff.append(12350.0) 

    if (~np.isnan(row.VHS_HAperMag3)) & (~np.isnan(row.VHS_HAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_H2.txt') 
        maglst.append(row.VHS_HAperMag3)
        errlst.append(row.VHS_HAperMag3Err)
        lameff.append(16460.0) 
    elif (~np.isnan(row.Viking_HAperMag3)) & (~np.isnan(row.Viking_HAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_H2.txt') 
        maglst.append(row.Viking_HAperMag3)
        errlst.append(row.Viking_HAperMag3Err)
        lameff.append(16460.0) 
    elif (~np.isnan(row.UKIDSS_HAperMag3)) & (~np.isnan(row.UKIDSS_HAperMag3Err)):    
        ftrlst.append('H.response')     
        maglst.append(row.UKIDSS_HAperMag3)
        errlst.append(row.UKIDSS_HAperMag3Err)
        lameff.append(16313.0) 
    elif (~np.isnan(row['2massMag_h'])) & (~np.isnan(row['2massMagErr_h'])): 
        ftrlst.append('H2MASS.response')         
        maglst.append(row['2massMag_h'])
        errlst.append(row['2massMagErr_h'])
        lameff.append(16620.0) 

    if (~np.isnan(row.VHS_KAperMag3)) & (~np.isnan(row.VHS_KAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_Ks2.txt')
        maglst.append(row.VHS_KAperMag3)
        errlst.append(row.VHS_KAperMag3Err)
        lameff.append(21490.0) 
    elif (~np.isnan(row.Viking_KsAperMag3)) & (~np.isnan(row.Viking_KsAperMag3Err)):
        ftrlst.append('VISTA_Filters_at80K_forETC_Ks2.txt')
        maglst.append(row.Viking_KsAperMag3)
        errlst.append(row.Viking_KsAperMag3Err)
        lameff.append(21490.0) 
    elif (~np.isnan(row.UKIDSS_KAperMag3)) & (~np.isnan(row.UKIDSS_KAperMag3Err)):     
        ftrlst.append('K.response')
        maglst.append(row.UKIDSS_KAperMag3)
        errlst.append(row.UKIDSS_KAperMag3Err)
        lameff.append(22010.0) 
    elif (~np.isnan(row['2massMag_k'])) & (~np.isnan(row['2massMagErr_k'])): 
        ftrlst.append('K2MASS.response') 
        maglst.append(row['2massMag_k'])  
        errlst.append(row['2massMagErr_k'])
        lameff.append(21590.0) 

    ftrlst, maglst, errlst, lameff = np.array(ftrlst), np.array(maglst), np.array(errlst), np.array(lameff)


    
       
    #-------Filters---------------------------------------------
    nftr = len(ftrlst)
    bp = np.empty(nftr,dtype='object')
    dlam = np.zeros(nftr)
    
    for nf in range(nftr):
        with open(os.path.join('/home/lc585/Dropbox/IoA/QSOSED/Model/Filter_Response/', ftrlst[nf]), 'r') as f:
            wavtmp, rsptmp = np.loadtxt(f,unpack=True)
        dlam[nf] = (wavtmp[1] - wavtmp[0])
        bptmp = np.ndarray(shape=(2,len(wavtmp)), dtype=float)
        bptmp[0,:], bptmp[1,:] = wavtmp, rsptmp
        bp[nf] = bptmp
    
    #--------------------------------------------------------------------------------
    
    f_0 = np.zeros(nftr) # flux zero points
    fvega = '/data/vault/phewett/vista_work/vega_2007.lis' 
    vspec = np.loadtxt(fvega) 
    vf = interp1d(vspec[:,0], vspec[:,1])
    
    for nf in range(nftr):
        sum1 = np.sum( bp[nf][1] * vf(bp[nf][0]) * bp[nf][0] * dlam[nf])
        sum2 = np.sum( bp[nf][1] * bp[nf][0] * dlam[nf])
        f_0[nf] = sum1 / sum2

    flxlst = f_0 * 10.0**(-0.4 * maglst) # data fluxes in erg/cm^2/s/A
    flxerrlst = flxlst * (-0.4) * np.log(10) * errlst 

    axs[1].scatter(lameff / (1.0 + row.z_IR), flxlst*1e16, s=50, facecolor=cs[5], edgecolor='black', zorder=10, label='Photometry')
   

    plslp1 = 0.46
    plslp2 = 0.03
    plbrk = 2822.0
    bbt = 1216.0
    bbflxnrm = 0.24
    elscal = 0.71
    scahal = 0.86
    galfra = 0.31
    ebv = 0.0
    imod = 18.0
    
    with open('/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit/input.yml', 'r') as f:
        parfile = yaml.load(f)
    
    fittingobj = load(parfile)
    
    lin = fittingobj.get_lin()
    galspc = fittingobj.get_galspc()
    ext = fittingobj.get_ext()
    galcnt = fittingobj.get_galcnt()
    ignmin = fittingobj.get_ignmin()
    ignmax = fittingobj.get_ignmax()
    wavlen_rest = fittingobj.get_wavlen()
    ztran = fittingobj.get_ztran()
    lyatmp = fittingobj.get_lyatmp()
    lybtmp = fittingobj.get_lybtmp()
    lyctmp = fittingobj.get_lyctmp()
    whmin = fittingobj.get_whmin()
    whmax = fittingobj.get_whmax()
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}
    cosmo = cd.set_omega_k_0(cosmo)
    flxcorr = np.array( [1.0] * len(wavlen_rest) )
    
    params = Parameters()
    params.add('plslp1', value = plslp1, vary=False)
    params.add('plslp2', value = plslp2, vary=False)
    params.add('plbrk', value = plbrk, vary=False)
    params.add('bbt', value = bbt, vary=False)
    params.add('bbflxnrm', value = bbflxnrm, vary=False)
    params.add('elscal', value = elscal, vary=False)
    params.add('scahal', value = scahal, vary=False)
    params.add('galfra', value = galfra, vary=False)
    params.add('ebv', value = ebv, vary=True)
    params.add('imod', value = imod, vary=False)
    params.add('norm', value = 1e-17, vary=True)

    resid_p = partial(resid_mag_fit,
                      flx=flxlst,
                      err=flxerrlst,
                      parfile=parfile,
                      wavlen_rest=wavlen_rest,
                      z=row.z_IR,
                      lin=lin,
                      galspc=galspc,
                      ext=ext,
                      galcnt=galcnt,
                      ignmin=ignmin,
                      ignmax=ignmax,
                      ztran=ztran,
                      lyatmp=lyatmp,
                      lybtmp=lybtmp,
                      lyctmp=lyctmp,
                      whmin=whmin,
                      whmax=whmax,
                      cosmo=cosmo,
                      flxcorr=flxcorr,
                      bp=bp,
                      dlam=dlam) 
    
    
    result = minimize(resid_p, params, method='leastsq')

    
    # ---------------------------------------------------------------------------------------
    
    wav_sed, flux_sed = qsrmod(result.params,
                               parfile,
                               wavlen_rest,
                               row.z_IR,
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

    spc = interp1d(wavlen_rest, flux_sed * result.params['norm'].value, bounds_error=True, fill_value=0.0)
    
    xs = np.arange(1000, 10000, 10)

    axs[1].plot(xs, spc(xs)*1e16, color='black', lw=1, label='Model')

    

    # do error weighted fit of spectra to SED model
    # Hewett et al. 1985 

    # mask out regions between bandpasses 

    wav_nir_obs = wav_nir * (1.0 + row.z_IR)
    goodinds = ((wav_nir_obs > 11800.0) & (wav_nir_obs < 13100.0))\
               | ((wav_nir_obs > 15000.0) & (wav_nir_obs < 17500.0))\
               | ((wav_nir_obs > 19500.0) & (wav_nir_obs < 23500.0))

    wav_nir = wav_nir[goodinds]
    flux_nir = flux_nir[goodinds]
    err_nir = err_nir[goodinds]

    goodinds = err_nir > 0.0 

    wav_nir = wav_nir[goodinds]
    flux_nir = flux_nir[goodinds]
    err_nir = err_nir[goodinds]

    k = np.nansum((flux_nir * spc(wav_nir)) / err_nir**2) / np.nansum((spc(wav_nir) / err_nir)**2)
    
    inds = np.argsort(np.diff(wav_nir))[-2:]
    wav_nir[inds] = np.nan
    flux_nir[inds] = np.nan

    axs[1].plot(wav_nir, flux_nir*1e16 / k, color=cs[0], label='Near-IR')

    axs[1].set_xlim(1250, 9000)
    axs[1].set_ylim(0, 5)

    axs[1].set_xlabel(r'Rest-frame wavelength [${\mathrm \AA}$]')
    axs[1].set_ylabel(r'F$_{\lambda}$ [Arbitary units]')

    # -------------------------------------------------

    axs[1].legend(scatterpoints=1)

    axs[0].text(0.1, 0.93, '(a) J092952+355450',
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[0].transAxes)

    axs[1].text(0.1, 0.93, '(b) J100247+002104',
                horizontalalignment='left',
                verticalalignment='center',
                transform = axs[1].transAxes)



    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter02/normalise_to_sdss.pdf')

    plt.show()

    return None 



