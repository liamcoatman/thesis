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


def normalise_to_sdss(row):

    """

    normalise_nirspec_to_sdss_spec(df.ix['QSO125']) 

    """

     
    import sys
    sys.path.insert(0, '/home/lc585/Dropbox/IoA/nirspec/python_code')
    from get_nir_spec import get_nir_spec
    from get_sdss_spec import get_sdss_spec
    
    sys.path.insert(0, '/home/lc585/Dropbox/IoA/QSOSED/Model/qsofit')
    from qsrmod import qsrmod
    from load import load
    import cosmolopy.distance as cd


    fig, ax = plt.subplots(figsize=figsize(1, 0.8))

    cs = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    cs_light = palettable.colorbrewer.qualitative.Pastel1_6.mpl_colors


    wav_nir, dw_nir, flux_nir, err_nir = get_nir_spec(row.NIR_PATH, row.INSTR)      
    
    wav_nir = wav_nir / (1.0 + row.z_IR)

    ax.plot(wav_nir, flux_nir, color=cs_light[0])
    

    if (row.SPEC_OPT == 'SDSS') | (row.SPEC_OPT == 'BOSS+SDSS'):
    
        wav_opt, dw_opt, flux_opt, err_opt = get_sdss_spec('SDSS', row.DR7_PATH)
    
    elif (row.SPEC_OPT == 'BOSS') :
    
        wav_opt, dw_opt, flux_opt, err_opt = get_sdss_spec('BOSS', row.DR12_PATH)
  
    wav_opt = wav_opt / (1.0 + row.z_IR)
    
    ax.plot(wav_opt, flux_opt, color=cs_light[1])
    
    
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
        ax.plot(wav_opt[item], flux_opt[item], color=cs[1])
    
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
               | ((wav_nir_obs > 21000.0) & (wav_nir_obs < 23500.0))

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

    ax.plot(wav_nir, flux_nir / k, color=cs[0], alpha=1.0)


    ax.plot(xs, spc(xs), color='black', lw=1)
    # ax.set_xscale('log')
    # ax.set_yscale('log')

        
    ax.set_xlim(1300,7300)
    ax.set_ylim(2e-17, 1e-15)

    ax.set_xlabel(r'Rest-frame wavelength [${\mathrm \AA}$]')
    ax.set_ylabel(r'F$_{\lambda}$ [Arbitary units]')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter02/normalise_to_sdss.pdf')

    plt.show()

    return None 