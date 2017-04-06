from astropy.table import Table, join 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import cPickle as pickle
import os
import time
import numpy.ma as ma 
from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
import palettable 
import matplotlib.patches as patches
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
import pandas as pd 
import astropy.constants as const 
import astropy.units as u 
import sys
import matplotlib.gridspec as gridspec
from lmfit import Parameters
from SpectraTools.fit_line import wave2doppler
from lmfit.models import GaussianModel, ConstantModel
from PlottingTools.gausshermite import gausshermite_4, gausshermite_2
from scipy.interpolate import interp1d
from lmfit import Model

def plot_MCMC_model(ax, xdata, ydata, sigma_y, trace, linestyle='--', label='', show_sigma=True):

    """
    Plot the linear model and 2sigma contours
    Take every b, m combination, and then at each
    x find the mean and the standard deviation of y
    """
    
    # ax.plot(xdata, ydata, 'ok')
    # ax.errorbar(xdata, ydata, yerr=sigma_y, linestyle='', color='black')

    m, b = trace[:2]
    # xfit = np.linspace(xdata.min(), xdata.max(), 10)
    xfit = np.linspace(-1.0, 6.0, 10)
    yfit = b[:, None] + m[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, 'k', linestyle=linestyle, zorder=5, label=label)

    if show_sigma:
        ax.fill_between(xfit, 
                        mu - sig, 
                        mu + sig, 
                        color=palettable.colorbrewer.qualitative.Pastel1_3.mpl_colors[1], 
                        zorder=1)

def civ_space_z_compare_plot():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors

    # import ipdb; ipdb.set_trace()
    # t_hw = Table.read('/data/lc585/QSOSED/Results/140827/civtab.fits') # old blueshifts
  
    t_hw = Table.read('/data/vault/phewett/LiamC/liam_civpar_zhw_160115.dat', format='ascii') # new HW10 
    t_ica = Table.read('/data/vault/phewett/LiamC/liam_civpar_zica_160115.dat', format='ascii') # new ICA 

    fig, axs = plt.subplots(3, 1, figsize=(figsize(0.65, vscale=2.31)), sharex=True)
          
    m1, m2 = t_hw['col2'], np.log10( t_hw['col3'])
    
    badinds = np.isnan(m1) | np.isnan(m2) | np.isinf(m1) | np.isinf(m2)

    m1 = m1[~badinds]
    m2 = m2[~badinds]

    xmin = -1000.0
    xmax = 3000.0
    ymin = 1.0
    ymax = 2.5

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    CS = axs[1].contour(X,Y,Z, colors=['grey'])

    threshold = CS.levels[0]

    z = kernel(values)

    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axs[1].scatter(x, y, c='grey', edgecolor='None', s=3, label='SDSS DR7', rasterized=False)

    axs[1].set_xlim(-1400,4500)
    axs[1].set_ylim(1,2.2)

    ########################################################################

    m1, m2 = t_ica['col2'], np.log10( t_ica['col3'])

    badinds = np.isnan(m1) | np.isnan(m2) | np.isinf(m1) | np.isinf(m2)

    m1 = m1[~badinds]
    m2 = m2[~badinds]

    xmin = -1000.0
    xmax = 3500.0
    ymin = 1.0
    ymax = 2.5

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    CS = axs[2].contour(X,Y,Z, colors=['grey'])

    threshold = CS.levels[0]

    z = kernel(values)

    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axs[2].scatter(x, y, c='grey', edgecolor='None', s=3, label='SDSS DR7', rasterized=False)

    axs[2].set_xlim(axs[1].get_xlim())
    axs[2].set_ylim(axs[1].get_ylim())

    ########################################################################

    # Second panel have SDSS blueshifts 
    shen = Table.read('/data/lc585/SDSS/dr7_bh_Nov19_2013.fits')
    
    t1 = Table()
    t1['NAME'] = shen['SDSS_NAME']
    t1['z'] = shen['REDSHIFT']
    t1['CIV_EQW'] = shen['EW_CIV']
    t1['VOFF_CIV_PEAK'] = shen['VOFF_CIV_PEAK']
    
    t2 = Table()
    t2['NAME'] = [i.replace('SDSSJ', '') for i in t_hw['col1']]
     
    t3 = join(t1, t2, join_type='right', keys='NAME')
    
    m1, m2 = t3['VOFF_CIV_PEAK'].data, np.log10(t3['CIV_EQW'].data)

    masked = ma.getmask(m1) | ma.getmask(m2)
    
    m1, m2 = m1[~masked], m2[~masked]
       
    xmin = -1000.0
    xmax = 2000.0
    ymin = 1.0
    ymax = 2.5
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    
    kernel = stats.gaussian_kde(values)
    
    Z = np.reshape(kernel(positions).T, X.shape)
    
    CS = axs[0].contour(X, Y, Z, colors=['grey'])
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axs[0].scatter(x, y, c='grey', edgecolor='None', s=3, label='SDSS DR7', rasterized=False)

    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_ylim(axs[1].get_ylim())

    axs[2].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[0].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')
    axs[1].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')
    axs[2].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')

    axs[0].text(4000, 2.12, '(a)', ha='center', va='center')
    axs[1].text(4000, 2.12, '(b)', ha='center', va='center')
    axs[2].text(4000, 2.12, '(c)', ha='center', va='center')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    fig.savefig('/home/lc585/thesis/figures/chapter03/civ_space_z_compare.pdf')

    plt.show() 

    return None 

def composite_plot():

    import palettable 
    from scipy.constants import c
    import matplotlib.pyplot as plt 
    import numpy as np 
    from PlottingTools.plot_setup import figsize, set_plot_properties
    import os 

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors

    stem = '/data/vault/phewett/ICAtest/DR7_zica'

    fig, ax = plt.subplots(figsize=(figsize(0.75, vscale=0.75)))

    #composites = ['civ_bs_p0002.comp',
    #              'civ_bs_p0598.comp',
    #              'civ_bs_p0952.comp',
    #              'civ_bs_p1317.comp',
    #              'civ_bs_p1629.comp',
    #              'civ_bs_p2198.comp']

    composites = ['civ_bs_p0002.comp',
                  'civ_bs_p1317.comp',
                  'civ_bs_p2198.comp']

    labels = ['0 km/s','1300 km/s','2200 km/s']

    # Transform to velocity space
    
    c = c * 1e-3
    line_wlen = np.mean([1548.202,1550.774])

    linestyles = ['-', '--', ':']
    lines = []
    for i, composite in enumerate(composites):
        f = np.genfromtxt( os.path.join(stem,composite))
        x_data = (f[:,0] - line_wlen ) * c / line_wlen
        l, = ax.plot( x_data, f[:,1], lw=2, label=labels[i], color=cs[i], linestyle='-')
        lines.append(l)

    ax.legend(handles=lines, labels=labels, prop={'size':10},frameon=True)

    ax.set_xlim(-10000,10000)
    ax.set_ylim(1,2.2)
    ax.grid() 

    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'$F_{\lambda}$ [Scaled Units]')

    ax.axvline(0.0, linestyle='--', color='grey')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/civ_composites.pdf')

    plt.show() 

    return None 

def example_liris_spectra():

    from fit_wht_spectra import get_line_fit_props
    from scipy import ndimage
    import sys 
    sys.path.insert(0, '/home/lc585/Dropbox/IoA/BlackHoleMasses')
    from wht_properties_v3 import get_wht_quasars
    import numpy as np 
    import matplotlib.pyplot as plt 
    from PlottingTools.plot_setup import figsize, set_plot_properties
    import os 
    from astropy.io import fits 
    from SpectraTools.get_wavelength import get_wavelength

    set_plot_properties() # change style 

    quasars = get_wht_quasars().all_quasars()
    fit_props = get_line_fit_props().all_quasars()

    quasars, fit_props = np.array(quasars), np.array(fit_props)

    quasars = quasars[[0, 3, 14, 15, 16, 17]]
    fit_props = fit_props[[0, 3, 14, 15, 16, 17]]

    n = len(quasars)

    fig, ax = plt.subplots(1, 1, figsize=(figsize(0.75)))

    p = quasars[4]
    fname = os.path.join('/data/lc585/WHT_20150331/html/',p.name,'tcdimcomb.ms.fits')

    hdulist = fits.open(fname)
    hdr = hdulist[0].header
    data = hdulist[0].data
    hdulist.close()
    wav, dw = get_wavelength(hdr)
    flux = data[0,:,:].flatten()

    wav = wav / (1.0 + p.z_ICA)
    flux = flux * 1e18
    flux = ndimage.filters.median_filter(flux, 3)

    goodinds1 = np.argmin(np.abs(wav-5500))
    goodinds2 = np.argmin(np.abs(wav-6200))

    ax.plot(wav[:goodinds1], flux[:goodinds1], color='black', lw=1)
    ax.plot(wav[goodinds2:], flux[goodinds2:], color='black', lw=1)
    ax.plot(wav[goodinds1:goodinds2], flux[goodinds1:goodinds2], color='grey', lw=1, linestyle='--')

    ax.axvspan(5500, 6200, edgecolor='black', facecolor=(0.75098039,0.75098039,0.75098039))

    ax.set_xlim(4500,7500)
    ax.set_ylim(0,3)
    

    ax.set_xlabel(r'Rest-Frame $\lambda$ [\AA]')

    ax.set_ylabel(r'F$_{\lambda}$ [$\times 10^{18}~\rm{erg}~\rm{s}^{-1}$~$\rm{cm}^{2}$~$\rm{\AA}^{-1}$]')


    ax.text(4863,2.3,r'H$\beta$',ha='center')
    ax.text(6565,2.3,r'H$\alpha$',ha='center')

    ax.plot([4863,4863], [1.5,2.2], linestyle='-', c='black')


    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter02/example_liris_spectrum.pdf')

    plt.show() 

    return None 

def fitting_comparison():

    from lmfit import Parameters
    import cPickle as pickle 
    from SpectraTools.fit_line import wave2doppler
    import astropy.units as u 
    from lmfit.models import GaussianModel, ConstantModel 
    from SpectraTools.fit_line_gauss_hermite import gausshermite
    from scipy.interpolate import interp1d
    import palettable 
    from PlottingTools.plot_setup import figsize, set_plot_properties 
    import matplotlib.pyplot as plt 
    import numpy as np 
    import os 

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors
    
    fig, ax = plt.subplots(figsize=figsize(0.75))
    
    xs, step = np.linspace(-20000,
                           20000,
                           1000,
                           retstep=True)
    
    bkgd = ConstantModel()
    mod_Ha = bkgd

    for i in range(3):
        gmod = GaussianModel(prefix='g{}_'.format(i))
        mod_Ha += gmod

    save_dir = 'CIV_QSO445_Gauss_Fit'
    
    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params_Ha = Parameters()
    params_Ha.load(parfile)
    parfile.close()
    
    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav_Ha = pickle.load(parfile)
    parfile.close()
    
    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx_Ha = pickle.load(parfile)
    parfile.close()
    
    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err_Ha = pickle.load(parfile)
    parfile.close()
    
    vdat_Ha = wave2doppler(wav_Ha, np.mean([1548.202,1550.774])*u.AA)
    
    ax.plot(vdat_Ha,
            flx_Ha,
            color='grey',
            lw=1,
            alpha=1)
    
    line2, = ax.plot(xs,
                    mod_Ha.eval(params=params_Ha, x=xs) ,
                    color = cs[0],
                    lw=2,
                    label = '3-Gaussian Composite')
    
    save_dir = 'CIV_QSO445_GH_Fit'

    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params_CIV = Parameters()
    params_CIV.load(parfile)
    parfile.close()
    
    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav_CIV = pickle.load(parfile)
    parfile.close()
    
    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx_CIV = pickle.load(parfile)
    parfile.close()
    
    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err_CIV = pickle.load(parfile)
    parfile.close()
    
    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()
    
    line1, = ax.plot(xs,
                     gausshermite(xs/sd, params_CIV, 2),
                     color = cs[1],
                     label = '2nd Order GH Polynomial',
                     lw=2)
    

    xs = np.arange(-19000, 5000, 1)

    # bin1 = (vdat_Ha.value > -19222) & (vdat_Ha.value < -18418)
    # bin2 = (vdat_Ha.value > -17303) & (vdat_Ha.value < -15384)
    # bin3 = (vdat_Ha.value > -13302) & (vdat_Ha.value < -11810)
    # bin4 = (vdat_Ha.value > -10186) & (vdat_Ha.value < -9153)
    # bin5 = (vdat_Ha.value > -6530) & (vdat_Ha.value < -6169)
    # bin6 = (vdat_Ha.value > -4267) & (vdat_Ha.value < -3758)
    # bin7 = (vdat_Ha.value > -1922) & (vdat_Ha.value < -1315)
    # bin8 = (vdat_Ha.value > -167) & (vdat_Ha.value < 177)
    # bin9 = (vdat_Ha.value > 948) & (vdat_Ha.value < 1243)
    # bin10 = (vdat_Ha.value > 2243) & (vdat_Ha.value < 2637)
    # bin11 = (vdat_Ha.value > 3867) & (vdat_Ha.value < 4392)
    # bin12 = (vdat_Ha.value > 6048) & (vdat_Ha.value < 6671)

    bin1 = (vdat_Ha.value > -20000) & (vdat_Ha.value < -18418)
    bin2 = (vdat_Ha.value > -17303) & (vdat_Ha.value < -15384)
    bin3 = (vdat_Ha.value > -13302) & (vdat_Ha.value < -11810)
    bin6 = (vdat_Ha.value > -4267) & (vdat_Ha.value < -3758)
    bin7 = (vdat_Ha.value > -1922) & (vdat_Ha.value < -1315)
    bin8 = (vdat_Ha.value > -167) & (vdat_Ha.value < 177)
    bin9 = (vdat_Ha.value > 948) & (vdat_Ha.value < 1243)
    bin10 = (vdat_Ha.value > 2243) & (vdat_Ha.value < 2637)
    bin11 = (vdat_Ha.value > 3867) & (vdat_Ha.value < 4392)
    bin12 = (vdat_Ha.value > 6048) & (vdat_Ha.value < 6671)


    fx = [np.mean(vdat_Ha.value[bin1]),  
          np.mean(vdat_Ha.value[bin2]),  
          np.mean(vdat_Ha.value[bin3]),  
          np.mean(vdat_Ha.value[bin6]),  
          np.mean(vdat_Ha.value[bin7]),            
          np.mean(vdat_Ha.value[bin8]),
          np.mean(vdat_Ha.value[bin9]),  
          np.mean(vdat_Ha.value[bin10]),  
          np.mean(vdat_Ha.value[bin11]),            
          np.mean(vdat_Ha.value[bin12])] 

    fy = [np.median(flx_Ha[bin1]),  
          np.median(flx_Ha[bin2]),  
          np.median(flx_Ha[bin3]),  
          np.median(flx_Ha[bin6]),  
          np.median(flx_Ha[bin7]),            
          np.median(flx_Ha[bin8]),
          np.median(flx_Ha[bin9]),  
          np.median(flx_Ha[bin10]),  
          np.median(flx_Ha[bin11]),            
          np.median(flx_Ha[bin12])] 



    f = interp1d(fx, fy, kind='cubic', bounds_error=False, fill_value=0.0)
     
    # find FWHM

    
    half_max = np.max(f(xs)) / 2.0 

    i = 0  
    while f(xs[i]) < half_max:
        i+=1

    root1 = xs[i]

    i = 0
    while f(xs[-i]) < half_max:
        i+=1

    root2 = xs[-i]

    # print root2 - root1 

    # line1, = ax.plot(xs,
    #                  f(xs),
    #                  color = 'black',
    #                  label = 'Cubic-Spline',
    #                  lw=1)

    ax.set_xlim(-18000,6686)
    ax.set_ylim(-0.05,1) 
    
    plt.legend(loc='upper left') 

    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'F$_{\lambda}$')

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter02/fitting_comparison.pdf')

    plt.show() 
   
    return None 


def get_data(data = 'linewidths'):

    if data == 'hogg':

        from astroML.datasets import fetch_hogg2010test
        data = fetch_hogg2010test()
        data = data[5:]  # no outliers
        xi = data['x']
        yi = data['y']
        dxi = data['sigma_x']
        dyi = data['sigma_y']
        rho_xy = data['rho_xy'] 

    elif data == 'linewidths':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort_values('Blueshift_CIV_Ha')
        
        xi = df.Blueshift_CIV_Ha.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Ha.values 
        dxi = df.Blueshift_CIV_Ha_Err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    elif data == 'linewidths_corrected_ha':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort_values('Blueshift_CIV_Ha')
       
        xi = df.Blueshift_CIV_Ha.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Ha_Corr.values
        blueshift_err = np.sqrt(df.Median_Broad_Ha_Err**2 + df.Median_CIV_BEST_Err**2) 
        dxi = blueshift_err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha_Corr)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    elif data == 'linewidths_ha':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort_values('Blueshift_CIV_Ha')
       
        xi = df.Blueshift_CIV_Ha.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Ha.values
        blueshift_err = np.sqrt(df.Median_Broad_Ha_Err**2 + df.Median_CIV_BEST_Err**2) 
        dxi = blueshift_err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    elif data == 'linewidths_hb':

        df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
        df = df[df.WARN_Hb == 0]
        df = df[['rescale' not in i for i in df.SPEC_NIR.values]]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort_values('Blueshift_CIV_Hb')

       
        xi = df.Blueshift_CIV_Hb.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Hb.values
        dxi = df.Blueshift_CIV_Hb_Err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    elif data == 'linewidths_hb_corrected':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Hb == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        df = df[['rescale' not in i for i in df.SPEC_NIR.values]]
        
        df = df.sort_values('Blueshift_CIV_Hb')

       
        xi = df.Blueshift_CIV_Hb.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Hb_Corr.values
        dxi = df.Blueshift_CIV_Hb_Err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))


    elif data == 'linewidths_hb_equiv':

        df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        df = df[['rescale' not in i for i in df.SPEC_NIR.values]]
        
        df = df.sort_values('Blueshift_CIV_Ha')

        fwhm = df['FWHM_Broad_Ha'] * 1.e-3 
        fwhm_err = df['FWHM_Broad_Ha_Err'] * 1.e-3 
        
        fwhm_hb = 1.23e3 * np.power(fwhm, 0.97)
        fwhm_hb_err = 1.23e3 * np.power(fwhm, 0.97-1.0) * 0.97 * fwhm_err

        xi = df.Blueshift_CIV_Ha.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / fwhm_hb
        blueshift_err = np.sqrt(df.Median_Broad_Ha_Err**2 + df.Median_CIV_BEST_Err**2) 
        dxi = blueshift_err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (fwhm_hb_err / fwhm_hb)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    return xi, yi, dxi, dyi, rho_xy 

def correction_and_bhm_ha():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    fig, axs = plt.subplots(2, 1, figsize=figsize(0.7, vscale=1.6), sharex=True)

    plt.subplots_adjust(hspace=0.05)

    #----------------------------------------------------------------------
    trace = np.load('/data/lc585/BHMassPaper2_Resubmitted_MCMC_Traces/trace_civ_hb_equiv_relation.npy') 
    xi, yi, dxi, dyi, rho_xy = get_data(data='linewidths_hb_equiv') 
    
    plot_MCMC_model(axs[0], xi, yi, dyi, trace[:2, :], linestyle='-')
    trace = None 
   
    axs[0].errorbar(xi, yi, yerr=dyi, xerr=dxi, linestyle='', color='grey', alpha=0.4, zorder=2)
    axs[0].scatter(xi, yi, color=cs[1], s=8, zorder=3)
    axs[0].set_ylabel(r'FWHM C\,{\sc iv} / FWHM H$\alpha$') 
    axs[0].set_ylim(0, 3.5)
    axs[0].text(-0.5, 3., '(a)')

    #--------------------------------------------------------------------
    

    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    df.dropna(subset=['Blueshift_CIV_Ha', 'LogMBH_CIV_VP06', 'LogMBH_Ha'], inplace=True)

    xi = df.Blueshift_CIV_Ha.values / 1.0e3 
    dxi = df.Blueshift_CIV_Ha_Err.values / 1.0e3

    bhm_civ = 10**df.LogMBH_CIV_VP06.values
    bhm_ha = 10**df.LogMBH_Ha.values
    
    yi = bhm_civ / bhm_ha

    d_bhm_civ = bhm_civ * np.log(10) * df.LogMBH_CIV_VP06_Err.values
    d_bhm_ha = bhm_ha * np.log(10) * df.LogMBH_Ha_Err.values

    dyi = yi * np.sqrt((d_bhm_civ/bhm_civ)**2 + (d_bhm_ha/bhm_ha)**2)

    axs[1].errorbar(xi, 
                    yi, 
                    yerr=dyi,  
                    xerr=dxi, 
                    linestyle='', 
                    color='grey', 
                    alpha=0.4, 
                    zorder=2)

    axs[1].scatter(xi, 
                   yi, 
                   color=cs[1], 
                   s=8, 
                   zorder=3)

    axs[1].axhline(1, color='black', linestyle = '--')

    trace = np.load('/data/lc585/BHMassPaper2_Resubmitted_MCMC_Traces/trace_civ_hb_equiv_relation.npy') 
 
    m, b = trace[:2]

    trace = None 

    xfit = np.linspace(-1.0, 6.0, 100)
    yfit = b[:, None] + m[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)


    f = lambda x, n: n*((np.median(m[:, None]) * x) + np.median(b[:, None]))**2

      
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(f, xi, yi)

    axs[1].plot(xfit, popt[0]*mu**2, 'k', linestyle='-', zorder=5)
    axs[1].fill_between(xfit, popt[0]*(mu - sig)**2, popt[0]*(mu + sig)**2, color=palettable.colorbrewer.qualitative.Pastel1_3.mpl_colors[1], zorder=1)

    axs[1].set_yscale('log')
    axs[1].set_ylim(0.1, 15)
    axs[1].set_xlim(-1, 6.0)

    axs[1].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[1].set_ylabel(r'BHM C\,{\sc iv} / BHM H$\alpha$')


    axs[1].text(-0.5, 10, '(b)')

  

    plt.setp([axs[1]], 
             xticks=[-1, 0, 1, 2, 3, 4, 5, 6], 
             xticklabels=['-1000', '0', '1000', '2000', '3000', '4000', '5000', '6000'],
             yticks=[0.1, 1, 10],
             yticklabels=['.1', '1', '10'])

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/fwhm_and_bhm_ha.pdf')

    plt.show() 

    return None 

def correction_and_bhm_hb():


    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    fig, axs = plt.subplots(2, 1, figsize=figsize(0.7, vscale=1.6), sharex=True)

    plt.subplots_adjust(hspace=0.05)

   
    #----------------------------------------------------------------------
    trace = np.load('/data/lc585/BHMassPaper2_Resubmitted_MCMC_Traces/trace_civ_hb_relation.npy') 
    xi, yi, dxi, dyi, rho_xy = get_data(data='linewidths_hb') 
    
    plot_MCMC_model(axs[0], xi, yi, dyi, trace[:2, :], linestyle='-')  
    trace = None 

    axs[0].errorbar(xi, yi, yerr=dyi, xerr=dxi, linestyle='', color='grey', alpha=0.4, zorder=2)
    axs[0].scatter(xi, yi, color=cs[1], s=8, zorder=3)
    axs[0].set_ylabel(r'FWHM C\,{\sc iv} / FWHM H$\beta$') 
    axs[0].set_ylim(0, 3.5)
    axs[0].text(-0.5, 3., '(a)')

    #--------------------------------------------------------------------
    

    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    df = df[df.WARN_Hb == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    df.dropna(subset=['Blueshift_CIV_Hb', 'LogMBH_CIV_VP06', 'LogMBH_Hb'], inplace=True)

    xi = df.Blueshift_CIV_Hb.values / 1.0e3 
    dxi = df.Blueshift_CIV_Hb_Err.values / 1.0e3

    bhm_civ = 10**df.LogMBH_CIV_VP06.values
    bhm_hb = 10**df.LogMBH_Hb.values
    
    yi = bhm_civ / bhm_hb

    d_bhm_civ = bhm_civ * np.log(10) * df.LogMBH_CIV_VP06_Err.values
    d_bhm_hb = bhm_hb * np.log(10) * df.LogMBH_Hb_Err.values

    dyi = yi * np.sqrt((d_bhm_civ/bhm_civ)**2 + (d_bhm_hb/bhm_hb)**2)

    axs[1].errorbar(xi, 
                    yi, 
                    yerr=dyi,  
                    xerr=dxi, 
                    linestyle='', 
                    color='grey', 
                    alpha=0.4, 
                    zorder=2)

    axs[1].scatter(xi, 
                   yi, 
                   color=cs[1], 
                   s=8, 
                   zorder=3)

    axs[1].axhline(1, color='black', linestyle = '--')

    trace = np.load('/data/lc585/BHMassPaper2_Resubmitted_MCMC_Traces/trace_civ_hb_relation.npy') 
 
    m, b = trace[:2]

    trace = None 

    xfit = np.linspace(-1.0, 6.0, 100)
    yfit = b[:, None] + m[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)


    f = lambda x, n: n*((np.median(m[:, None]) * x) + np.median(b[:, None]))**2

      
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(f, xi, yi)

    axs[1].plot(xfit, popt[0]*mu**2, 'k', linestyle='-', zorder=5)
    axs[1].fill_between(xfit, popt[0]*(mu - sig)**2, popt[0]*(mu + sig)**2, color=palettable.colorbrewer.qualitative.Pastel1_3.mpl_colors[1], zorder=1)
  
    axs[1].set_yscale('log')
    axs[1].set_ylim(0.1, 15)
    axs[1].set_xlim(-1, 6)

    plt.setp([axs[1]], 
             xticks=[-1, 0, 1, 2, 3, 4, 5, 6], 
             xticklabels=['-1000', '0', '1000', '2000', '3000', '4000', '5000', '6000'],
             yticks=[0.1, 1, 10],
             yticklabels=['.1', '1', '10'])

    axs[1].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[1].set_ylabel(r'BHM C\,{\sc iv} / BHM H$\beta$')

    axs[1].text(-0.5, 10, '(b)')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/fwhm_and_bhm_hb.pdf')

    plt.show() 

    return None 

def test_corrections():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    fig, axs = plt.subplots(3, 1, figsize=figsize(0.9, vscale=1.5), sharex=True, sharey=True)
    

    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)


    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[df.WARN_1400_BEST == 0]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    for ax in axs: ax.grid() 


    #----- Our correction ------------------------------------

    # xi = df.LogMBH_Ha.values 
    
    # fwhm = df['FWHM_Broad_Ha'] * 1.e-3 
            
    # fwhm_hb = 1.23e3 * np.power(fwhm, 0.97)
    
    # m, b = 0.41, 0.62
   
    # fwhm_ha = df.FWHM_CIV_BEST / (m * df.Blueshift_CIV_Ha * 1e-3 + b)

    # fwhm_ha = fwhm_ha * 1e-3   
    # l1350 = 10**(df['LogL1350'].values) * 1e-44

    # p1 = np.power(10, 6.71)
    # p2 = np.power(fwhm_ha, 2)
    # p3 = np.power(l1350, 0.53)

    # yi = p1 * p2 * p3

    # axs[3].scatter(df['Blueshift_CIV_Ha'], 
    #                yi / 10**df['LogMBH_Ha'], 
    #                facecolor=cs[1], 
    #                label='This paper', 
    #                edgecolor='None', 
    #                marker='o', 
    #                alpha=1.0,
    #                zorder=1,
    #                s=25)

    # axs[3].set_yscale('log')
    # axs[3].set_ylim(0.1, 10)
    # axs[3].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')

    #-----------------------------------------------------------------------------------

    bhm_hb_predicted_runnoe = 10**(df['LogMBH_CIV_VP06']\
                              - 0.734 - 1.227 * df['1400_CIV_BEST'].apply(np.log10))
    


    axs[1].scatter(df['Blueshift_CIV_Ha'], 
                   bhm_hb_predicted_runnoe / 10**df['LogMBH_Ha'], 
                   facecolor=cs[1], 
                   label='Runnoe et al. (2013a)', 
                   edgecolor='None', 
                   marker='o', 
                   alpha=1.0,
                   zorder=1,
                   s=25)

    axs[1].set_yscale('log')


    bhm_hb_predicted_denney = 10**(df['LogMBH_CIV_VP06']\
                              + 0.219 - 1.63 * np.log10(df['FWHM_CIV_BEST'] / df['Sigma_CIV_BEST']))

 
    axs[0].scatter(df['Blueshift_CIV_Ha'], 
                   bhm_hb_predicted_denney / 10**df['LogMBH_Ha'], 
                   facecolor=cs[1], 
                   label='Denney (2012)', 
                   edgecolor='None', 
                   marker='o', 
                   alpha=1.0,
                   zorder=1,
                   s=25)

    axs[0].set_yscale('log')

    log_bhm_hb_predicted_park = 7.48 + 0.52*np.log10((10**df['LogL1350'])*1e-44)\
                                + 0.56*np.log10(df['FWHM_CIV_BEST']*1e-3)

    bhm_hb_predicted_park = 10**log_bhm_hb_predicted_park
    
    axs[2].scatter(df['Blueshift_CIV_Ha'], 
                   bhm_hb_predicted_park / 10**df['LogMBH_Ha'], 
                   facecolor=cs[1], 
                   label='Park et al. (2013)', 
                   edgecolor='None', 
                   marker='o', 
                   alpha=1.0,
                   zorder=1,
                   s=25)
    
    axs[2].set_yscale('log')

    axs[2].set_ylim(0.1, 10)
    axs[2].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')


    fig.text(0.04, 0.5, r'BHM C\,{\sc iv} (Corrected) / BHM H$\alpha$', va='center', rotation='vertical')

    # axs[3].text(0.05, 0.9, '(d) This work', transform = axs[3].transAxes)
    axs[0].text(0.05, 0.9, '(a) Denney (2012)', transform = axs[0].transAxes)
    axs[1].text(0.05, 0.9, '(b) Runnoe et al. (2013)', transform = axs[1].transAxes) 
    axs[2].text(0.05, 0.9, '(c) Park et al. (2013)', transform = axs[2].transAxes) 

    # axs[0, 0].set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])

    # axs[0, 0].set_yticks([0.1, 1, 10])
    # axs[0, 0].set_yticklabels(['0.1', '1', '10'])

    # axs[1, 0].set_yticks([0.1, 1, 10])
    # axs[1, 0].set_yticklabels(['0.1', '1', '10'])

    fig.tight_layout()

    plt.subplots_adjust(hspace=0.1, left = 0.15)

    fig.savefig('/home/lc585/thesis/figures/chapter03/corrections.pdf')

    plt.show() 

    return None 

def test_corrections_coatman():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    fig, ax = plt.subplots(1, 1, figsize=figsize(0.9, vscale=0.6))
    

    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)


    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[df.WARN_1400_BEST == 0]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    ax.grid() 


    #----- Our correction ------------------------------------

    xi = df.LogMBH_Ha.values 
    
    fwhm = df['FWHM_Broad_Ha'] * 1.e-3 
            
    fwhm_hb = 1.23e3 * np.power(fwhm, 0.97)
    
    m, b = 0.41, 0.62
   
    fwhm_ha = df.FWHM_CIV_BEST / (m * df.Blueshift_CIV_Ha * 1e-3 + b)

    fwhm_ha = fwhm_ha * 1e-3   
    l1350 = 10**(df['LogL1350'].values) * 1e-44

    p1 = np.power(10, 6.71)
    p2 = np.power(fwhm_ha, 2)
    p3 = np.power(l1350, 0.53)

    yi = p1 * p2 * p3

    ax.scatter(df['Blueshift_CIV_Ha'], 
               yi / 10**df['LogMBH_Ha'], 
               facecolor=cs[1], 
               label='This paper', 
               edgecolor='None', 
               marker='o', 
               alpha=1.0,
               zorder=1,
               s=25)

    ax.set_yscale('log')
    ax.set_ylim(0.1, 10)
    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'BHM C\,{\sc iv} (Corrected) / BHM H$\alpha$')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/corrections_coatman.pdf')

    plt.show() 

    return None 

def gaussian(mu, sig, x):
    return (2.0 * np.pi * sig**2)**-0.5 * np.exp(-(x - mu)**2 / (2.0*sig**2))

def log_likelihood(p, x):
    return np.sum(np.log(gaussian(p[0], p[1], x.value) ))

 


def ha_z_comparison(): 

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_8.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    
    fig, axs = plt.subplots(2, 1, figsize=figsize(0.9, 1.3), sharex=True) 

    xi = const.c.to(u.km/u.s)*(df.OIII_FIT_HA_Z - df.z_Broad_Ha)/(1.0 + df.OIII_FIT_HA_Z)
    xi = xi[~np.isnan(xi)]
    
    axs[0].hist(xi,
                histtype='stepfilled',
                color=cs[1],
                bins=np.arange(-1000, 1000, 100),
                zorder=1,
                normed=False)

    print np.mean(xi), np.median(xi), np.std(xi)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.WARN_Hb == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]

    xi = const.c.to(u.km/u.s)*(df.OIII_FIT_HB_Z - df.z_Broad_Hb)/(1.0 + df.OIII_FIT_HB_Z)
    xi = xi[~np.isnan(xi)]
    
    axs[1].hist(xi,
                histtype='stepfilled',
                color=cs[1],
                bins=np.arange(-1000, 1000, 100),
                zorder=1,
                normed=False)

    print np.mean(xi), np.median(xi), np.std(xi)

    axs[1].set_xlabel(r'$c(z_{{\rm H}\alpha,1} - z_{{\rm H}\alpha,2}) / (1 + z_{{\rm H}\alpha,1})$ [km~$\rm{s}^{-1}$]')

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter03/ha_z_comparison.pdf')

    plt.show() 


    
    return None 


def shen_comparison_hb():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    
    fig, ax = plt.subplots(figsize=(figsize(0.8, vscale=0.9)))

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Hb == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]
    df.dropna(subset=['FWHM_BROAD_HB_S16', 'FWHM_Broad_Hb'], inplace=True)

    scatter1 = ax.scatter(df.loc[:, 'FWHM_BROAD_HB_S16'],
                          df.loc[:,'FWHM_Broad_Hb'],
                          c=cs[1],
                          s=25,
                          edgecolor='None')


    ax.plot([2000,10000], [2000,10000], color='black', linestyle='--')
    
    ax.set_xlim(2000,10000)
    ax.set_ylim(ax.get_xlim())

    ax.set_xlabel(r'Shen (2016)~~FWHM(H$\beta$) [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'FWHM(H$\beta$) [km~$\rm{s}^{-1}$]')

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/shen_comparison_hb.pdf')

    plt.show()

    return None 

def shen_comparison_ha():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    
    fig, ax = plt.subplots(figsize=(figsize(0.8, vscale=0.9)))

    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]  
    df.dropna(subset=['FWHM_Ha_S12', 'FWHM_Broad_Ha'], inplace=True)  

    scatter1 = ax.scatter(df.loc[:, 'FWHM_Ha_S12'],
                          df.loc[:, 'FWHM_Broad_Ha'],
                          c=cs[1],
                          s=25,
                          edgecolor='None')
    
    ax.plot([2000,9000], [2000,9000], color='black', linestyle='--')

    
    ax.set_xlim(2000,9000)
    ax.set_ylim(ax.get_xlim())

    ax.set_xlabel(r'Shen \& Lui (2012)~~FWHM(H$\alpha$) [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'FWHM(H$\alpha$) [km~$\rm{s}^{-1}$]')

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/shen_comparison_ha.pdf')

    plt.show()

    return None 

def shen_comparison_civ():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    
    fig, ax = plt.subplots(figsize=(figsize(0.8, vscale=0.9)))

    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[df1.WARN_CIV == 0]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[df2.WARN_CIV == 0]
    df2 = df2[df2.BAL_FLAG != 1] 
        
    df = pd.concat([df1, df2]).drop_duplicates()
    df[df.FWHM_CIV_S11 == 0.0] = np.nan
    df = df.dropna(subset=['FWHM_CIV_S11', 'FWHM_CIV'])
    df = df[df.SPEC_OPT == 'SDSS']
    
   
    scatter1 = ax.scatter(df['FWHM_CIV_S11'],
                          df['FWHM_CIV'],
                          c=cs[1],
                          s=25,
                          edgecolor='None')

    
    ax.plot([1000,15000], [1000,15000], color='black', linestyle='--')
    
    ax.set_xlim(1000,10000)
    ax.set_ylim(ax.get_xlim())

    ax.set_xlabel(r'Shen et al. (2011)~~FWHM(C\,{\sc iv}) [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'FWHM(C\,{\sc iv}) [km~$\rm{s}^{-1}$]')

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/shen_comparison_civ.pdf')

    plt.show()

    return None 



def civ_ha_comparisons_paper1(): 

    set_plot_properties() # change style 

    sys.path.insert(0, '/home/lc585/Dropbox/IoA/BlackHoleMasses')
    from wht_properties_v4 import get_wht_quasars

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    quasars = get_wht_quasars()

    bs_civ = np.array([i.get_blueshift_civ().value for i in quasars.all_quasars()])
    bs_civ_err  = np.array([i.get_blueshift_civ_err().value for i in quasars.all_quasars()])
    sdss_name = np.array([i.sdss_name for i in quasars.all_quasars()])
    fwhm_ha = np.array([i.get_fwhm_ha_corr().value for i in quasars.all_quasars()])
    fwhm_civ = np.array([i.get_fwhm_civ_corr().value for i in quasars.all_quasars()])
    fwhm_ha_err = np.array([i.fwhm_ha_err.value for i in quasars.all_quasars()])
    fwhm_civ_err = np.array([i.fwhm_civ_err.value for i in quasars.all_quasars()])
    sigma_ha = np.array([i.get_sigma_ha_corr().value for i in quasars.all_quasars()])
    sigma_civ = np.array([i.get_sigma_civ_corr().value for i in quasars.all_quasars()])
    sigma_ha_err = np.array([i.sigma_ha_err.value for i in quasars.all_quasars()])
    sigma_civ_err = np.array([i.sigma_civ_err.value for i in quasars.all_quasars()])


    fig, axs = plt.subplots(3, 2,figsize=(figsize(1, vscale=1)), sharex=True) 

    axs[0,0].errorbar(bs_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      fwhm_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      xerr=bs_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      yerr=fwhm_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      linestyle='',
                      marker='o', 
                      markersize=5,
                      markerfacecolor=cs[1],
                      markeredgecolor='None',
                      ecolor=cs[1],
                      capsize=0)
    
    axs[0,1].errorbar(bs_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      fwhm_ha[sdss_name != 'SDSSJ073813.19+271038.1'],
                      xerr=bs_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      yerr=fwhm_ha_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      linestyle='',
                      marker='^', 
                      markersize=5,
                      markerfacecolor=cs[0],
                      markeredgecolor='None',
                      ecolor=cs[0],
                      capsize=0)
    
    axs[1,0].errorbar(bs_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      sigma_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      xerr=bs_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      yerr=sigma_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      linestyle='',
                      marker='o', 
                      markersize=5,
                      markerfacecolor=cs[1],
                      markeredgecolor='None',
                      ecolor=cs[1],
                      capsize=0)
    
    axs[1,1].errorbar(bs_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      sigma_ha[sdss_name != 'SDSSJ073813.19+271038.1'],
                      xerr=bs_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      yerr=sigma_ha_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      linestyle='',
                      marker='^', 
                      markersize=5,
                      markerfacecolor=cs[0],
                      markeredgecolor='None',
                      ecolor=cs[0],
                      capsize=0)
    
    yerr = (fwhm_civ[sdss_name != 'SDSSJ073813.19+271038.1'] / sigma_civ[sdss_name != 'SDSSJ073813.19+271038.1']) * np.sqrt((fwhm_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'] / fwhm_civ[sdss_name != 'SDSSJ073813.19+271038.1'])**2 + (sigma_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'] / sigma_civ[sdss_name != 'SDSSJ073813.19+271038.1'])**2)
    axs[2,0].errorbar(bs_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      fwhm_civ[sdss_name != 'SDSSJ073813.19+271038.1'] / sigma_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      xerr=bs_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      yerr=yerr,
                      linestyle='',
                      marker='o', 
                      markersize=5,
                      markerfacecolor=cs[1],
                      markeredgecolor='None',
                      ecolor=cs[1],
                      capsize=0)
    
    yerr = (fwhm_ha[sdss_name != 'SDSSJ073813.19+271038.1'] / sigma_ha[sdss_name != 'SDSSJ073813.19+271038.1']) * np.sqrt((fwhm_ha_err[sdss_name != 'SDSSJ073813.19+271038.1'] / fwhm_ha[sdss_name != 'SDSSJ073813.19+271038.1'])**2 + (sigma_ha_err[sdss_name != 'SDSSJ073813.19+271038.1'] / sigma_ha[sdss_name != 'SDSSJ073813.19+271038.1'])**2)
    axs[2,1].errorbar(bs_civ[sdss_name != 'SDSSJ073813.19+271038.1'],
                      fwhm_ha[sdss_name != 'SDSSJ073813.19+271038.1'] / sigma_ha[sdss_name != 'SDSSJ073813.19+271038.1'],
                      xerr=bs_civ_err[sdss_name != 'SDSSJ073813.19+271038.1'],
                      yerr=yerr,
                      linestyle='',
                      marker='^', 
                      markersize=5,
                      markerfacecolor=cs[0],
                      markeredgecolor='None',
                      ecolor=cs[0],
                      capsize=0)
    
    
    
    axs[0,0].axvline(1200, color='black', lw=1, linestyle='--')
    axs[0,1].axvline(1200, color='black', lw=1, linestyle='--')
    axs[1,0].axvline(1200, color='black', lw=1, linestyle='--')
    axs[1,1].axvline(1200, color='black', lw=1, linestyle='--')
    axs[2,0].axvline(1200, color='black', lw=1, linestyle='--')
    axs[2,1].axvline(1200, color='black', lw=1, linestyle='--')
     
    axs[0,0].set_ylim(1000,10000)
    axs[0,1].set_ylim(axs[0,0].get_ylim())
    axs[1,0].set_ylim(2500,5500)
    axs[1,1].set_ylim(1500,4500)
    axs[2,0].set_ylim(0.5,3)
    axs[2,1].set_ylim(axs[2,0].get_ylim())

    axs[2,0].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[2,1].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[0,0].set_ylabel(r'FWHM [km~$\rm{s}^{-1}$]')
    axs[1,0].set_ylabel(r'$\sigma$ [km~$\rm{s}^{-1}$]')
    axs[2,0].set_ylabel(r'FWHM/$\sigma$')
     
    axs[0,0].set_title(r'C\,{\sc iv}')
    axs[0,1].set_title(r'H$\alpha$')
    
    fig.tight_layout()  
    
    plt.show() 

    return None 

def civ_comparisons_paper2(): 

    from matplotlib.ticker import MaxNLocator 

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV == 0]
    df = df[df.BAL_FLAG != 1]

    bs_civ = df.Blueshift_CIV_Ha 
    fwhm_civ = df.FWHM_CIV_BEST
    sigma_civ = df.Sigma_CIV_BEST

    fig, axs = plt.subplots(3, 
                            1,
                            figsize=(figsize(0.8, vscale=2)), 
                            sharex=True) 

    axs[0].plot(bs_civ,
                fwhm_civ,
                linestyle='',
                marker='o', 
                markersize=4,
                markerfacecolor=cs[1],
                markeredgecolor='None')
    
    axs[1].plot(bs_civ,
                sigma_civ,
                linestyle='',
                marker='o', 
                markersize=4,
                markerfacecolor=cs[1],
                markeredgecolor='None')
    

    print df.loc[df.Blueshift_CIV_Ha < 1500.0, 'Sigma_CIV_BEST'].median() 
    print df.loc[df.Blueshift_CIV_Ha > 1500.0, 'Sigma_CIV_BEST'].median() 

    axs[2].errorbar(bs_civ,
                    fwhm_civ / sigma_civ,
                    linestyle='',
                    marker='o', 
                    markersize=4,
                    markerfacecolor=cs[1],
                    markeredgecolor='None')


    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid()

    axs[0].set_ylim(500,11000)
    axs[1].set_ylim(1500, 6000)
    axs[2].set_ylim(0.5, 3)


    axs[2].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[0].set_ylabel(r'FWHM [km~$\rm{s}^{-1}$]')
    axs[1].set_ylabel(r'$\sigma$ [km~$\rm{s}^{-1}$]')
    axs[2].set_ylabel(r'FWHM/$\sigma$')

    labels = ['(a)', '(b)', '(c)']


    for i, label in enumerate(labels):

        axs[i].text(0.1, 0.93, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = axs[i].transAxes)

    fig.tight_layout()  

    fig.savefig('/home/lc585/thesis/figures/chapter03/civ_comparisons_paper2.pdf')
    
    plt.show() 

    return None 

def ha_comparisons_paper2(): 

    from matplotlib.ticker import MaxNLocator 

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV == 0]
    df = df[df.BAL_FLAG != 1]


    bs_civ = df.Blueshift_CIV_Ha 
    fwhm_ha = df.FWHM_Broad_Ha
    sigma_ha = df.Sigma_Broad_Ha 

    fig, axs = plt.subplots(3, 
                            1,
                            figsize=(figsize(0.8, vscale=2)), 
                            sharex=True) 

    axs[0].plot(bs_civ,
                fwhm_ha,
                linestyle='',
                marker='o', 
                markersize=4,
                markerfacecolor=cs[1],
                markeredgecolor='None')

    print np.mean(fwhm_ha[bs_civ > 2000.0]), np.std(fwhm_ha[bs_civ > 2000.0])     
    
    axs[1].plot(bs_civ,
                sigma_ha,
                linestyle='',
                marker='o', 
                markersize=4,
                markerfacecolor=cs[1],
                markeredgecolor='None')

    
    axs[2].errorbar(bs_civ,
                    fwhm_ha / sigma_ha,
                    linestyle='',
                    marker='o', 
                    markersize=4,
                    markerfacecolor=cs[1],
                    markeredgecolor='None')

    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid() 


    axs[0].set_ylim(500,11000)
    axs[1].set_ylim(500, 5000)
    axs[2].set_ylim(0.5, 3)


    axs[2].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[0].set_ylabel(r'FWHM [km~$\rm{s}^{-1}$]')
    axs[1].set_ylabel(r'$\sigma$ [km~$\rm{s}^{-1}$]')
    axs[2].set_ylabel(r'FWHM/$\sigma$')

    labels = ['(a)', '(b)', '(c)']


    for i, label in enumerate(labels):

        axs[i].text(0.9, 0.93, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = axs[i].transAxes)
    
    fig.tight_layout()  

    fig.savefig('/home/lc585/thesis/figures/chapter03/ha_comparisons_paper2.pdf')
    
    plt.show() 

    return None 

def hb_comparisons_paper2(): 

    from matplotlib.ticker import MaxNLocator 

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Hb == 0]
    df = df[df.WARN_CIV == 0]
    df = df[df.BAL_FLAG != 1]


    bs_civ = df.Blueshift_CIV_Hb 
    fwhm_hb = df.FWHM_Broad_Hb
    sigma_hb = df.Sigma_Broad_Hb 

    fig, axs = plt.subplots(3, 
                            1,
                            figsize=(figsize(0.8, vscale=2)), 
                            sharex=True) 

    axs[0].plot(bs_civ,
                fwhm_hb,
                linestyle='',
                marker='o', 
                markersize=4,
                markerfacecolor=cs[0],
                markeredgecolor='None')
    
    
    axs[1].plot(bs_civ,
                sigma_hb,
                linestyle='',
                marker='o', 
                markersize=4,
                markerfacecolor=cs[0],
                markeredgecolor='None')

    
    axs[2].errorbar(bs_civ,
                    fwhm_hb / sigma_hb,
                    linestyle='',
                    marker='o', 
                    markersize=4,
                    markerfacecolor=cs[0],
                    markeredgecolor='None')

    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid() 


    axs[0].set_ylim(500,11000)
    axs[1].set_ylim(500, 5000)
    axs[2].set_ylim(0.5, 3)


    axs[2].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[0].set_ylabel(r'FWHM [km~$\rm{s}^{-1}$]')
    axs[1].set_ylabel(r'$\sigma$ [km~$\rm{s}^{-1}$]')
    axs[2].set_ylabel(r'FWHM/$\sigma$')
    
    fig.tight_layout()  

    fig.savefig('/home/lc585/thesis/figures/chapter03/hb_comparisons_paper2.pdf')
    
    plt.show() 

    return None 

def example_spectra(name, line, ax, offset):

    


    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    cs_light = palettable.colorbrewer.qualitative.Pastel1_9.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    instr = df.ix[name, 'INSTR']

    import sys
    sys.path.insert(1, '/home/lc585/Dropbox/IoA/nirspec/python_code')
    
    if instr == 'FIRE': from fit_properties_fire import get_line_fit_props
    if instr == 'GNIRS': from fit_properties_gnirs import get_line_fit_props
    if instr == 'ISAAC': from fit_properties_isaac import get_line_fit_props
    if instr == 'LIRIS': from fit_properties_liris import get_line_fit_props
    if instr == 'NIRI': from fit_properties_niri import get_line_fit_props
    if instr == 'NIRSPEC': from fit_properties_nirspec import get_line_fit_props
    if instr == 'SOFI_JH': from fit_properties_sofi_jh import get_line_fit_props
    if instr == 'SOFI_LC': from fit_properties_sofi_lc import get_line_fit_props
    if instr == 'TRIPLE': from fit_properties_triple import get_line_fit_props
    if instr == 'TRIPLE_S15': from fit_properties_triple_shen15 import get_line_fit_props
    if instr == 'XSHOOT': from fit_properties_xshooter import get_line_fit_props
    if instr == 'SINF': from fit_properties_sinfoni import get_line_fit_props
    if instr == 'SINF_KK': from fit_properties_sinfoni_kurk import get_line_fit_props
    
    q = get_line_fit_props().all_quasars()
    p = q[df.ix[name, 'NUM']]

    if line == 'Ha': w0 = 6564.89*u.AA
    if line == 'Hb': w0 = 4862.721*u.AA
    if (line == 'CIV') | (line == 'CIV_XSHOOTER'): w0 = np.mean([1548.202,1550.774])*u.AA
    
    xs, step = np.linspace(-20000,
                            20000,
                            1000,
                           retstep=True)

    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, line)


    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params = Parameters()
    params.load(parfile)
    parfile.close()

    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav = pickle.load(parfile)
    parfile.close()

    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx = pickle.load(parfile)
    parfile.close()

    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err = pickle.load(parfile)
    parfile.close()

    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()

    vdat = wave2doppler(wav, w0)

    if (line == 'Hb') & (p.hb_model == 'Hb'):

        mod = GaussianModel(prefix='oiii_4959_n_')
    
        mod += GaussianModel(prefix='oiii_5007_n_')
    
        mod += GaussianModel(prefix='oiii_4959_b_')
    
        mod += GaussianModel(prefix='oiii_5007_b_')
    
        if p.hb_narrow is True: 
            mod += GaussianModel(prefix='hb_n_')  
    
        for i in range(p.hb_nGaussians):
    
            mod += GaussianModel(prefix='hb_b_{}_'.format(i))  

         
        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['oiii_5007_n_center'].value
        p1['sigma'].value = params['oiii_5007_n_sigma'].value
        p1['amplitude'].value = params['oiii_5007_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['oiii_4959_n_center'].value
        p1['sigma'].value = params['oiii_4959_n_sigma'].value
        p1['amplitude'].value = params['oiii_4959_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')        

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['oiii_5007_b_center'].value
        p1['sigma'].value = params['oiii_5007_b_sigma'].value
        p1['amplitude'].value = params['oiii_5007_b_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')       

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['oiii_4959_b_center'].value
        p1['sigma'].value = params['oiii_4959_b_sigma'].value
        p1['amplitude'].value = params['oiii_4959_b_amplitude'].value   

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')             

        for i in range(p.hb_nGaussians):
    
            g1 = GaussianModel()
            p1 = g1.make_params()
    
            p1['center'].value = params['hb_b_{}_center'.format(i)].value
            p1['sigma'].value = params['hb_b_{}_sigma'.format(i)].value
            p1['amplitude'].value = params['hb_b_{}_amplitude'.format(i)].value  
        
            ax.plot(np.sort(vdat.value) - offset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=cs_light[4])  
    
        if p.hb_narrow is True: 
    
            g1 = GaussianModel()
            p1 = g1.make_params()
        
            p1['center'] = params['hb_n_center']
            p1['sigma'] = params['hb_n_sigma']
            p1['amplitude'] = params['hb_n_amplitude']   
        
            ax.plot(np.sort(vdat.value) - offset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=cs_light[4],
                    linestyle='-')                    

    if (line == 'Ha') & (p.ha_model == 'Ha'):

        mod = GaussianModel(prefix='ha_n_')  
        mod += GaussianModel(prefix='nii_6548_n_')
        mod += GaussianModel(prefix='nii_6584_n_')
        mod += GaussianModel(prefix='sii_6717_n_')
        mod += GaussianModel(prefix='sii_6731_n_')

        for i in range(p.ha_nGaussians):
            mod += GaussianModel(prefix='ha_b_{}_'.format(i))  

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['ha_n_center'].value
        p1['sigma'].value = params['ha_n_sigma'].value
        p1['amplitude'].value = params['ha_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['nii_6548_n_center'].value
        p1['sigma'].value = params['nii_6548_n_sigma'].value
        p1['amplitude'].value = params['nii_6548_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['nii_6584_n_center'].value
        p1['sigma'].value = params['nii_6584_n_sigma'].value
        p1['amplitude'].value = params['nii_6584_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')

        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['sii_6717_n_center'].value
        p1['sigma'].value = params['sii_6717_n_sigma'].value
        p1['amplitude'].value = params['sii_6717_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-')      
                 
        g1 = GaussianModel()
        p1 = g1.make_params()

        p1['center'].value = params['sii_6731_n_center'].value
        p1['sigma'].value = params['sii_6731_n_sigma'].value
        p1['amplitude'].value = params['sii_6731_n_amplitude'].value

        ax.plot(np.sort(vdat.value) - offset, 
                g1.eval(p1, x=np.sort(vdat.value)),
                c=cs_light[4],
                linestyle='-') 

        for i in range(p.ha_nGaussians):

            g1 = GaussianModel()
            p1 = g1.make_params()

            p1['center'].value = params['ha_b_{}_center'.format(i)].value
            p1['sigma'].value = params['ha_b_{}_sigma'.format(i)].value
            p1['amplitude'].value = params['ha_b_{}_amplitude'.format(i)].value  
    
            ax.plot(np.sort(vdat.value) - offset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=cs_light[4])  


    if ((line == 'CIV') | (line == 'CIV_XSHOOTER')) & (p.civ_model == 'GaussHermite'):

        param_names = []

        for i in range(p.civ_gh_order + 1):
            
            param_names.append('amp{}'.format(i))
            param_names.append('sig{}'.format(i))
            param_names.append('cen{}'.format(i))

        if p.civ_gh_order == 0: 
 
            mod = Model(gausshermite_0, independent_vars=['x'], param_names=param_names) 
    
        if p.civ_gh_order == 1: 
 
            mod = Model(gausshermite_1, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 2: 
 
            mod = Model(gausshermite_2, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 3: 
 
            mod = Model(gausshermite_3, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 4: 
 
            mod = Model(gausshermite_4, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 5: 
 
            mod = Model(gausshermite_5, independent_vars=['x'], param_names=param_names) 

        if p.civ_gh_order == 6: 
 
            mod = Model(gausshermite_6, independent_vars=['x'], param_names=param_names) 

    if (line == 'Ha') & (p.ha_model == 'MultiGauss'):

        mod = ConstantModel()
        
        for i in range(p.ha_nGaussians):
            gmod = GaussianModel(prefix='g{}_'.format(i))
            mod += gmod

        for i in range(p.ha_nGaussians):

            g1 = GaussianModel()
            p1 = g1.make_params()

            p1['center'].value = params['g{}_center'.format(i)].value
            p1['sigma'].value = params['g{}_sigma'.format(i)].value
            p1['amplitude'].value = params['g{}_amplitude'.format(i)].value  
    
            ax.plot(np.sort(vdat.value) - offset, 
                    g1.eval(p1, x=np.sort(vdat.value)),
                    c=cs_light[4])  


    vdat = vdat.value
    
    # ax.errorbar(vdat,
    #             flx,
    #             yerr=err,
    #             linestyle='',
    #             color='black',
    #             lw=1,
    #             alpha=1)

    ax.plot(vdat - offset,
            flx,
            linestyle='-',
            color='grey',
            lw=1,
            alpha=1,
            zorder=5)

    ax.plot(xs - offset,
            mod.eval(params=params, x=xs/sd) ,
            color='black',
            lw=1,
            zorder=6)

    ax.axhline(0.0, color='black', linestyle=':')

    # ax.set_xlim(-10000,12000)
    # ax.set_ylim(0, 3.0)

    # ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    # ax.set_ylabel(r' F$_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]')

    # fig.tight_layout()

    # fig.savefig(name + '_' + line + '.pdf')
    # plt.show() 

    return None 

def example_residual(name, line, ax):

    from lmfit import Model

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    instr = df.ix[name, 'INSTR']

    import sys
    sys.path.insert(1, '/home/lc585/Dropbox/IoA/nirspec/python_code')
    
    if instr == 'FIRE': from fit_properties_fire import get_line_fit_props
    if instr == 'GNIRS': from fit_properties_gnirs import get_line_fit_props
    if instr == 'ISAAC': from fit_properties_isaac import get_line_fit_props
    if instr == 'LIRIS': from fit_properties_liris import get_line_fit_props
    if instr == 'NIRI': from fit_properties_niri import get_line_fit_props
    if instr == 'NIRSPEC': from fit_properties_nirspec import get_line_fit_props
    if instr == 'SOFI_JH': from fit_properties_sofi_jh import get_line_fit_props
    if instr == 'SOFI_LC': from fit_properties_sofi_lc import get_line_fit_props
    if instr == 'TRIPLE': from fit_properties_triple import get_line_fit_props
    if instr == 'TRIPLE_S15': from fit_properties_triple_shen15 import get_line_fit_props
    if instr == 'XSHOOT': from fit_properties_xshooter import get_line_fit_props
    if instr == 'SINF': from fit_properties_sinfoni import get_line_fit_props
    if instr == 'SINF_KK': from fit_properties_sinfoni_kurk import get_line_fit_props
    
    q = get_line_fit_props().all_quasars()
    p = q[df.ix[name, 'NUM']]

    if line == 'Ha': w0 = 6564.89*u.AA
    if line == 'Hb': w0 = 4862.721*u.AA
    if (line == 'CIV') | (line == 'CIV_XSHOOTER'): w0 = np.mean([1548.202,1550.774])*u.AA

    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', name, line)

    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params = Parameters()
    params.load(parfile)
    parfile.close()

    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav = pickle.load(parfile)
    parfile.close()

    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx = pickle.load(parfile)
    parfile.close()

    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err = pickle.load(parfile)
    parfile.close()

    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()
  
    vdat = wave2doppler(wav, w0)

    if (line == 'Hb') & (p.hb_model == 'Hb'):

        mod = GaussianModel(prefix='oiii_4959_n_')
    
        mod += GaussianModel(prefix='oiii_5007_n_')
    
        mod += GaussianModel(prefix='oiii_4959_b_')
    
        mod += GaussianModel(prefix='oiii_5007_b_')
    
        if p.hb_narrow is True: 
            mod += GaussianModel(prefix='hb_n_')  
    
        for i in range(p.hb_nGaussians):
            mod += GaussianModel(prefix='hb_b_{}_'.format(i))  
         
      
    if (line == 'Ha') & (p.ha_model == 'Ha'):

        mod = GaussianModel(prefix='ha_n_')  
        mod += GaussianModel(prefix='nii_6548_n_')
        mod += GaussianModel(prefix='nii_6584_n_')
        mod += GaussianModel(prefix='sii_6717_n_')
        mod += GaussianModel(prefix='sii_6731_n_')

        for i in range(p.ha_nGaussians):
            mod += GaussianModel(prefix='ha_b_{}_'.format(i))  

        
    if ((line == 'CIV') | (line == 'CIV_XSHOOTER')) & (p.civ_model == 'GaussHermite'):

        param_names = []

        for i in range(p.civ_gh_order + 1):
            
            param_names.append('amp{}'.format(i))
            param_names.append('sig{}'.format(i))
            param_names.append('cen{}'.format(i))

        if p.civ_gh_order == 0: 
 
            mod = Model(gausshermite_0, independent_vars=['x'], param_names=param_names) 
    
        if p.civ_gh_order == 1: 
 
            mod = Model(gausshermite_1, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 2: 
 
            mod = Model(gausshermite_2, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 3: 
 
            mod = Model(gausshermite_3, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 4: 
 
            mod = Model(gausshermite_4, independent_vars=['x'], param_names=param_names) 
     
        if p.civ_gh_order == 5: 
 
            mod = Model(gausshermite_5, independent_vars=['x'], param_names=param_names) 

        if p.civ_gh_order == 6: 
 
            mod = Model(gausshermite_6, independent_vars=['x'], param_names=param_names) 

    if (line == 'Ha') & (p.ha_model == 'MultiGauss'):

        mod = ConstantModel()
        
        for i in range(p.ha_nGaussians):
            gmod = GaussianModel(prefix='g{}_'.format(i))
            mod += gmod

    ax.plot(vdat,
            (flx - mod.eval(params=params, x=vdat.value/sd)) / err,
            color=cs[8],
            lw=1)


    ax.axhline(0.0, color='black', linestyle=':')


    return None 

def example_spectrum_grid():

    set_plot_properties() # change style  

    fig = plt.figure(figsize=figsize(1, vscale=1.5))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.1)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    
    names = ['QSO016',  # GNIRS                      
             'QSO025',  # GNIRS              
             'QSO399',  # SOFI                             
             'QSO128']  # P200/TRIPLESPEC    

    # from python_code/get_snr 

    snr = [[18, 22, 9],
           [22, 16, 13],
           [14, 8, 27],
           [6, 9, 18]]

    ylims = [[[-0.04, 4.5], [-0.04, 2.5], [-0.04, 6.5]],
             [[-0.04, 3.0], [-0.04, 1.0], [-0.04, 2.0]],
             [[-0.04, 5.0], [-0.04, 2.5], [-0.04, 3.5]],
             [[-0.04, 5.5], [-0.04, 2.5], [-0.04, 4.5]]]

    titles = ['J121427.77-030721.0',
              'J231441.63-082406.8',
              'J121140.59+103002.0',
              'J094206.95+352307.4']



    for i in range(4):

        offset = df.ix[names[i], 'Median_Broad_Ha']
        
        inner_grid = gridspec.GridSpecFromSubplotSpec(12, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        ax = plt.Subplot(fig, inner_grid[:3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], 'Ha', ax, offset)
        ax.set_ylim(ylims[i][0])
        ax.set_xlim(-12000, 12000)
        ax.set_title(titles[i], fontsize=10)
        ax.text(0.05, 0.8, 'S/N: {}'.format(snr[i][0]), transform=ax.transAxes, zorder=10)
        fig.add_subplot(ax)

        if (i == 0) | (i == 2):
            ax.text(0.05, 0.5, r'H$\alpha$', transform= ax.transAxes)

        ax = plt.Subplot(fig, inner_grid[3])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], 'Ha', ax)
        ax.set_xlim(-12000, 12000)
        ax.set_ylim(-8, 8)
        
        if (i == 0) | (i == 2):
            ax.set_yticks([-8,0,8])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])

        fig.add_subplot(ax)



        #--------------------------------------------------------

        ax = plt.Subplot(fig, inner_grid[4:7])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], 'Hb', ax, offset)
        ax.set_ylim(ylims[i][1])
        ax.set_xlim(-12000, 12000)
        ax.text(0.05, 0.8, 'S/N: {}'.format(snr[i][1]), transform=ax.transAxes, zorder=10)
        fig.add_subplot(ax)

        if (i == 0) | (i == 2):
            ax.text(0.05, 0.5, r'H$\beta$', transform= ax.transAxes)

        ax = plt.Subplot(fig, inner_grid[7])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], 'Hb', ax)
        ax.set_xlim(-12000, 12000)
        ax.set_ylim(-8, 8)
        if (i == 0) | (i == 2):
            ax.set_yticks([-8,0,8])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])
        fig.add_subplot(ax)

        #------------------------------------------------------

        ax = plt.Subplot(fig, inner_grid[8:11])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        example_spectra(names[i], 'CIV', ax, offset)
        ax.set_ylim(ylims[i][2])
        ax.set_xlim(-12000, 12000)
        ax.text(0.05, 0.8, 'S/N: {}'.format(snr[i][2]), transform=ax.transAxes, zorder=10)
        fig.add_subplot(ax)

        if (i == 0) | (i == 2):
            ax.text(0.05, 0.5, r'C\,{\sc iv}', transform= ax.transAxes)

        ax = plt.Subplot(fig, inner_grid[11])
        
        if i < 2:
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')

        ax.spines['top'].set_visible(False)
        example_residual(names[i], 'CIV', ax)
        ax.set_xlim(-12000, 12000)
        ax.set_ylim(-8, 8)
        if (i == 0) | (i == 2):
            ax.set_yticks([-8,0,8])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])


        fig.add_subplot(ax)


    fig.text(0.50, 0.05, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'$F_{\lambda}$ [Arbitrary units]', rotation=90)

    fig.savefig('/home/lc585/thesis/figures/chapter03/example_spectrum_grid.pdf')

 
    plt.show() 


    return None 



def ha_hb_composite():

    from SpectraTools.get_nir_spec import get_nir_spec
    from SpectraTools.make_composite import make_composite
    from SpectraTools.fit_line import fit_line

    set_plot_properties() # change style  

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.8))

    # Hb --------------------------------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.OIII_FIT_HB_Z_FLAG > 0 ] 
    df = df[df.OIII_FIT_VEL_HB_PEAK_ERR < 600.0] # Really bad 
    df = df[df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0] # Really bad 
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EXTREM_FLAG == 0]
 
    wav_new = np.arange(4700.0, 5100.0, 1.0) 

    flux_array = []
    wav_array = [] 
    z_array = [] 
    name_array = [] 
        
    for idx, row in df.iterrows():

        save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', idx, 'OIII') 

        wav, flux = np.genfromtxt(os.path.join(save_dir, 'spec_cont_sub.txt'), unpack=True)

        wav = wav * (1.0 + row.z_IR_OIII_FIT)

        z = row.OIII_FIT_HB_Z
        if row.OIII_Z_FLAG == 1:
            z = row.OIII_FIT_Z_FULL_OIII_PEAK

        flux_array.append(flux)
        wav_array.append(wav)
        z_array.append(z)
        name_array.append(idx)

    wav_array = np.array(wav_array)
    flux_array = np.array(flux_array)
    z_array = np.array(z_array)


    wav_new, flux, err, ns  = make_composite(wav_new,
                                             wav_array, 
                                             flux_array, 
                                             z_array,
                                             names=name_array,
                                             verbose=False)

    vdat = wave2doppler(wav_new*u.AA, w0=4862.721*u.AA) 

    ax.plot(vdat, flux / np.nanmax(flux), color=cs[0], label=r'H$\beta$ + [OIII]')


    # Ha --------------------------------------------------------------------------------------
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.OIII_FIT_HA_Z_FLAG > 0] 
    df = df[df.OIII_FIT_VEL_HA_PEAK_ERR < 400.0] # Really bad 
    df = df[df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0] # Really bad 

    wav_new = np.arange(6400.0, 6800.0, 1.0) 

    flux_array = []
    wav_array = [] 
    z_array = [] 
    name_array = []  
        
    for idx, row in df.iterrows():

        save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', idx, 'Ha_z') 

        wav, flux = np.genfromtxt(os.path.join(save_dir, 'spec_cont_sub.txt'), unpack=True)

        wav = wav * (1.0 + row.z_IR)
       
        flux_array.append(flux)
        wav_array.append(wav)
        z_array.append(row.OIII_FIT_HA_Z)
        name_array.append(idx)

    wav_array = np.array(wav_array)
    flux_array = np.array(flux_array)
    z_array = np.array(z_array)

    wav_new, flux, err, ns  = make_composite(wav_new,
                                             wav_array, 
                                             flux_array, 
                                             z_array,
                                             names=name_array,
                                             verbose=False)

    vdat = wave2doppler(wav_new*u.AA, w0=6564.89*u.AA) 

    ax.plot(vdat, flux / np.nanmax(flux), color=cs[1], label=r'H$\alpha$')

    ax.legend(fancybox=True)

    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'$F_{\lambda}$ [Arbitrary units]')
    plt.grid()

    fig.tight_layout() 
    
    fig.savefig('/home/lc585/thesis/figures/chapter03/ha_hb_composite.pdf')

    plt.show() 

    return None 


def ha_edd_civ_bs(): 

    """
    Plot of Ha Eddington ratio against CIV blueshift
    """

    set_plot_properties() # change style  

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    # Try without civ constraints 

    fig, ax = plt.subplots(figsize=(figsize(0.9, vscale=0.7)))
    
    ax.plot(df.Blueshift_CIV_Ha,
            df.Edd_Ratio_Ha,
            linestyle='',
            marker='o',
            markerfacecolor=cs[1],
            markeredgecolor=cs[1],
            markersize=5)
    

    
    ax.set_yscale('log')
    
    ax.set_ylim(0.05, 3)
    ax.set_xlim(-1000, 6000)
    
    ax.set_xlabel(r'C\,{\sc iv} Blueshift [km~${\rm s}^{-1}$]')
    ax.set_ylabel( r'$L_{\rm Bol} / L_{\rm Edd} ({\rm H}\alpha)$' )
    
    ax.grid() 
    fig.tight_layout()

    plt.show() 
    
    fig.savefig('/home/lc585/thesis/figures/chapter03/ha_edd_civ_bs.pdf')

    return None 


def civ_space_plot():

    """
    Plot all objects. 
    Exclude BALs. 
    Use my parameteric measure of the CIV blueshift, since otherwise won't have everything. 
    Use peak of broad Ha / Hb to give systemic redshift. 
    """

    set_plot_properties() # change style   

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.0
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # no labels
    nullfmt = NullFormatter() 

    fig = plt.figure(figsize=figsize(1.0, vscale=1.0))

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

    t_ica = Table.read('/data/vault/phewett/LiamC/liam_civpar_zica_160115.dat', format='ascii') # new ICA 
        
    m1, m2 = t_ica['col2'], np.log10( t_ica['col3'])

    badinds = np.isnan(m1) | np.isnan(m2) | np.isinf(m1) | np.isinf(m2)

    m1 = m1[~badinds]
    m2 = m2[~badinds]

    xmin = -1000.0
    xmax = 3500.0
    ymin = 1.0
    ymax = 2.5

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    CS = axScatter.contour(X,Y,Z, colors=[cs[-1]])

    threshold = CS.levels[0]

    z = kernel(values)

    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axScatter.scatter(x, y, c=cs[-1], edgecolor='None', s=3, label='SDSS DR7' )

    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[df1.WARN_CIV == 0]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[df2.WARN_CIV == 0]
    df2 = df2[df2.BAL_FLAG != 1] 
        
    df = pd.concat([df1, df2]).drop_duplicates()

    axScatter.scatter(df.Blueshift_CIV_Balmer_Best,
                      np.log10(df.EQW_CIV_BEST),
                      c=cs[1],
                      s=20,
                      edgecolor='None',
                      label = 'This work',
                      zorder=10)
   
    
    axScatter.set_xlim(-2000, 6000)
    axScatter.set_ylim(1,2.2)

    axScatter.set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axScatter.set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')

    legend = axScatter.legend(frameon=True, scatterpoints=1) 

    axHistx.hist(m1, 
                 bins=np.arange(-2000, 6000, 500), 
                 facecolor=cs[-1], 
                 edgecolor='None', 
                 alpha=0.4, 
                 normed=True)

    axHisty.hist(m2, 
                 bins=np.arange(1, 2.2, 0.1), 
                 facecolor=cs[-1], 
                 edgecolor='None', 
                 orientation='horizontal', 
                 alpha=0.4, 
                 normed=True)

    axHistx.hist(df.Blueshift_CIV_Balmer_Best, 
                 bins=np.arange(-2000, 6000, 500), 
                 histtype='step', 
                 edgecolor=cs[1], 
                 normed=True, 
                 lw=2)

    axHisty.hist(np.log10(df.EQW_CIV_BEST), 
                 bins=np.arange(1, 2.2, 0.1), 
                 histtype='step', 
                 edgecolor=cs[1], 
                 normed=True, 
                 lw=2, 
                 orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    fig.savefig('/home/lc585/thesis/figures/chapter03/civ_space.pdf')

    plt.show()

    return None 

def civ_comparison():

    from lmfit import Parameters
    import cPickle as pickle 
    from SpectraTools.fit_line import wave2doppler
    import astropy.units as u 
    from lmfit.models import GaussianModel, ConstantModel 
    from SpectraTools.fit_line import gausshermite_3
 

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors
    
    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.6))
    
    xs, step = np.linspace(-20000,
                           20000,
                           1000,
                           retstep=True)
    
    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/','QSO442','CIV')
    
    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params_CIV = Parameters()
    params_CIV.load(parfile)
    parfile.close()
    
    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav_CIV = pickle.load(parfile)
    parfile.close()
    
    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx_CIV = pickle.load(parfile)
    parfile.close()
    
    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err_CIV = pickle.load(parfile)
    parfile.close()
    
    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()


    param_names = []

    for i in range(4):
        
        param_names.append('amp{}'.format(i))
        param_names.append('sig{}'.format(i))
        param_names.append('cen{}'.format(i))

 
    mod = Model(gausshermite_3, independent_vars=['x'], param_names=param_names) 
   
    norm = np.sum(mod.eval(params=params_CIV, x=xs/sd)) / 200.0
    
    line1, = ax.plot(xs - 38.0,
                     mod.eval(params=params_CIV, x=xs/sd) / norm,
                     color = 'black',
                     linestyle='--',
                     lw=2,
                     label='J123611+112922')

    save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/','QSO437','CIV')
    
    parfile = open(os.path.join(save_dir,'my_params.txt'), 'r')
    params_CIV = Parameters()
    params_CIV.load(parfile)
    parfile.close()
    
    wav_file = os.path.join(save_dir, 'wav.txt')
    parfile = open(wav_file, 'rb')
    wav_CIV = pickle.load(parfile)
    parfile.close()
    
    flx_file = os.path.join(save_dir, 'flx.txt')
    parfile = open(flx_file, 'rb')
    flx_CIV = pickle.load(parfile)
    parfile.close()
    
    err_file = os.path.join(save_dir, 'err.txt')
    parfile = open(err_file, 'rb')
    err_CIV = pickle.load(parfile)
    parfile.close()
    
    sd_file = os.path.join(save_dir, 'sd.txt')
    parfile = open(sd_file, 'rb')
    sd = pickle.load(parfile)
    parfile.close()

    param_names = []

    for i in range(5):
        
        param_names.append('amp{}'.format(i))
        param_names.append('sig{}'.format(i))
        param_names.append('cen{}'.format(i))

 
    mod = Model(gausshermite_4, independent_vars=['x'], param_names=param_names) 

    norm = np.sum(mod.eval(params=params_CIV, x=xs/sd)) / 200.0

    line1, = ax.plot(xs + 357.,
                     mod.eval(params=params_CIV, x=xs/sd) / norm,
                     color = 'black',
                     lw=2,
                     label = 'J152529+292813')
    
    
    ax.set_xlim(-20000,20000)
    # ax.set_ylim(-0.05,0.8) 
    
    plt.legend(loc='upper right') 

    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'F$_{\lambda}$ [Arbitrary units]')

    ax.axvline(0.0, color='grey', lw=1, linestyle='--')

    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter03/civ_comparison.pdf')
    
    plt.show() 

    return None 


def ha_hb_width_comparison():

    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    fig, ax = plt.subplots(figsize=figsize(1, vscale=1))

    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_Hb == 0]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    xi = df.FWHM_Broad_Ha.apply(np.log10)
    dxi = df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha / np.log(10)  
    yi = df.FWHM_Broad_Hb.apply(np.log10)
    dyi = df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb / np.log(10)  

    print np.std(xi - yi)

    ax.errorbar(xi, yi, xerr=dxi, yerr=dyi, linestyle='', color='grey', alpha=0.4, zorder=2)
    ax.scatter(xi, yi, color=cs[1], s=8, zorder=3)

    logx = np.linspace(3.2, 4, 50)
    logy = np.log10(1.07) - 0.09 + 1.03*logx
    ax.plot(logx, logy, c='black', label='Green \& Ho', linestyle='--')

  
    # ----------------Plot fit--------------------------------

    trace = np.load('/data/lc585/BHMassPaper2_Resubmitted_MCMC_Traces/trace_ha_hb_relation.npy')
   
    m, b = trace[:2]
    # xfit = np.linspace(xdata.min(), xdata.max(), 10)
    yfit = b[:, None] + m[:, None] * (logx - 3.0) # 3 because divided x by 10**3 in model
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)
    trace = None 

    print np.std(yi - (np.median(b[:, None]) + np.median(m[:, None]) * (xi.values - 3.0)) )

    ax.plot(logx, mu, 'k', linestyle='-', zorder=5, label='This work')
    ax.fill_between(logx, mu - sig, mu + sig, color=palettable.colorbrewer.qualitative.Pastel1_6.mpl_colors[1], zorder=1)

    #--------------------------------------------------------

    plt.legend(loc='lower right', handlelength=2.5, frameon=False)

    ax.set_xlim(3.3, 4)
    ax.set_ylim(ax.get_xlim())

    ax.set_xlabel(r'log FWHM H$\alpha$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'log FWHM H$\beta$ [km~$\rm{s}^{-1}$]')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter03/ha_hb_width_comparison.pdf')

    plt.show()

    return None 



def civ_ha_bhm_comparison():

    from matplotlib.ticker import FormatStrFormatter

    cs = palettable.colorbrewer.qualitative.Set1_9.mpl_colors 

    fig, axs = plt.subplots(2, 1, figsize=figsize(0.8, vscale=1.6), sharex=True)
     
    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    # uncorrected mass comparison 
    xi = df.LogMBH_Ha.values
    yi = df.LogMBH_CIV_VP06.values
    dxi = df.LogMBH_Ha_Err.values
    dyi = df.LogMBH_CIV_VP06_Err.values

    print np.std(xi - yi) 


    axs[0].scatter(xi, 
                   yi, 
                   s=15,
                   edgecolor='None',
                   color='black',
                   zorder=2)

    xmin = 8.5 
    xmax = 10.5
    ymin = 8.5
    ymax = 10.5

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([xi, yi])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    axs[0].imshow(np.flipud(Z.T), 
                  extent=(xmin, xmax, ymin, ymax), 
                  aspect='auto', 
                  zorder=0, 
                  cmap='Blues')

    axs[0].plot([xmin, xmax], 
                [ymin, ymax], 
                color='black', 
                zorder=1, 
                linestyle='--')


    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)

    axs[0].set_ylabel(r'log BHM C\,{\sc iv} [M$_\odot$]')

    axs[0].text(0.1, 0.9, '(a)', transform= axs[0].transAxes)
    
    #-----------------------------------------------------------------
    # corrected mass comparison  

    xi = df.LogMBH_Ha.values 
    dxi = df.LogMBH_Ha_Err.values 

    fwhm = df['FWHM_Broad_Ha'] * 1.e-3 
    fwhm_err = df['FWHM_Broad_Ha_Err'] * 1.e-3 
        
    fwhm_hb = 1.23e3 * np.power(fwhm, 0.97)
    fwhm_hb_err = 1.23e3 * np.power(fwhm, 0.97-1.0) * 0.97 * fwhm_err

    m, b = 0.41, 0.62
   
    fwhm_ha = df.FWHM_CIV_BEST / (m * df.Blueshift_CIV_Ha * 1e-3 + b)

    fwhm_ha = fwhm_ha * 1e-3   
    l1350 = 10**(df['LogL1350'].values) * 1e-44

    p1 = np.power(10, 6.71)
    p2 = np.power(fwhm_ha, 2)
    p3 = np.power(l1350, 0.53)

    yi = np.log10(p1 * p2 * p3)

    print np.std(xi - yi) 

    axs[1].scatter(xi, 
                   yi, 
                   s=15,
                   edgecolor='None',
                   color='black',
                   zorder=2)

    xmin = 8.5 
    xmax = 10.5
    ymin = 8.5
    ymax = 10.5

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([xi, yi])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    axs[1].imshow(np.flipud(Z.T), 
                  extent=(xmin, xmax, ymin, ymax), 
                  aspect='auto', 
                  zorder=0, 
                  cmap='Blues')

    axs[1].plot([xmin, xmax], 
                [ymin, ymax], 
                color='black', 
                zorder=1, 
                linestyle='--')


    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax)

    axs[1].set_xlabel(r'log BHM H$\alpha$ [M$_\odot$]')
    axs[1].set_ylabel(r'log BHM C\,{\sc iv} (Corrected) [M$_\odot$]')  
    
    axs[1].text(0.1, 0.9, '(b)', transform= axs[1].transAxes)


    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)


    fig.savefig('/home/lc585/thesis/figures/chapter03/bhm_comparison.pdf')

    plt.show() 


    return None 

def dispersion_comparison():

    fig, ax = plt.subplots(1, 1, figsize=figsize(1, vscale=0.9))
     
    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]

    ax.scatter(df.Sigma_Broad_Ha, 
               df.Sigma_CIV_BEST, 
               s=15,
               edgecolor='None',
               color='black',
               zorder=2)

    from scipy.stats import spearmanr
    print spearmanr(df.Sigma_Broad_Ha, df.Sigma_CIV_BEST)
    

    m1, m2 = df.Sigma_Broad_Ha, df.Sigma_CIV_BEST
    xmin = 800.0
    xmax = 5000.0
    ymin = 800.0
    ymax = 5000.0

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])

    kernel = stats.gaussian_kde(values)
      
    Z = np.reshape(kernel(positions).T, X.shape)

    # cs = ax.contourf(X, Y, Z, 100, hold='on', cmap='Blues', zorder=0)

    # for c in cs.collections:
    #     c.set_edgecolor("face")

    # cs = ax.contour(X, Y, Z, 5, hold='on', colors='black', zorder=1)

    ax.imshow(np.flipud(Z.T), extent=(xmin, xmax, ymin, ymax), aspect='auto', zorder=0, cmap='Blues')

    # ax.plot([xmin, xmax], [ymin, ymax], color='black', zorder=1, linestyle='--')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r'$\sigma$ H$\alpha$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'$\sigma$ C\,{\sc iv} [km~$\rm{s}^{-1}$]')

    ax.xaxis.set_ticks([1000, 2000, 3000, 4000, 5000])
    ax.yaxis.set_ticks([1000, 2000, 3000, 4000, 5000])

    # ax.text(0.8, 
    #         0.1, 
    #         r'$\rho$ = {0:.2f}'.format(pearsonr(df.Sigma_Broad_Ha, df.Sigma_CIV_BEST)[0]), 
    #         fontsize=12,
    #         transform= ax.transAxes)
    
    fig.tight_layout() 

    fig.savefig('/home/lc585/thesis/figures/chapter03/dispersion_comparison.pdf')

    plt.show() 

    return None 


def bal_composite():

    from SpectraTools.get_nir_spec import get_nir_spec
    from SpectraTools.make_composite import make_composite
    from SpectraTools.fit_line import fit_line, doppler2wave

    set_plot_properties() # change style  

    cs = palettable.colorbrewer.qualitative.Set2_3.mpl_colors

    fig, ax = plt.subplots(figsize=figsize(1, vscale=0.9))
    fig.tight_layout() 

    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15)

    wav_new = np.arange(4700.0, 5100.0, 1.0) 

    # Hb --------------------------------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.BAL_FLAG == 1]
    df = df[(df.WARN_Hb == 0) | (df.WARN_Hb == 1)]
    df.drop('QSO179', inplace=True) # Duplicated 

    df['z'] = np.nan

    useoiii = (df.OIII_EQW_FLAG == 0) & (df.OIII_EXTREM_FLAG == 0) & (df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0)
    df.loc[useoiii, 'z'] = df.loc[useoiii, 'OIII_FIT_Z_FULL_OIII_PEAK'] 
    
    useha = df.z.isnull() & (df.OIII_FIT_HA_Z_FLAG > 0) & (df.OIII_FIT_VEL_HA_PEAK_ERR < 400.0)
    df.loc[useha, 'z'] = df.loc[useha, 'OIII_FIT_HA_Z'] 
        
    usehb = df.z.isnull() & (df.OIII_FIT_HB_Z_FLAG >= 0) & (df.OIII_FIT_VEL_HB_PEAK_ERR < 750.0)
    df.loc[usehb, 'z'] = df.loc[usehb, 'OIII_FIT_HB_Z']  

    flux_array, wav_array, z_array, name_array = [], [], [], [] 

    # deredshift spectra 
    for idx, row in df.iterrows():
        save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', idx, 'OIII') 
        wav, flux = np.genfromtxt(os.path.join(save_dir, 'spec_cont_sub.txt'), unpack=True)
        wav = wav * (1.0 + row.z_IR_OIII_FIT)

        flux_array.append(flux)
        wav_array.append(wav)
        z_array.append(row.z)
        name_array.append(idx)

    wav_array = np.array(wav_array)
    flux_array = np.array(flux_array)
    z_array = np.array(z_array)

    wav_new, flux, err, ns  = make_composite(wav_new,
                                             wav_array, 
                                             flux_array, 
                                             z_array,
                                             names=name_array,
                                             verbose=False)

    vdat = wave2doppler(wav_new*u.AA, w0=4862.721*u.AA) 

    ax.plot(vdat, flux / np.nanmax(flux), color=cs[0], label=r'BAL')

    # --------------------------------------------------------------------

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.OIII_FIT_HB_Z_FLAG > 0 ] 
    df = df[df.OIII_BAD_FIT_FLAG == 0]
    df = df[df.FE_FLAG == 0]
    df = df[df.OIII_EXTREM_FLAG == 0]
    df = df[(df.WARN_CIV_BEST == 0) | (df.WARN_CIV_BEST == 1)]
    df = df[df.BAL_FLAG != 1]
   
    df['z'] = np.nan

    useoiii = (df.OIII_EQW_FLAG == 0) & (df.OIII_EXTREM_FLAG == 0) & (df.OIII_FIT_VEL_FULL_OIII_PEAK_ERR < 400.0)
    df.loc[useoiii, 'z'] = df.loc[useoiii, 'OIII_FIT_Z_FULL_OIII_PEAK'] 
    
    useha = df.z.isnull() & (df.OIII_FIT_HA_Z_FLAG > 0) & (df.OIII_FIT_VEL_HA_PEAK_ERR < 400.0)
    df.loc[useha, 'z'] = df.loc[useha, 'OIII_FIT_HA_Z'] 
        
    usehb = df.z.isnull() & (df.OIII_FIT_HB_Z_FLAG >= 0) & (df.OIII_FIT_VEL_HB_PEAK_ERR < 750.0)
    df.loc[usehb, 'z'] = df.loc[usehb, 'OIII_FIT_HB_Z'] 

    w0 = np.mean([1548.202,1550.774])*u.AA  
    median_wav = doppler2wave(df.Median_CIV_BEST.values*(u.km/u.s), w0) * (1.0 + df.z_IR.values)
    blueshift_civ = const.c.to('km/s') * (w0 - median_wav / (1.0 + df.z)) / w0

    df1 = df[blueshift_civ.value < 1000.0]
    df2 = df[blueshift_civ.value > 1500.0]

    flux_array, wav_array, z_array, name_array = [], [], [], [] 

    # deredshift spectra 
    for idx, row in df.iterrows():
        save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', idx, 'OIII') 
        wav, flux = np.genfromtxt(os.path.join(save_dir, 'spec_cont_sub.txt'), unpack=True)
        wav = wav * (1.0 + row.z_IR_OIII_FIT)

        flux_array.append(flux)
        wav_array.append(wav)
        z_array.append(row.z)
        name_array.append(idx)

    wav_array = np.array(wav_array)
    flux_array = np.array(flux_array)
    z_array = np.array(z_array)

    wav_new, flux, err, ns  = make_composite(wav_new,
                                             wav_array, 
                                             flux_array, 
                                             z_array,
                                             names=name_array,
                                             verbose=False)

    vdat = wave2doppler(wav_new*u.AA, w0=4862.721*u.AA) 

    ax.plot(vdat, flux / np.nanmax(flux), color=cs[1], label=r'non-BAL, C\,{\sc iv} Blueshift $<$ 1000 km~$\rm{s}^{-1}$')

    # # ----------------------------------

    flux_array, wav_array, z_array, name_array = [], [], [], [] 

    # deredshift spectra 
    for idx, row in df2.iterrows():
        save_dir = os.path.join('/data/lc585/nearIR_spectra/linefits/', idx, 'OIII') 
        wav, flux = np.genfromtxt(os.path.join(save_dir, 'spec_cont_sub.txt'), unpack=True)
        wav = wav * (1.0 + row.z_IR_OIII_FIT)

        flux_array.append(flux)
        wav_array.append(wav)
        z_array.append(row.z)
        name_array.append(idx)

    wav_array = np.array(wav_array)
    flux_array = np.array(flux_array)
    z_array = np.array(z_array)

    wav_new, flux, err, ns  = make_composite(wav_new,
                                             wav_array, 
                                             flux_array, 
                                             z_array,
                                             names=name_array,
                                             verbose=False)

    vdat = wave2doppler(wav_new*u.AA, w0=4862.721*u.AA) 

    ax.plot(vdat, flux / np.nanmax(flux), color=cs[2], label=r'non-BAL, C\,{\sc iv} Blueshift $>$ 1500 km~$\rm{s}^{-1}$')

    

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True)

    ax.set_xlim(-10000, 12000)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'$F_{\lambda}$ [Arbitrary units]')
    ax.grid()

    
    fig.savefig('/home/lc585/thesis/figures/chapter03/bal_composite.pdf')

    plt.show() 

    return None 




