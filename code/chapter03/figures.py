from astropy.table import Table, join 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import cPickle as pickle
import os
import time
import numpy.ma as ma 
from PlottingTools.plot_setup import figsize, set_plot_properties
import palettable 
import matplotlib.patches as patches
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
import pandas as pd 

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

def civ_space_plot():

    set_plot_properties() # change style 

    cs = palettable.colorbrewer.qualitative.Set2_8.mpl_colors

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

    CS = axs[1].contour(X,Y,Z, colors=[cs[-1]])

    threshold = CS.levels[0]

    z = kernel(values)

    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axs[1].scatter(x, y, c='grey', edgecolor='None', s=3, label='SDSS DR7', rasterized=True)

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

    CS = axs[2].contour(X,Y,Z, colors=[cs[-1]])

    threshold = CS.levels[0]

    z = kernel(values)

    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axs[2].scatter(x, y, c='grey', edgecolor='None', s=3, label='SDSS DR7', rasterized=True)

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
    
    CS = axs[0].contour(X, Y, Z, colors=[cs[-1]])
    
    threshold = CS.levels[0]
    
    z = kernel(values)
    
    # mask points above density threshold
    x = np.ma.masked_where(z > threshold, m1)
    y = np.ma.masked_where(z > threshold, m2)

    # plot unmasked points
    axs[0].scatter(x, y, c='grey', edgecolor='None', s=3, label='SDSS DR7', rasterized=True)

    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_ylim(axs[1].get_ylim())

    axs[2].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')
    axs[0].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')
    axs[1].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')
    axs[2].set_ylabel(r'log(C\,{\sc iv} EW) [\AA]')

    import sys 
    sys.path.insert(0, '/home/lc585/Dropbox/IoA/BlackHoleMasses')
    from wht_properties_v3 import get_wht_quasars

    quasars = get_wht_quasars().all_quasars()

    t1 = Table()
    t1['SDSS_NAME'] = [i.sdss_name for i in quasars] 

    t2 = Table()
    t2['SDSS_NAME'] = t_hw['col1']
    t2['Blueshift'] = t_hw['col2']
    t2['EQW'] = t_hw['col3']

    t3 = join(t1, t2, keys='SDSS_NAME', join_type='left')

    axs[1].scatter(t3['Blueshift'],
    	           np.log10(t3['EQW']),
    	           c=cs[1],
    	           s=25,
    	           edgecolor='black',
    	           label = 'Our Sample')

    t4 = Table()
    t4['SDSS_NAME'] = t_ica['col1']
    t4['Blueshift'] = t_ica['col2']
    t4['EQW'] = t_ica['col3']

    t5 = join(t1, t4, keys='SDSS_NAME', join_type='left')

    axs[2].scatter(t5['Blueshift'],
                   np.log10(t5['EQW']),
                   c=cs[1],
                   s=25,
                   edgecolor='black',
                   label = 'Our Sample')

    t6 = Table()
    t6['SDSS_NAME'] = ['SDSSJ' + i for i in shen['SDSS_NAME']]
    t6['CIV_EQW'] = shen['EW_CIV']
    t6['VOFF_CIV_PEAK'] = shen['VOFF_CIV_PEAK']   

    t7 = join(t1, t6,  keys='SDSS_NAME', join_type='left')

    axs[0].scatter(t7['VOFF_CIV_PEAK'],
                   np.log10(t7['CIV_EQW']),
                   c=cs[1],
                   s=25,
                   edgecolor='black')


    legend = axs[2].legend(frameon=True) 

    axs[1].add_patch(patches.Rectangle((-499, np.log10(25)),     # (x,y)
                                       714,                    # width
                                       np.log10(40.0/25.0), # height
                                       fill=False,
                                       edgecolor='black',
                                       linestyle='solid',
                                       lw=1))      

    axs[1].add_patch(patches.Rectangle((1219, np.log10(25)),     # (x,y)
                                        203,                    # width
                                        np.log10(40.0/25.0), # height
                                        fill=False,
                                        edgecolor='black',
                                        linestyle='solid',
                                        lw=1))  

    axs[1].add_patch(patches.Rectangle((1981, np.log10(25)),     # (x,y)
                                       1002,                    # width
                                       np.log10(40.0/25.0), # height
                                       fill=False,
                                       edgecolor='black',
                                       linestyle='solid',
                                       lw=1))                                          

    axs[0].text(-922, 2.12, '(a)', ha='center', va='center')
    axs[1].text(-922, 2.12, '(b)', ha='center', va='center')
    axs[2].text(-922, 2.12, '(c)', ha='center', va='center')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.03)

    fig.savefig('/home/lc585/thesis/figures/chapter02/civ_space.pdf')

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

    cs = palettable.colorbrewer.qualitative.Set1_4.mpl_colors

    del cs[2]

    stem = '/data/vault/phewett/ICAtest/DR7_zica'

    fig, ax = plt.subplots(figsize=(figsize(0.75)))

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
    for i,composite in enumerate(composites):
        f = np.genfromtxt( os.path.join(stem,composite))
        x_data = (f[:,0] - line_wlen ) * c / line_wlen
        l, = ax.plot( x_data, f[:,1], lw=2, label=labels[i], color='black', linestyle=linestyles[i] )
        lines.append(l)

    ax.legend(handles=lines, labels=labels, prop={'size':10},frameon=True)

    ax.set_xlim(-10000,10000)
    ax.set_ylim(1,2.2)

    ax.set_xlabel(r'$\Delta v$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'$F_{\lambda}$ [Scaled Units]')

    ax.axvline(0.0, linestyle='--', color='grey')

    fig.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter02/civ_composites.pdf')

    plt.show() 

    return None 

def example_spectra():

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

    fig.savefig('/home/lc585/thesis/figures/chapter02/example_spectrum.pdf')

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


def example_spectrum_grid():

    """
    Needs fixing
    """
    

    fig = plt.figure(figsize=figsize(1.5, vscale = 2.15 * (np.sqrt(5.0)-1.0)/2.0))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(2, 3, wspace=0.0, hspace=0.05)

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)  
    
    names = ['QSO016',  
             'QSO014',  
             'QSO025',  
             'QSO399',  
             'QSO470',  
             'QSO128']  

    ylims = [[[-0.04, 4.5], [-0.04, 6.5]],
             [[-0.04, 3.5], [-0.04, 2.5]],
             [[-0.04, 3.0], [-0.04, 2.0]],
             [[-0.04, 5.0], [-0.04, 3.5]],
             [[-0.04, 4.0], [-0.04, 1.0]],
             [[-0.04, 5.5], [-0.04, 4.5]]]

    titles = ['J121427.77-030721.0',
              'J103325.93+012836.3',
              'J231441.63-082406.8',
              'J121140.59+103002.0',
              'J023359.72+004938.5',
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
        ax.set_title(titles[i], size=11)
        ax.text(0.05, 0.8, 'S/N: {}'.format(snr[i][0]), transform=ax.transAxes, zorder=10)
        fig.add_subplot(ax)

        if (i == 0) | (i == 3):
            ax.text(0.05, 0.5, r'H$\alpha$', transform= ax.transAxes)

        ax = plt.Subplot(fig, inner_grid[3])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        example_residual(names[i], 'Ha', ax)
        ax.set_xlim(-12000, 12000)
        ax.set_ylim(-8, 8)
        
        if (i == 0) | (i == 3):
            ax.set_yticks([-5,0,5])
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

        if (i == 0) | (i == 3):
            ax.text(0.05, 0.5, r'C\,{\sc iv}', transform= ax.transAxes)

        ax = plt.Subplot(fig, inner_grid[11])
        
        if i < 3:
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')

        ax.spines['top'].set_visible(False)
        example_residual(names[i], 'CIV', ax)
        ax.set_xlim(-12000, 12000)
        ax.set_ylim(-8, 8)
        if (i == 0) | (i == 3):
            ax.set_yticks([-5,0,5])
            ax.yaxis.set_ticks_position('left')
        else:
            ax.set_yticks([])


        fig.add_subplot(ax)


    fig.text(0.50, 0.05, r'$\Delta v$ [km~$\rm{s}^{-1}$]', ha='center')
    fig.text(0.05, 0.55, r'$F_{\lambda}$ [Arbitrary units]', rotation=90)

    fig.savefig('/home/lc585/thesis/figures/chapter02/gridspectra_1.pdf')
 
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

    fig, axs = plt.subplots(4, 1, figsize=figsize(0.7, vscale=2), sharex=True, sharey=True)
    

    df = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)


    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    df = df[df.WARN_1400_BEST == 0]
    df = df[['rescale' not in i for i in df.SPEC_NIR.values]]


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

    axs[3].scatter(df['Blueshift_CIV_Ha'], 
                   yi / 10**df['LogMBH_Ha'], 
                   facecolor=cs[1], 
                   label='This paper', 
                   edgecolor='None', 
                   marker='o', 
                   alpha=1.0,
                   zorder=1,
                   s=25)

    axs[3].set_yscale('log')
    axs[3].set_ylim(0.1, 10)
    axs[3].axhline(1, color='black', linestyle='--')
    axs[3].set_xlabel(r'C\,{\sc iv} Blueshift [km~$\rm{s}^{-1}$]')

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
    axs[1].axhline(1, color='black', linestyle='--')


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
    axs[0].axhline(1, color='black', linestyle='--')

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
    axs[2].axhline(1, color='black', linestyle='--')

    fig.text(0.04, 0.5, r'BHM C\,{\sc iv} (Corrected) / BHM H$\alpha$', va='center', rotation='vertical')

    axs[3].text(0.05, 0.9, '(d) This work', transform = axs[3].transAxes)
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