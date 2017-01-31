def plot(): 

    from PlottingTools.plot_setup_thesis import figsize, set_plot_properties
    from astropy.table import Table, join
    import numpy as np 
    import matplotlib.pyplot as plt 

    set_plot_properties() # change style 

    tab = Table.read('/data/lc585/QSOSED/Results/141209/sample1/out_add.fits')
    
    tab = tab[ ~np.isnan(tab['BBT_STDERR'])]
    tab = tab[ tab['BBT_STDERR'] < 500. ]
    tab = tab[ tab['BBT_STDERR'] > 5.0 ]
    tab = tab[ (tab['LUM_IR_SIGMA']*tab['RATIO_IR_UV']) < 1.]
    
    tab1 = Table.read('/data/lc585/QSOSED/Results/141124/sample2/out.fits') # 9000 - 23500 fit
    tab1['BBPLSLP'] = tab1['BBPLSLP'] - 1.0
    
    goodnames = Table()
    goodnames['NAME'] = tab['NAME']
    
    tab1 = join( tab1, goodnames, join_type='right', keys='NAME')
    
    fig, ax = plt.subplots(figsize=figsize(1, 0.7))

    im = ax.hexbin(tab['BBT'],
                   tab['RATIO_IR_UV'],
                   C = tab1['BBPLSLP'],
                   gridsize=80,)
    
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
    
    plt.tight_layout()

    fig.savefig('/home/lc585/thesis/figures/chapter06/ratio_tbb_beta.pdf')
    
    plt.show() 
    
    return None 