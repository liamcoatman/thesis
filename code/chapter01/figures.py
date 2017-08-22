def shang_sed():

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.constants import c
    from PlottingTools.plot_setup_thesis import figsize, set_plot_properties

    set_plot_properties() # change style 
    
    rltab = np.genfromtxt('rlmsedMR.txt')

    fig, ax = plt.subplots(figsize=figsize(1.0, vscale=0.8))
    
    ax.plot(1.0e10 * c / (10**rltab[:,0]), 10**rltab[:,1], color='black', lw=1)
  
    ax.set_xlabel(r'Wavelength [\AA]')
    ax.set_ylabel( r'${\lambda}f_{\lambda}$ [Arbitrary Units]')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_ylim(1e-2, 10)
    ax.set_xlim(10, 10**8)

    ax.text(0.22, 0.3, 'Accretion \n disc', transform = ax.transAxes, multialignment='center')
    ax.text(0.52, 0.8, 'Torus', transform = ax.transAxes, multialignment='center')
    ax.text(0.1, 0.85, 'BLR \& \n NLR', transform = ax.transAxes, multialignment='center')

    ax.arrow(0.30, 0.4, 0.0, 0.2, color='black', transform = ax.transAxes, head_width=0.015 )
    ax.arrow(0.22, 0.88, 0.04, 0.0, color='black', transform=ax.transAxes, head_width=0.015)
    ax.arrow(0.57, 0.78, 0.0, -0.06, color='black', transform=ax.transAxes, head_width=0.015)

    plt.tight_layout()

    plt.savefig('../../figures/chapter01/shangsed.pdf')

    plt.show() 
    
    return None

if __name__ == '__main__':
    shang_sed()