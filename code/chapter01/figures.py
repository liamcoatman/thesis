def shang_sed():

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.constants import c
    
    rltab = np.genfromtxt('rlmsedMR.txt')
    rqtab = np.genfromtxt('rqmsedMR.txt')
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    
    ax.plot( 1.0e10 * c / (10**rltab[:,0]), 10**rltab[:,1], color='black', label='Radio-loud' )
    ax.plot( 1.0e10 * c / (10**rqtab[:,0]), 10**rqtab[:,1], color='green', label='Radio-quiet' )
    
    ax.axvline(912,color='black',linestyle='--')
    ax.axvline(10**4,color='black',linestyle='--')
    ax.axvline(6.e5,color='black',linestyle='--')
    
    ax.set_xlabel(r'Wavelength $\AA$',fontsize=10)
    ax.set_ylabel( r'log(${\lambda}f_{\lambda}$) (Arbitrary Units)',fontsize=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_ylim(1e-3,12)
    ax.set_xlim(10,10**8)
    
    plt.legend( loc='lower left', prop={'size':10} )
    
    ax.text(1.3e3, 2e-1, 'Big Blue \n Bump', fontsize=12)
    ax.text(2e4, 2e-1, 'Near-IR Bump', fontsize=12)
    
    plt.tick_params(axis='both',which='major',labelsize=10)
    plt.tight_layout()
    
    plt.savefig('shangsed.pdf')

    return None 