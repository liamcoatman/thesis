import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import running 

def bbt_vs_z_with_correction():

    tab = Table.read('/data/lc585/QSOSED/Results/140530/out1.fits')
    
    newtab = tab[ (tab['BBT'] > 600.) & (tab['BBT'] < 1800.) ]
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot( newtab['Z'], newtab['BBT'], linestyle='', marker='o', markersize=2, markerfacecolor='grey', markeredgecolor='None', alpha=0.5)
    
    yrun = running.RunningMedian(tab['BBT'],101)[0::50]
    xrun = running.RunningMedian(tab['Z'],101)[0::50]
    
    ax.plot(xrun,yrun,linewidth=2,color='black')
    
    tab = Table.read('/data/lc585/QSOSED/Results/140530/out2.fits')
    
    yrun = running.RunningMedian(tab['BBT'],101)[0::50]
    xrun = running.RunningMedian(tab['Z'],101)[0::50]
    
    ax.plot(xrun,yrun,linewidth=2,color='blue')
    
    
    ax.set_xlabel(r'$z$',fontsize=12)
    ax.set_ylabel(r'T$_{\rm BB}$',fontsize=12)
    
    plt.tick_params(axis='both',which='major',labelsize=10)
    plt.tight_layout()
    plt.savefig('bbt_z.pdf')
    
    plt.show() 



def posterior_plot():

    import matplotlib.pyplot as plt 
    import pymc
    import sys
    sys.path.insert(0,'/home/lc585/Dropbox/IoA/QSOSED/Model/MCMC')
    import sedfit_pymc_v2 as fit
    from idlplot import plot,ploterror,oplot,tvhist2d,plothist

    f = '/data/lc585/QSOSED/Results/140324_MCMC_CHAINS/0019.pickle'
    pymcMC = pymc.database.pickle.load(f) 
    plot_posteriors(pymcMC)

    keys = ['ebv','elscal','imod']
    
        fig = plt.figure(figsize=(9.7,6.2))
                    
        counter = 0
    for k1 in keys:
        var1 = pymcMC.trace(k1)[:]
        #chain for variable k1
                
        for k2 in keys:
            var2 = pymcMC.trace(k2)[:]
            #chain for variable k1  
            
                        ax = fig.add_subplot(3,3,counter+1)
                        if k1 != k2:
                            tvhist2d(var2,var1,noerase=True,bins=[30,30])
                
            else:
                            plothist(var1,noerase=True,nbins=30)
            
                        if counter == 0:
                            ax.set_xlabel('E(B-V)',fontsize=10)
                            ax.set_ylabel('#',fontsize=10)

                        if counter == 3:
                            ax.set_xlabel('E(B-V)',fontsize=10)
                            ax.set_ylabel('EW Scaling',fontsize=10)

                        if counter == 4:
                            ax.set_xlabel('EW Scaling',fontsize=10)
                            ax.set_ylabel('#',fontsize=10)

                        if counter == 6:
                            ax.set_xlabel('E(B-V)',fontsize=10)
                            ax.set_ylabel('Normalisation',fontsize=10)

                        if counter == 7:
                            ax.set_xlabel('EW Scaling',fontsize=10)
                            ax.set_ylabel('Normalisation',fontsize=10)

                        
                        if counter == 8:
                            ax.set_xlabel('Normalisation',fontsize=10)
                            ax.set_ylabel('#',fontsize=10)                    
            
                         
                        if (counter == 1) | (counter == 2) | (counter == 5): 
                            plt.delaxes(ax)
                        
                      
                        counter +=1
                    
                   

                        plt.tick_params(axis='both',which='major',labelsize=8) 

      
        plt.tight_layout()
    plt.savefig('test.pdf') 


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