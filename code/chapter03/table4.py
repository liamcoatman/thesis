"""

Make table 3.4

"""

def make_table():

    import pandas as pd 
    import numpy as np 
    from astropy.table import Table 
    from astropy import constants as const 
    import astropy.units as u
    
    df1 = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[df1.WARN_CIV_BEST == 0]
    df1 = df1[df1.BAL_FLAG != 1]
    df2 = df1[['rescale' not in i for i in df1.SPEC_NIR.values]]
    
    df2 = pd.read_csv('/home/lc585/BHMassPaper2_Submitted_Data/masterlist_liam_resubmitted.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[df2.WARN_CIV_BEST == 0]
    df2 = df2[df2.BAL_FLAG != 1] 
    df2 = df2[['rescale' not in i for i in df2.SPEC_NIR.values]]
    
    df = pd.concat([df1, df2]).drop_duplicates()
    
    df.sort_values('RA_DEG', inplace=True)
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    
    names = []
    fwhm_broad_has = []
    fwhm_broad_ha_errs = []
    sigma_broad_has = []
    sigma_broad_ha_errs = []
    z_broad_has = []
    z_broad_ha_errs = []
    log_bhm_has = []
    log_bhm_ha_errs = [] 
    fwhm_broad_hbs = []
    fwhm_broad_hb_errs = []
    sigma_broad_hbs = []
    sigma_broad_hb_errs = []
    z_broad_hbs = []
    z_broad_hb_errs = []
    log_bhm_hbs = []
    log_bhm_hb_errs = [] 
    fwhm_civs = []
    fwhm_civ_errs = []
    sigma_civs = []
    sigma_civ_errs = []
    eqw_civs = []
    eqw_civ_errs = []
    civ_1400s = []
    blueshift_civ_has = []
    blueshift_civ_ha_errs = []
    blueshift_civ_hbs = []
    blueshift_civ_hb_errs = []
    logbhm_civ_vp06s = []
    logbhm_civ_vp06_errs = []
    logbhm_civ_c17s = []

    
    
    for idx, row in df.iterrows():
    
       
        names.append(idx)
    
        fwhm_broad_ha = np.round(row.FWHM_Broad_Ha, decimals=0)
        if row.WARN_Ha == 0:
            fwhm_broad_has.append(fwhm_broad_ha)
        else:
        	fwhm_broad_has.append(np.nan)
    
        fwhm_broad_ha_err = np.round(row.FWHM_Broad_Ha_Err, decimals=0)
        if row.WARN_Ha == 0:
            fwhm_broad_ha_errs.append(fwhm_broad_ha_err)
        else:
        	fwhm_broad_ha_errs.append(np.nan)
    
    
        sigma_broad_ha = np.round(row.Sigma_Broad_Ha, decimals=0)
        if row.WARN_Ha == 0:	
            sigma_broad_has.append(sigma_broad_ha)	
        else:
            sigma_broad_has.append(np.nan)	
    
        sigma_broad_ha_err = np.round(row.Sigma_Broad_Ha_Err, decimals=0)
        if row.WARN_Ha == 0:
            sigma_broad_ha_errs.append(sigma_broad_ha_err)	
        else:
            sigma_broad_ha_errs.append(np.nan)
    
    
        z_broad_ha = np.round(row.z_Broad_Ha, decimals=4)
        if row.WARN_Ha == 0:
            z_broad_has.append(z_broad_ha)  
        else:
            z_broad_has.append(np.nan)  
    
        z_broad_ha_err = np.round(((1.0 + row.z_Broad_Ha)*row.Median_Broad_Ha_Err*(u.km/u.s)) / const.c.to('km/s'), decimals=5)
        if row.WARN_Ha == 0:
            z_broad_ha_errs.append(z_broad_ha_err)  
        else:
            z_broad_ha_errs.append(np.nan)      


        log_bhm_ha = np.round(row.LogMBH_Ha, decimals=2)
        if row.WARN_Ha == 0:
            log_bhm_has.append(log_bhm_ha)  
        else:
            log_bhm_has.append(np.nan)  
        
        log_bhm_ha_err = np.round(row.LogMBH_Ha_Err, decimals=2)
        if row.WARN_Ha == 0:
            log_bhm_ha_errs.append(log_bhm_ha_err)  
        else:
            log_bhm_ha_errs.append(np.nan)  


   
        fwhm_broad_hb = np.round(row.FWHM_Broad_Hb, decimals=0)
        if row.WARN_Hb == 0:	
            fwhm_broad_hbs.append(fwhm_broad_hb)	
        else:
        	fwhm_broad_hbs.append(np.nan)	
    
        fwhm_broad_hb_err = np.round(row.FWHM_Broad_Hb_Err, decimals=0)
        if row.WARN_Hb == 0:
            fwhm_broad_hb_errs.append(fwhm_broad_hb_err)
        else:
            fwhm_broad_hb_errs.append(np.nan)	
    
        sigma_broad_hb = np.round(row.Sigma_Broad_Hb, decimals=0)
        if row.WARN_Hb == 0:
            sigma_broad_hbs.append(sigma_broad_hb)
        else:
            sigma_broad_hbs.append(np.nan)	
    
        sigma_broad_hb_err = np.round(row.Sigma_Broad_Hb_Err, decimals=0)
        if row.WARN_Hb == 0:
            sigma_broad_hb_errs.append(sigma_broad_hb_err)
        else:
            sigma_broad_hb_errs.append(np.nan)	
    
        z_broad_hb = np.round(row.z_Broad_Hb, decimals=4)
        if row.WARN_Hb == 0:
            z_broad_hbs.append(z_broad_hb)  
        else:
            z_broad_hbs.append(np.nan)  

        z_broad_hb_err = np.round(((1.0 + row.z_Broad_Hb)*row.Median_Broad_Hb_Err*(u.km/u.s)) / const.c.to('km/s'), decimals=5)
        if row.WARN_Hb == 0:
            z_broad_hb_errs.append(z_broad_hb_err)  
        else:
            z_broad_hb_errs.append(np.nan)      

        log_bhm_hb = np.round(row.LogMBH_Hb, decimals=2)
        if row.WARN_Hb == 0:
            log_bhm_hbs.append(log_bhm_hb)  
        else:
            log_bhm_hbs.append(np.nan)  
        
        log_bhm_hb_err = np.round(row.LogMBH_Hb_Err, decimals=2)
        if row.WARN_Hb == 0:
            log_bhm_hb_errs.append(log_bhm_hb_err)  
        else:
            log_bhm_hb_errs.append(np.nan)  

        fwhm_civ = np.round(row.FWHM_CIV_BEST, decimals=0)
        if row.WARN_CIV_BEST == 0:
            fwhm_civs.append(fwhm_civ)
        else:
            fwhm_civs.append(np.nan)
    
        fwhm_civ_err = np.round(row.FWHM_CIV_BEST_Err, decimals=0)
        if row.WARN_CIV_BEST == 0:
            fwhm_civ_errs.append(fwhm_civ_err)
        else:
            fwhm_civ_errs.append(np.nan)
    
        sigma_civ = np.round(row.Sigma_CIV_BEST, decimals=0)
        if row.WARN_CIV_BEST == 0:
            sigma_civs.append(sigma_civ)
        else:
            sigma_civs.append(np.nan)
    
        sigma_civ_err = np.round(row.Sigma_CIV_BEST_Err, decimals=0)
        if row.WARN_CIV_BEST == 0:
            sigma_civ_errs.append(sigma_civ_err)
        else:
            sigma_civ_errs.append(np.nan)
    
        eqw_civ = np.round(row.EQW_CIV_BEST, decimals=1)
        if row.WARN_CIV_BEST == 0:
            eqw_civs.append(eqw_civ)
        else:
            eqw_civs.append(np.nan)
    
        eqw_civ_err = np.round(row.EQW_CIV_BEST_Err, decimals=1)
        if row.WARN_CIV_BEST == 0:
            eqw_civ_errs.append(eqw_civ_err)
        else:
            eqw_civ_errs.append(np.nan)

        civ_1400 = np.round(row['1400_CIV_BEST'], decimals=2)
        if (row.WARN_CIV_BEST == 0) & (int(row.WARN_1400_BEST) != 2):
            civ_1400s.append(civ_1400)
        else:
            civ_1400s.append(np.nan)

        blueshift_civ_ha = np.round(row.Blueshift_CIV_Ha, decimals=0)
        if (row.WARN_Ha == 0) & (row.WARN_CIV_BEST == 0):
            blueshift_civ_has.append(blueshift_civ_ha)
        else: 
            blueshift_civ_has.append(np.nan)
    
        blueshift_civ_ha_err = np.round(row.Blueshift_CIV_Ha_Err, decimals=0)
        if (row.WARN_Ha == 0) & (row.WARN_CIV_BEST == 0):
            blueshift_civ_ha_errs.append(blueshift_civ_ha_err)
        else: 
            blueshift_civ_ha_errs.append(np.nan)
    
        blueshift_civ_hb = np.round(row.Blueshift_CIV_Hb, decimals=0)
        if (row.WARN_Hb == 0) & (row.WARN_CIV_BEST == 0):
            blueshift_civ_hbs.append(blueshift_civ_hb)
        else: 
            blueshift_civ_hbs.append(np.nan)
    
        blueshift_civ_hb_err = np.round(row.Blueshift_CIV_Hb_Err, decimals=0)
        if (row.WARN_Hb == 0) & (row.WARN_CIV_BEST == 0):
            blueshift_civ_hb_errs.append(blueshift_civ_hb_err)
        else: 
            blueshift_civ_hb_errs.append(np.nan)

    
        logbhm_civ_vp06 = np.round(row.LogMBH_CIV_VP06, decimals=2)
        if (row.WARN_CIV_BEST == 0):
            logbhm_civ_vp06s.append(logbhm_civ_vp06)
        else: 
            logbhm_civ_vp06s.append(np.nan)
    

        logbhm_civ_vp06_err = np.round(row.LogMBH_CIV_VP06_Err, decimals=2)
        if (row.WARN_CIV_BEST == 0):
            logbhm_civ_vp06_errs.append(logbhm_civ_vp06_err)
        else: 
            logbhm_civ_vp06_errs.append(np.nan)


        m, b = 0.41, 0.63 
        a = 6.71

        if row.WARN_Ha == 0: 
            
            blueshift = row.Blueshift_CIV_Ha*1e-3
            blueshift = max(blueshift, 0.0)

            fwhm = row.FWHM_CIV_BEST / (m*blueshift + b)
            bhm = np.log10(10**a * (fwhm * 1.0e-3)**2 * ((10**row.LogL1350) * 1.0e-44)**0.53)

        elif row.WARN_Hb == 0: 

            blueshift = row.Blueshift_CIV_Hb*1e-3
            blueshift = max(blueshift, 0.0)

            fwhm = row.FWHM_CIV_BEST / (m*blueshift + b)
            bhm = np.log10(10**a * (fwhm * 1.0e-3)**2 * ((10**row.LogL1350) * 1.0e-44)**0.53)
        

        logbhm_civ_c17 = np.round(bhm, decimals=2)

        if row.WARN_CIV_BEST == 0:
            logbhm_civ_c17s.append(logbhm_civ_c17)
        else: 
            logbhm_civ_c17s.append(np.nan)

    


    df = pd.DataFrame() 

    df['UID'] = names
    df['FWHM_BROAD_HA'] = fwhm_broad_has
    df['FWHM_BROAD_HA_ERR'] = fwhm_broad_ha_errs
    df['SIGMA_BROAD_HA'] = sigma_broad_has
    df['SIGMA_BROAD_HA_ERR'] = sigma_broad_ha_errs
    df['Z_BROAD_HA'] = z_broad_has
    df['Z_BROAD_HA_ERR'] = z_broad_ha_errs
    df['LOGMBH_HA'] = log_bhm_has
    df['LOGMBH_HA_ERR'] = log_bhm_ha_errs
    df['FWHM_BROAD_HB'] = fwhm_broad_hbs
    df['FWHM_BROAD_HB_ERR'] = fwhm_broad_hb_errs
    df['SIGMA_BROAD_HB'] = sigma_broad_hbs
    df['SIGMA_BROAD_HB_ERR'] = sigma_broad_hb_errs
    df['Z_BROAD_HB'] = z_broad_hbs
    df['Z_BROAD_HB_ERR'] = z_broad_hb_errs
    df['LOGMBH_HB'] = log_bhm_hbs
    df['LOGMBH_HB_ERR'] = log_bhm_hb_errs
    df['FWHM_CIV'] = fwhm_civs
    df['FWHM_CIV_ERR'] = fwhm_civ_errs
    df['SIGMA_CIV'] = sigma_civs
    df['SIGMA_CIV_ERR'] = sigma_civ_errs
    df['EQW_CIV'] = eqw_civs
    df['EQW_CIV_ERR'] = eqw_civ_errs
    df['1400_CIV'] = civ_1400s
    df['BLUESHIFT_CIV_HA'] = blueshift_civ_has
    df['BLUESHIFT_CIV_HA_ERR'] = blueshift_civ_ha_errs
    df['BLUESHIFT_CIV_HB'] = blueshift_civ_hbs
    df['BLUESHIFT_CIV_HB_ERR'] = blueshift_civ_hb_errs
    df['LOGMBH_CIV_VP06'] = logbhm_civ_vp06s
    df['LOGMBH_CIV_VP06_ERR'] = logbhm_civ_vp06_errs
    df['LOGMBH_CIV_C17'] = logbhm_civ_c17s

    df.sort_values('UID', inplace=True)


    df.to_csv('/home/lc585/thesis/data/table3.4.csv',
              index=False,
              )

    return df

# t.write('table2.csv', 
#         format='ascii.fixed_width', 
#         comment=False)