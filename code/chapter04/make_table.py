def make_table(): 

    import pandas as pd 
    import numpy as np
    import astropy.constants as const 
    import astropy.units as u 

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.OIII_FLAG_2 > 0]
    df.reset_index(inplace=True)
    
    df['OIII_5007_V5_PEAK'] = df['OIII_5007_V5'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    df['OIII_5007_V10_PEAK'] = df['OIII_5007_V10'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    df['OIII_5007_V25_PEAK'] = df['OIII_5007_V25'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    df['OIII_5007_V50_PEAK'] = df['OIII_5007_V50'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    df['OIII_5007_V75_PEAK'] = df['OIII_5007_V75'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    df['OIII_5007_V90_PEAK'] = df['OIII_5007_V90'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    df['OIII_5007_V95_PEAK'] = df['OIII_5007_V95'] - df['OIII_FIT_VEL_FULL_OIII_PEAK']
    
    df.loc[df.OIII_FIT_HB_Z_FLAG == -1, 'OIII_FIT_VEL_HB_PEAK'] = np.nan 
    df.loc[df.OIII_FIT_HB_Z_FLAG == -1, 'OIII_FIT_VEL_HB_PEAK_ERR'] = np.nan 
    df.loc[df.OIII_FIT_HB_Z_FLAG == -1, 'OIII_FIT_HB_Z'] = np.nan 
    df.loc[df.OIII_FIT_HB_Z_FLAG == -1, 'OIII_FIT_HB_Z_ERR'] = np.nan
    
    df.loc[df.OIII_FIT_HA_Z_FLAG <= 0, 'OIII_FIT_VEL_HA_PEAK'] = np.nan 
    df.loc[df.OIII_FIT_HA_Z_FLAG <= 0, 'OIII_FIT_VEL_HA_PEAK_ERR'] = np.nan 
    df.loc[df.OIII_FIT_HA_Z_FLAG <= 0, 'OIII_FIT_HA_Z'] = np.nan 
    df.loc[df.OIII_FIT_HA_Z_FLAG <= 0, 'OIII_FIT_HA_Z_ERR'] = np.nan # Doesn't exist yet             
    
    # Make Hb relative to OIII peak 
    df['OIII_FIT_VEL_HB_PEAK'] = df['OIII_FIT_VEL_HB_PEAK'] - df['OIII_FIT_VEL_FULL_OIII_PEAK'] 
    df['OIII_FIT_VEL_HB_PEAK_ERR'] = np.sqrt(df['OIII_FIT_VEL_HB_PEAK_ERR']**2 -     df['OIII_FIT_VEL_FULL_OIII_PEAK_ERR']**2)
    
    # Make Ha relative to OIII peak 
    df['zdiff'] = const.c.to(u.km/u.s)*(df.z_IR - df.z_IR_OIII_FIT)/(1.0 + df.z_IR_OIII_FIT)
    df['OIII_FIT_VEL_HA_PEAK'] = df['OIII_FIT_VEL_HA_PEAK'] + df['zdiff']
    df['OIII_FIT_VEL_HA_PEAK_ERR'] = np.sqrt(df['OIII_FIT_VEL_HA_PEAK_ERR']**2 -     df['OIII_FIT_VEL_FULL_OIII_PEAK_ERR']**2)
    
    
    columns = []
    
    columns.append('index')
    columns.append('OIII_5007_V5_PEAK')
    columns.append('OIII_5007_V5_PEAK_ERR')
    columns.append('OIII_5007_V10_PEAK')
    columns.append('OIII_5007_V10_PEAK_ERR')
    columns.append('OIII_5007_V25_PEAK')
    columns.append('OIII_5007_V25_PEAK_ERR')
    columns.append('OIII_5007_V50_PEAK')
    columns.append('OIII_5007_V50_PEAK_ERR')
    columns.append('OIII_5007_V75_PEAK')
    columns.append('OIII_5007_V75_PEAK_ERR')
    columns.append('OIII_5007_V90_PEAK')
    columns.append('OIII_5007_V90_PEAK_ERR')
    columns.append('OIII_5007_V95_PEAK')
    columns.append('OIII_5007_V95_PEAK_ERR')
    columns.append('OIII_FIT_Z_FULL_OIII_PEAK')
    columns.append('OIII_FIT_Z_FULL_OIII_PEAK_ERR')
    columns.append('OIII_5007_W50')
    columns.append('OIII_5007_W50_ERR')
    columns.append('OIII_5007_W80')
    columns.append('OIII_5007_W80_ERR')
    columns.append('OIII_5007_W90')
    columns.append('OIII_5007_W90_ERR')
    columns.append('OIII_5007_R_80')
    columns.append('OIII_5007_R_80_ERR')
    columns.append('OIII_5007_EQW_3')
    columns.append('OIII_5007_EQW_3_ERR')
    columns.append('OIII_5007_LUM_2')
    columns.append('OIII_5007_LUM_2_ERR')
    columns.append('OIII_FIT_EQW_FE_4434_4684')
    columns.append('OIII_FIT_EQW_FE_4434_4684_ERR')
    columns.append('OIII_FIT_VEL_HB_PEAK') 
    columns.append('OIII_FIT_VEL_HB_PEAK_ERR') 
    columns.append('OIII_FIT_VEL_HA_PEAK') 
    columns.append('OIII_FIT_VEL_HA_PEAK_ERR') 
    columns.append('OIII_FIT_HB_Z') 
    columns.append('OIII_FIT_HB_Z_ERR') 
    columns.append('OIII_FIT_HA_Z') 
    columns.append('OIII_FIT_HA_Z_ERR') 
    columns.append('REDCHI') 
    columns.append('FE_FLAG') 
    columns.append('OIII_EXTREM_FLAG')
    columns.append('OIII_FIT_HA_REDCHI')
    
    
    
    df = df[columns]
    
    df.rename(columns={'ID': 'ID',
                       'index': 'UID',
                       'OIII_5007_V5_PEAK': 'OIII_V5',
                       'OIII_5007_V5_PEAK_ERR': 'OIII_V5_ERR',
                       'OIII_5007_V10_PEAK': 'OIII_V10',
                       'OIII_5007_V10_PEAK_ERR': 'OIII_V10_ERR',
                       'OIII_5007_V25_PEAK': 'OIII_V25',
                       'OIII_5007_V25_PEAK_ERR': 'OIII_V25_ERR',
                       'OIII_5007_V50_PEAK': 'OIII_V50',
                       'OIII_5007_V50_PEAK_ERR': 'OIII_V50_ERR',
                       'OIII_5007_V75_PEAK': 'OIII_V75',
                       'OIII_5007_V75_PEAK_ERR': 'OIII_V75_ERR',
                       'OIII_5007_V90_PEAK': 'OIII_V90',
                       'OIII_5007_V90_PEAK_ERR': 'OIII_V90_ERR',
                       'OIII_5007_V95_PEAK': 'OIII_V95',
                       'OIII_5007_V95_PEAK_ERR': 'OIII_V95_ERR',
                       'OIII_FIT_Z_FULL_OIII_PEAK': 'z_OIII',
                       'OIII_FIT_Z_FULL_OIII_PEAK_ERR': 'z_OIII_ERR',
                       'OIII_5007_W50': 'OIII_W50',
                       'OIII_5007_W50_ERR': 'OIII_W50_ERR',
                       'OIII_5007_W80': 'OIII_W80',
                       'OIII_5007_W80_ERR': 'OIII_W80_ERR',
                       'OIII_5007_W90': 'OIII_W90',
                       'OIII_5007_W90_ERR': 'OIII_W90_ERR',
                       'OIII_5007_R_80': 'OIII_A',
                       'OIII_5007_R_80_ERR': 'OIII_A_ERR',
                       'OIII_5007_EQW_3': 'OIII_EQW',
                       'OIII_5007_EQW_3_ERR': 'OIII_EQW_ERR',
                       'OIII_5007_LUM_2': 'OIII_LUM',
                       'OIII_5007_LUM_2_ERR': 'OIII_LUM_ERR',
                       'OIII_FIT_EQW_FE_4434_4684': 'EQW_FE_4434_4684',
                       'OIII_FIT_EQW_FE_4434_4684_ERR': 'EQW_FE_4434_4684_ERR',
                       'OIII_FIT_VEL_HB_PEAK': 'HB_VPEAK', 
                       'OIII_FIT_VEL_HB_PEAK_ERR': 'HB_VPEAK_ERR', 
                       'OIII_FIT_VEL_HA_PEAK': 'HA_VPEAK', 
                       'OIII_FIT_VEL_HA_PEAK_ERR': 'HA_VPEAK_ERR', 
                       'OIII_FIT_HB_Z': 'HB_Z',
                       'OIII_FIT_HB_Z_ERR': 'HB_Z_ERR',
                       'OIII_FIT_HA_Z': 'HA_Z',
                       'OIII_FIT_HA_Z_ERR': 'HA_Z_ERR',
                       'REDCHI':'OIII_REDCHI',
                       'FE_FLAG':'OIII_FE_FLAG',
                       'OIII_FIT_HA_REDCHI': 'HA_REDCHI'}, inplace=True)

    return df 