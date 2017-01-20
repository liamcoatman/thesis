import pandas as pd 
from astropy.table import Table 
from astropy.io import ascii 
import numpy as np

"""
Make big source table 
"""

def make_table():

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.SPEC_NIR != 'None']

    # band?  
    # programme 
    # duplicates
    # z_oiii 
    # luminosity 5100A 
    # link to optical spectra + sdss  
    # date 
     
    # names 
    names = []
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    for idx, row in df.iterrows():
        name = 'J' + str(row.RA) + str(row.DEC)
        for i, j in dic.iteritems():
            name = name.replace(i, j)
        name = name[:10] + name[11:20]
        names.append(name)
  
    # redshift 
    def oiii_z(row):
        if row.OIII_FLAG_2 == 1:
            return format(np.around(row.OIII_FIT_Z_FULL_OIII_PEAK, decimals=4), '.4f')
        else:
            return ''
    z = df.apply(oiii_z, axis=1).values 

    # instrument / telescope 
    def get_instr(row):

        if row.INSTR == 'FIRE':
            return pd.Series({'Instrument': 'FIRE', 
                              'Telescope': 'Magellan-Baade'})

        if row.INSTR == 'GNIRS':
            return pd.Series({'Instrument': 'GNIRS', 
                              'Telescope': 'Gemini-N'})

        if row.INSTR == 'ISAAC':
            return pd.Series({'Instrument': 'ISAAC', 
                              'Telescope': 'VLT'})

        if row.INSTR == 'LIRIS':
            return pd.Series({'Instrument': 'LIRIS', 
                              'Telescope': 'WHT'})

        if row.INSTR == 'NIRI':
            return pd.Series({'Instrument': 'NIRI', 
                              'Telescope': 'Gemini-N'})

        if row.INSTR == 'NIRSPEC':
            return pd.Series({'Instrument': 'NIRSPEC', 
                              'Telescope': 'Keck-II'})

        if row.INSTR == 'SINF':
            return pd.Series({'Instrument': 'SINFONI', 
                              'Telescope': 'VLT'})

        if row.INSTR == 'SINF_KK':
            return pd.Series({'Instrument': 'SINFONI', 
                              'Telescope': 'VLT'})

        if row.INSTR == 'SOFI_JH':
            return pd.Series({'Instrument': 'SofI', 
                              'Telescope': 'NTT'})

        if row.INSTR == 'SOFI_LC':
            return pd.Series({'Instrument': 'SofI', 
                              'Telescope': 'NTT'})

        if row.INSTR == 'TRIPLE':
            return pd.Series({'Instrument': 'TRIPLESPEC', 
                              'Telescope': 'Palomar 200-inch'})

        if row.INSTR == 'TRIPLE_S15':
            return pd.Series({'Instrument': 'TRIPLESPEC', 
                              'Telescope': 'ARC 3.5m'})

        if row.INSTR == 'XSHOOT':
            return pd.Series({'Instrument': 'XSHOOTER', 
                              'Telescope': 'VLT'}) 

    
    df = df.apply(get_instr, axis=1)

    instrument = df.Instrument.values 
    telescope = df.Telescope.values

    # dates 

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.SPEC_NIR != 'None']

    dates = []
    for idx, row in df.iterrows():
      dates.append(row.DATE)


    # band? 




    # imag = np.around(df.psfMag_i.values, decimals=2)
    # specopt = df.SPEC_OPT.values
    # specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    # snr_ha, snr_civ = [], []

    # snr_ha = []
    # snr_civ = []
    
    # for i, row in df.iterrows():
    
    #     if row.WARN_CIV == 0:
    #         snr_civ.append(row.SNR_CIV)
    #     else:
    #         snr_civ.append('')
    #     if row.WARN_Ha == 0:
    #         snr_ha.append(row.SNR_Ha)
    #     else:
    #         snr_ha.append('')



    
    # bal = np.zeros(len(df), dtype=np.int)
    # bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    # radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    # radio[radio < -99] = -1
    
    # dates = []
    # for idx, row in df.iterrows():
    # 	dates.append(row.DATE)
    
    # exptimes = []
    # for idx, row in df.iterrows():
    # 	exptimes.append(row.EXPTIME)
    
    
    # edge_flag_ha = []
    # snr_flag_ha = ['QSO463', 'QSO455']
    # abs_flag_ha = ['QSO465']
    
    # j, h, k = True, True, True  
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             print i
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
    
    # edge_flag_civ = []
    # snr_flag_civ = []
    # abs_flag_civ = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    # tnew['Exp'] = exptimes
    tnew['z'] = z
    tnew['instrument'] = instrument 
    tnew['telescope'] = telescope 
    # tnew['imag'] = imag 
    # tnew['Opt. Spec.'] = specopt
    # tnew['S/N Ha'] = snr_ha
    # tnew['S/N CIV'] = snr_civ
    # tnew['Radio'] = radio 
    
    tnew.sort('Name')
    
    ascii.write(tnew, format='latex')



# ntt_coatman() 


def xshooter():

    """
    XSHOOTER spectra from the ESO archive
    Only those reduced 
    Say which program these are from 
    """

    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV == 0) | (df1.WARN_CIV_XSHOOT == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV == 0) | (df2.WARN_CIV_XSHOOT == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 
    
    df3 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df3 = df3[df3.WARN_Hb == 0]
    df3 = df3[df3.WARN_Ha == 0]
        
    df = pd.concat([df1, df2, df3]).drop_duplicates()
    
    df = df[df.INSTR == 'XSHOOT']
    
    # df2 = pd.DataFrame()
    
    # df2['ESOID'] = ['087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '087.A-0610(A)',
    #                 '090.A-0824(A)',
    #                 '090.A-0824(A)',
    #                 '090.A-0824(A)',
    #                 '090.A-0824(A)',
    #                 '090.A-0824(A)',
    #                 '092.A-0764(A)',
    #                 '092.A-0764(A)',
    #                 '092.A-0764(A)',
    #                 '089.A-0855(A)',
    #                 '089.A-0855(A)',
    #                 '089.A-0855(A)',
    #                 '089.A-0855(A)',
    #                 '089.A-0855(A)',
    #                 '089.A-0855(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)',
    #                 '091.A-0299(A)']
    
    # df2.index = ['QSO176',
    #              'QSO177',
    #              'QSO178',
    #              'QSO179',
    #              'QSO180',
    #              'QSO243',
    #              'QSO244',
    #              'QSO245',
    #              'QSO289',
    #              'QSO290',
    #              'QSO292',
    #              'QSO293',
    #              'QSO294',
    #              'QSO295',
    #              'QSO296',
    #              'QSO297',
    #              'QSO299',
    #              'QSO300',
    #              'QSO301',
    #              'QSO302',
    #              'QSO303',
    #              'QSO304',
    #              'QSO305',
    #              'QSO306',
    #              'QSO307',
    #              'QSO308',
    #              'QSO309',
    #              'QSO310',
    #              'QSO311',
    #              'QSO312',
    #              'QSO313',
    #              'QSO314',
    #              'QSO315',
    #              'QSO316']
    
    # df = pd.concat([df, df2], axis=1) 
    
    df = df.sort('RA_DEG')
    
    names = []
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    for idx, row in df.iterrows():
        name = 'J' + str(row.RA) + str(row.DEC)
        for i, j in dic.iteritems():
            name = name.replace(i, j)
        name = name[:10] + name[11:20]
        names.append(name)
    
    # eso = df.ESOID.values 
    
    z = []
    zsource = []
    for idx, row in df.iterrows():
        if 'SDSS' in str(row.SPEC_OPT):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif 'BOSS' in str(row.SPEC_OPT):
            z.append(row.z_PCA_DR12)
            zsource.append('PCA_DR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)
    
    z = np.around(z, decimals=4)
    z = [format(i, '.4f') for i in z]
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
        if not np.isnan(float(row.EXPTIME)):
            exptimes.append(str(int(round(float(row.EXPTIME) / 10.0) * 10.0))) 
        else:
            exptimes.append('')
    
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'None'] = ''
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    specopt[np.where(df.WARN_CIV == 2)[0]] = ''
    
    snr_ha = []
    snr_hb = []
    snr_civ = []
    snr_civ_xshoot = []
    
    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(row.SNR_Ha)
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')
        if row.WARN_CIV_XSHOOT == 0:
            snr_civ_xshoot.append(row.SNR_CIV_XSHOOT)
        else:
            snr_civ_xshoot.append('')
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # edge_flag_ha = ['QSO178', 'QSO245', 'QSO309', 'QSO310']
    # snr_flag_ha = []
    # abs_flag_ha = [] 
    
    # j, h, k = True, True, True  
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | (row.WARN_Ha == -1) | np.isnan(row.SNR_Ha):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N.}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption.}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
    
    # edge_flag_hb = []
    # snr_flag_hb = ['QSO179', 'QSO312', 'QSO311', 'QSO303', 'QSO290', 'QSO243', 'QSO310', 'QSO302']
    # abs_flag_hb = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | (row.WARN_Hb == -1) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N.}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption.}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    
    # edge_flag_civ = []
    # snr_flag_civ = ['QSO301', 'QSO180', 'QSO295', 'QSO302']
    # abs_flag_civ = ['QSO310', 'QSO176']
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | (row.WARN_CIV == -1) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N.}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption.}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    
    # edge_flag_civ_xshoot = []
    # snr_flag_civ_xshoot = []
    # abs_flag_civ_xshoot = ['QSO176', 'QSO243', 'QSO292', 'QSO293', 'QSO296', 'QSO297']
    # other_flag_civ_xshoot = ['QSO300', 'QSO307', 'QSO311', 'QSO312']
    
    # y = True 
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV_XSHOOT == 2) | (row.WARN_CIV_XSHOOT == -1) | np.isnan(row.SNR_CIV_XSHOOT):
    #         snr_civ_xshoot.append('')
    #     elif i in edge_flag_civ_xshoot:
    #         if j:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ_xshoot:
    #         if h:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N.}')
    #             h = False 
    #         else:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ_xshoot:
    #         if k:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footnote{\\label{footnote3}Absorption.}')
    #             k = False 
    #         else:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footref{footnote3}')
    #     elif i in other_flag_civ_xshoot:
    #         if y:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footnote{\\label{footnote4}Other.}')
    #             y = False 
    #         else:
    #             snr_civ_xshoot.append(str(np.around(row.SNR_CIV_XSHOOT, decimals=1)) + '\\footref{footnote4}')
    #     else:
    #         snr_civ_xshoot.append(np.around(row.SNR_CIV_XSHOOT, decimals=1))
    
    
    
    tnew = Table()
    
    tnew['Name'] = names
    # tnew['ESO ID'] = eso
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['z'] = z
    tnew['Opt. Spec.'] = specopt
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['S/N CIV X'] = snr_civ_xshoot
    tnew['S/N CIV'] = snr_civ
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['zsource'] = zsource
    
    tnew.remove_rows((tnew['S/N Ha'] == '') & (tnew['S/N Hb'] == ''))
    
    ascii.write(tnew, format='latex')
    
    print len(tnew)

# xshooter() 

"""
14 z~3 quasars from Shen (2016)
"""

def shen_highz():


    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Hb == 0]
    df = df[(df.WARN_CIV == 0)]
    df = df[df.BAL_FLAG != 1] 

    df = df[df.INSTR == 'FIRE']
    df = df[df.z > 3.0]
   
    names = ['J' + row.DR12_NAME.replace('SDSSJ', '') for i, row in df.iterrows()]

    z = np.around(df.z_HW_DR7.values, decimals=4)
    imag = np.around(df.psfMag_i.values, decimals=2)
    imag = [format(i, '.2f') for i in imag]
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
        if np.isnan(row.EXPTIME):
            exptimes.append('')
        else:
            exptimes.append(row.EXPTIME)
    
    snr_ha, snr_hb, snr_civ = [], [], []
    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1

    snr_hb = []
    snr_civ = []
    
    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(format(np.around(row.SNR_CIV, decimals=2), '.2f'))
        else:
            snr_civ.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(format(np.around(row.SNR_Hb, decimals=2), '.2f'))
        else:
            snr_hb.append('')
  
    
    # j, h, k = True, True, True  
    
    
    # edge_flag_hb = []
    # snr_flag_hb = ['QSO528', 'QSO524', 'QSO520', 'QSO519']
    # abs_flag_hb = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # edge_flag_civ = []
    # snr_flag_civ = []
    # abs_flag_civ = ['QSO516']
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['S/N Hb'] = snr_hb
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['imag'] = imag 
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    
    tnew.sort('Name')
    
    # tnew.remove_rows((tnew['S/N Hb'] == '') | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

# shen_highz()

def shen_lowz():

    """
    Shen & Liu (2012) sample 
    SDSSJ123355.21+031327.6 marked as BAL in DR12, but not in DR7. Doesn't look like a BAL. 
    100930.51+023052.4 absorption at peak not well fit for 
    """

    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 

    df = pd.concat([df1, df2]).drop_duplicates()

    df = df[(df.INSTR == 'FIRE') | (df.INSTR == 'TRIPLE_S15')]
    # df = df[df.z < 3.0]

    names = ['J' + row.DR12_NAME.replace('SDSSJ', '') for i, row in df.iterrows()]
    z = np.around(df.z_HW_DR7.values, decimals=4)
    z = [format(i, '.4f') for i in z]
    imag = np.around(df.psfMag_i.values, decimals=2)
    imag = [format(i, '.2f') for i in imag]
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    specnir = df.INSTR.values
    specnir[specnir == 'TRIPLE_S15'] = 'TRIPLESPEC'
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)

    newdates = []
    for date in dates:
        print date
        if len(date) == 6:
            date = date[:2] + '-' + date[2:4] + '-' + date[4:]
        else:
        	date = date[:2] + '-' + date[2:4] + '-' + date[4:6] + '/' + date[7:9] + '-' + date[9:11] + '-' + date[11:]
        newdates.append(date)
    
    exptimes = []
    for idx, row in df.iterrows():
        if np.isnan(row.EXPTIME):
            exptimes.append('')
        else:
            exptimes.append(row.EXPTIME)
    
    snr_ha, snr_hb, snr_civ = [], [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(format(np.around(row.SNR_CIV, decimals=2), '.2f'))
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(format(np.around(row.SNR_Ha, decimals=2), '.2f'))
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(format(np.around(row.SNR_Hb, decimals=2), '.2f'))
        else:
            snr_hb.append('')

    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # j, h, k = True, True, True  
    
    # edge_flag_ha = []
    # snr_flag_ha = []
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
    
    # snr_flag_hb = ['QSO546', 'QSO514', 'QSO506', 'QSO497', 'QSO495', 'QSO494', 'QSO492', 'QSO486', 'QSO484', 'QSO483', 'QSO482', 'QSO479']
    # edge_flag_hb = ['QSO509', 'QSO481'] 
    # abs_flag_hb = []
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # abs_flag_civ = ['QSO530']
    # snr_flag_civ = []
    # edge_flag_civ = ['QSO515', 'QSO489']  
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['NIR Spec.'] = specnir
    tnew['Date'] = newdates 
    tnew['Exp'] = exptimes
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['imag'] = imag 
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    
    tnew.sort('Name')
    
    tnew.remove_rows(tnew['S/N CIV'] == '')

    ascii.write(tnew, format='latex')

    print len(tnew)

# shen_lowz() 


def gnirs(): 

    """
    GNIRS sample 
    
    25 quasars with NIR + optical spectra. 2 < z < 2.3. Not all in SDSS so give BOSS DR12 name. All cover Ha and Hb. 
    Not sure what redshifts are from - possibly SDSS/BOSS. 
    """
    
    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 
    
    df3 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df3 = df3[df3.WARN_Hb == 0]
    df3 = df3[df3.WARN_Ha == 0]
        
    df = pd.concat([df1, df2, df3]).drop_duplicates()

    df = df[(df.INSTR == 'GNIRS')]
    
    
    names = ['J' + row.DR12_NAME.replace('SDSSJ', '') for i, row in df.iterrows()]

    z = []
    zsource = []
    for idx, row in df.iterrows():
        if 'SDSS' in str(row.SPEC_OPT):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif 'BOSS' in str(row.SPEC_OPT):
            z.append(row.z_PCA_DR12)
            zsource.append('PCA_DR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)

    z = [format(i, '.4f') for i in z]

    imag = np.around(df.psfMag_i.values, decimals=2)
    imag = [format(i, '.2f') for i in imag]
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    specopt[specopt == 'None'] = ''
    
    snr_ha, snr_hb, snr_civ = [], [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(row.SNR_Ha)
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')

    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    dates = []
    for idx, row in df.iterrows():
    	if isinstance(row.DATE, basestring):
            dates.append(row.DATE)
        else:
        	dates.append('')

    
    exptimes = []
    for idx, row in df.iterrows():
        exptimes.append(int(row.EXPTIME))
    
    # j, h, k = True, True, True  
    
    # edge_flag_ha = []
    # snr_flag_ha = []
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
    
    # snr_flag_hb = ['QSO024', 'QSO001']
    # edge_flag_hb = [] 
    # abs_flag_hb = []
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # abs_flag_civ = []
    # snr_flag_civ = ['QSO011', 'QSO008', 'QSO018']
    # edge_flag_civ = []  
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['z'] = z
    tnew['imag'] = imag 
    tnew['Opt. Spec.'] = specopt
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['S/N CIV'] = snr_civ
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['zsource'] = zsource
    
    tnew.sort('Name')
    
    # tnew.remove_rows(((tnew['S/N Ha'] == '') & (tnew['S/N Hb'] == '')) | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')
        

# gnirs()    

def qpq(): 

    """
    GNIRS sample 
    
    25 quasars with NIR + optical spectra. 2 < z < 2.3. Not all in SDSS so give BOSS DR12 name. All cover Ha and Hb. 
    Not sure what redshifts are from - possibly SDSS/BOSS. 
    """
    
    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV_BEST == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV_BEST == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 

    df = pd.concat([df1, df2]).drop_duplicates()

    df = df[(df.INSTR == 'GNIRS') | (df.INSTR == 'ISAAC') | (df.INSTR == 'NIRI') | (df.INSTR == 'XSHOOT')]

    df.loc[df.INSTR == 'XSHOOT', 'INSTR'] = 'XSHOOTER'

    df.loc[df.WARN_CIV_XSHOOT == 0, 'SPEC_OPT'] = 'XSHOOTER' 
        
    names = []
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    for idx, row in df.iterrows():
        name = 'J' + row.RA + row.DEC
        for i, j in dic.iteritems():
            name = name.replace(i, j)
        names.append(name)

    z = []
    zsource = []
    for idx, row in df.iterrows():
        if not np.isnan(row.z_HW_DR7):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif not np.isnan(row.z_PCA_DR12):
            z.append(row.z_PCA_DR12)
            zsource.append('PCADR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)

    z = [format(i, '.4f') for i in z]

    imag = []
    for i, row in df.iterrows():
        if np.isnan(row.psfMag_i):
            imag.append('')
        else:
            imag.append(format(np.around(row.psfMag_i, decimals=2), '.2f'))

    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    specopt[specopt == 'None'] = ''
    
    snr_ha, snr_hb, snr_civ = [], [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV_BEST == 0:
            snr_civ.append(format(np.around(row.SNR_CIV_BEST, decimals=2), '.2f'))
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(format(np.around(row.SNR_Ha, decimals=2), '.2f'))
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(format(np.around(row.SNR_Hb, decimals=2), '.2f'))
        else:
            snr_hb.append('')

    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    dates = []
    for idx, row in df.iterrows():
        if isinstance(row.DATE, basestring):
            dates.append(row.DATE)
        else:
            dates.append('')

    
    exptimes = []
    for idx, row in df.iterrows():
        if isinstance(row.EXPTIME, basestring):
            exptimes.append(int(float(row.EXPTIME)))
        else:
            exptimes.append('')
    
    # j, h, k = True, True, True  
    
    # edge_flag_ha = []
    # snr_flag_ha = []
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
    
    # snr_flag_hb = ['QSO024', 'QSO001']
    # edge_flag_hb = [] 
    # abs_flag_hb = []
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # abs_flag_civ = []
    # snr_flag_civ = ['QSO011', 'QSO008', 'QSO018']
    # edge_flag_civ = []  
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Spec NIR'] = df.INSTR
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['imag'] = imag 
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['zsource'] = zsource
    
    tnew.sort('Name')
    
    # tnew.remove_rows(((tnew['S/N Ha'] == '') & (tnew['S/N Hb'] == '')) | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')


# qpq()

def triple():

    """
    TRIPLESPEC quasar pairs sample 
    
    """
    
    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 
        
    df = pd.concat([df1, df2]).drop_duplicates()

    df = df[(df.INSTR == 'TRIPLE')]
    
    
    names = []
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    for idx, row in df.iterrows():
        name = 'J' + row.RA + row.DEC
        for i, j in dic.iteritems():
            name = name.replace(i, j)
        names.append(name)
    
    z = []
    zsource = []
    for idx, row in df.iterrows():
        if 'SDSS' in str(row.SPEC_OPT):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif 'BOSS' in str(row.SPEC_OPT):
            z.append(row.z_PCA_DR12)
            zsource.append('PCA_DR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)

    z = [format(i, '.4f') for i in z]

    imag = np.around(df.psfMag_i.values, decimals=2)
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    snr_ha, snr_hb, snr_civ = [], [], []
    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
        exptimes.append(int(float(row.EXPTIME)))

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(row.SNR_Ha)
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')

    
    # j, h, k = True, True, True  
    
    # snr_flag_ha = ['QSO108', 'QSO118', 'QSO110', 'QSO111', 'QSO175']
    # edge_flag_ha = ['QSO130']
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha) | (row.Ha == 0):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
    
    # snr_flag_hb = ['QSO154', 'QSO138', 'QSO135', 'QSO146', 'QSO114', 'QSO123', 'QSO152', 'QSO173', 'QSO121', 'QSO137', 'QSO144', 'QSO145', 'QSO156', 'QSO161', 'QSO171', 'QSO175', 'QSO111', 'QSO118']
    # edge_flag_hb = ['QSO150', 'QSO130', 'QSO110'] 
    # abs_flag_hb = []
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # edge_flag_civ = []  
    # abs_flag_civ = ['QSO135', 'QSO146']
    # snr_flag_civ = ['QSO138', 'QSO175']
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['imag'] = imag 
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['zsource'] = zsource
    
    tnew.sort('Name')
    
    # tnew.remove_rows(((tnew['S/N Ha'] == '') & (tnew['S/N Hb'] == '')) | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

# triple() 

def sofi_hennawi():

    """
    SOFI HENNAWI sample. 52 quasars (one duplicate).  
    
    """
    
    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 
        
    df = pd.concat([df1, df2]).drop_duplicates()

    df = df[(df.INSTR == 'SOFI_LC') ]
    
    names = []
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    for idx, row in df.iterrows():
        name = 'J' + row.RA + row.DEC
        for i, j in dic.iteritems():
            name = name.replace(i, j)
        names.append(name)
      
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
    	exptimes.append(int(float(row.EXPTIME)))
    
    z = []
    zsource = []
    for idx, row in df.iterrows():
        z.append(row.z)
        zsource.append(row.z_source)

    z = [format(i, '.4f') for i in z]

    imag = []
    for i, row in df.iterrows():
        if np.isnan(row.psfMag_i):
            imag.append('')
        else:
            imag.append(str(np.around(row.psfMag_i, decimals=2)))
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    specopt[specopt == 'None'] = ''
    
    snr_ha, snr_hb, snr_civ = [], [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(row.SNR_Ha)
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')

    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # j, h, k = True, True, True  
    
    # snr_flag_ha = []
    # edge_flag_ha = []
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha) | (row.Ha == 0):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
     
    # abs_flag_hb = []
    # edge_flag_hb = ['QSO344', 'QSO360', 'QSO366', 'QSO391', 'QSO392', 'QSO404', 'QSO417', 'QSO421']
    # snr_flag_hb = ['QSO358', 'QSO362', 'QSO370', 'QSO373', 'QSO374', 'QSO379', 'QSO384', 'QSO393', 'QSO395'] 
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # edge_flag_civ = []  
    # abs_flag_civ = ['QSO393']
    # snr_flag_civ = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['imag'] = imag 
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['dsfa'] = zsource
    
    tnew.sort('Name')
    
    # tnew.remove_rows(((tnew['S/N Ha'] == '') & (tnew['S/N Hb'] == '')) | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

# sofi_hennawi() 


def sinfoni():

    """
    SINFONI sample 
    
    8 from Lutz  Run ID: 083.B-0456(A) 
    11 from Kurk  Run ID: 090.B-0674(B)
    
    """
    
    
    df1 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df1 = df1[df1.WARN_Ha == 0]
    df1 = df1[(df1.WARN_CIV == 0)]
    df1 = df1[df1.BAL_FLAG != 1]
    
    df2 = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df2 = df2[df2.WARN_Hb == 0]
    df2 = df2[(df2.WARN_CIV == 0)]
    df2 = df2[df2.BAL_FLAG != 1] 

        
    df = pd.concat([df1, df2]).drop_duplicates()
    
    df = df[(df.INSTR == 'SINF') ]
    
      
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
        try:
            t = int(float(row.EXPTIME))
    	except:
            t = row.EXPTIME
    	exptimes.append(t)

    eso = []
    for i, row in df.iterrows():
        if row.INSTR == 'SINF':
        	eso.append('083.B-0456(A)')
        else:
        	eso.append('090.B-0674(B)')
    
    names = []
    dic = {'h':'', 'm':'', 's':'', 'd': ''}
    for idx, row in df.iterrows():
        name = 'J' + row.RA + row.DEC
        for i, j in dic.iteritems():
            name = name.replace(i, j)
        names.append(name)
    
    z = []
    zsource = []
    for idx, row in df.iterrows():
        if 'SDSS' in str(row.SPEC_OPT):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif 'BOSS' in str(row.SPEC_OPT):
            z.append(row.z_PCA_DR12)
            zsource.append('PCA_DR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)

    z = np.around(z, decimals=4)
    z = [format(i, '.4f') for i in z]

    
    kmag = []
    for i, row in df.iterrows():
        if np.isnan(row['2massMag_k']):
            kmag.append('')
        else:
            kmag.append(str(np.around(row['2massMag_k'], decimals=2)))

    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    snr_ha, snr_hb, snr_civ = [], [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(row.SNR_Ha)
        else:
            snr_ha.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')
    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # j, h, k = True, True, True  
    
    # snr_flag_ha = []
    # edge_flag_ha = []
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha) | (row.Ha == 0):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
     
    # abs_flag_hb = []
    # edge_flag_hb = []
    # snr_flag_hb = ['QSO642'] 
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1))
    
    # edge_flag_civ = []  
    # abs_flag_civ = []
    # snr_flag_civ = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    
    
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['S/N Ha'] = snr_ha
    tnew['S/N Hb'] = snr_hb
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['kmag'] = kmag 
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['zsource'] = zsource
    
    
    tnew.sort('Name')
    
    # tnew.remove_rows(((tnew['S/N Ha'] == '') & (tnew['S/N Hb'] == '')) | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

    print len(tnew)

# sinfoni() 

 
def wht():

    """
    WHT sample 
    
    19 quasars, with Ha 
    """
    
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_CIV_BEST == 0]
    df = df[df.BAL_FLAG != 1]
    
    df = df[(df.INSTR == 'LIRIS')]
    
    names = [row.DR12_NAME.replace('SDSS', '') for i, row in df.iterrows()]
    z = [format(np.around(i, decimals=4), '.4f') for i in df.z_HW_DR7]
    imag = [format(np.around(i, decimals=2), '.2f') for i in df.psfMag_i.values]
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    snr_ha, snr_civ = [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(format(np.around(row.SNR_CIV, decimals=2), '.2f'))
        else:
            snr_civ.append('')
        if row.WARN_Ha == 0:
            snr_ha.append(format(np.around(row.SNR_Ha, decimals=2), '.2f'))
        else:
            snr_ha.append('')

    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
    	exptimes.append(int(float(row.EXPTIME)))
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # j, h, k = True, True, True  
    
    # snr_flag_ha = ['QSO432', 'QSO430', 'QSO437', 'QSO436', 'QSO433', 'QSO435', 'QSO434', 'QSO438']
    # edge_flag_ha = []
    # abs_flag_ha = [] 
    
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Ha == 2) | np.isnan(row.SNR_Ha) | (row.Ha == 0):
    #         snr_ha.append('')
    #     elif i in edge_flag_ha:
    #         if j:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_ha:
    #         if h:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_ha:
    #         if k:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_ha.append(str(np.around(row.SNR_Ha, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_ha.append(np.around(row.SNR_Ha, decimals=1))
     
    
    # edge_flag_civ = []  
    # abs_flag_civ = []
    # snr_flag_civ = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    
    
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['S/N Ha'] = snr_ha
    tnew['Opt. Spec.'] = specopt
    tnew['S/N CIV'] = snr_civ
    tnew['z'] = z
    tnew['imag'] = imag 
    tnew['Radio'] = radio
    
    
    tnew.sort('Name')
    
    # tnew.remove_rows((tnew['S/N Ha'] == '')  | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

# wht() 
    
def niri():

    """
    NIRI
    26 quasars with NIR and SDSS spectra 
    There must be more optical spectra from Joe's binary quasars searches etc. 
    Some have partial Ha coverage but not enough so don't use Ha for these 
    """ 
     
    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Hb == 0]
    df = df[df.WARN_CIV == 0]
    df = df[df.BAL_FLAG != 1]
    
    df = df[(df.INSTR == 'NIRI')]
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
    	exptimes.append(int(float(row.EXPTIME)))
    
    names = [row.DR12_NAME.replace('SDSS', '') for i, row in df.iterrows()]
       
    z = []
    zsource = []
    for idx, row in df.iterrows():
        if 'SDSS' in str(row.SPEC_OPT):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif 'BOSS' in str(row.SPEC_OPT):
            z.append(row.z_PCA_DR12)
            zsource.append('PCA_DR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)

    z = np.around(z, decimals=4)
    z = [format(i, '.4f') for i in z]

    imag = np.around(df.psfMag_i.values, decimals=2)
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    snr_hb, snr_civ = [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')
    
    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # j, h, k = True, True, True  
    
    # abs_flag_hb = []
    # edge_flag_hb = []
    # snr_flag_hb = ['QSO048'] 
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1)) 
    
    # edge_flag_civ = []  
    # abs_flag_civ = ['QSO042', 'QSO036', 'QSO048']
    # snr_flag_civ = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    
    
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['z'] = z
    tnew['imag'] = imag 
    tnew['Opt. Spec.'] = specopt
    tnew['S/N Hb'] = snr_hb
    tnew['S/N CIV'] = snr_civ
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    # tnew['zsource'] = zsource
    
    tnew.sort('Name')
    
    # tnew.remove_rows((tnew['S/N Hb'] == '')  | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

# niri()

def isaac():

    """
    ISAAC sample 
    17 quasars.
    Only cover Hb/OIII region  
    Remove all those with Warn Hb 
    """


    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    df = df[df.WARN_Hb == 0]
    df = df[df.WARN_CIV == 0]
    df = df[df.BAL_FLAG != 1]
    
    df = df[(df.INSTR == 'ISAAC')]
    
    dates = []
    for idx, row in df.iterrows():
    	dates.append(row.DATE)
    
    exptimes = []
    for idx, row in df.iterrows():
    	exptimes.append(int(float(row.EXPTIME)))
    
    names = [row.DR12_NAME.replace('SDSS', '') for i, row in df.iterrows()]
    z = []
    zsource = []
    for idx, row in df.iterrows():
        if 'SDSS' in str(row.SPEC_OPT):
            z.append(row.z_HW_DR7)
            zsource.append('HW')
        elif 'BOSS' in str(row.SPEC_OPT):
            z.append(row.z_PCA_DR12)
            zsource.append('PCA_DR12')
        else:
            z.append(row.z)
            zsource.append(row.z_source)

    z = np.around(z, decimals=4)
    z = [format(i, '.4f') for i in z]
    imag = np.around(df.psfMag_i.values, decimals=2)
    specopt = df.SPEC_OPT.values
    specopt[specopt == 'BOSS+SDSS'] = 'BOSS'
    
    snr_hb, snr_civ = [], []

    for i, row in df.iterrows():
    
        if row.WARN_CIV == 0:
            snr_civ.append(row.SNR_CIV)
        else:
            snr_civ.append('')
        if row.WARN_Hb == 0:
            snr_hb.append(row.SNR_Hb)
        else:
            snr_hb.append('')

    bal = np.zeros(len(df), dtype=np.int)
    bal[np.where((df.BAL_FLAG_DR12 == 1) | (df.BAL_FLAG_ALLEN == 1) | (df.BAL_FLAG_S11 == 1))[0]] = 1
    
    radio = np.asarray(df.RADIO_FLAG, dtype=np.int)
    radio[radio < -99] = -1
    
    # j, h, k = True, True, True  
    
    # abs_flag_hb = []
    # edge_flag_hb = []
    # snr_flag_hb = [] 
    
    # for i, row in df.iterrows():
    #     if (row.WARN_Hb == 2) | np.isnan(row.SNR_Hb):
    #         snr_hb.append('')
    #     elif i in edge_flag_hb:
    #         if j:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_hb:
    #         if h:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_hb:
    #         if k:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_hb.append(str(np.around(row.SNR_Hb, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_hb.append(np.around(row.SNR_Hb, decimals=1)) 
    
    # edge_flag_civ = []  
    # abs_flag_civ = ['QSO088']
    # snr_flag_civ = []
    
    # for i, row in df.iterrows():
    #     if (row.WARN_CIV == 2) | np.isnan(row.SNR_CIV):
    #         snr_civ.append('')
    #     elif i in edge_flag_civ:
    #         if j:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote2}Line close to edge.}')
    #             j = False
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote2}')
    #     elif i in snr_flag_civ:
    #         if h:
    #             print i
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote1}Poor S/N}')
    #             h = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote1}')
    #     elif i in abs_flag_civ:
    #         if k:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footnote{\\label{footnote3}Absorption}')
    #             k = False 
    #         else:
    #             snr_civ.append(str(np.around(row.SNR_CIV, decimals=1)) + '\\footref{footnote3}')
    #     else:
    #         snr_civ.append(np.around(row.SNR_CIV, decimals=1))
    
    
    
    
    tnew = Table()
    
    tnew['Name'] = names
    tnew['Date'] = dates 
    tnew['Exp'] = exptimes
    tnew['z'] = z
    tnew['imag'] = imag 
    tnew['Opt. Spec.'] = specopt
    tnew['S/N Hb'] = snr_hb
    tnew['S/N CIV'] = snr_civ
    # tnew['BAL'] = bal
    tnew['Radio'] = radio 
    tnew['zsource'] = zsource
    
    tnew.sort('Name')
    
    tnew.remove_rows((tnew['S/N Hb'] == '')  | (tnew['S/N CIV'] == ''))
    
    ascii.write(tnew, format='latex')

# isaac()

