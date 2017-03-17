import pandas as pd 
import numpy as np 
from astropy import units as u
from astropy.coordinates import SkyCoord
from get_nir_spec import get_nir_spec
import astropy.units as u 
from astropy import constants as const
from SpectraTools.mad import mad
from SpectraTools.fit_line import doppler2wave, wave2doppler

# instrument / telescope 
def get_instr(row):

    if row.INSTR == 'FIRE': return 'FIRE/Magellan'
    if row.INSTR == 'GNIRS': return 'GNIRS/Gemini'
    if row.INSTR == 'ISAAC': return 'ISAAC/VLT'
    if row.INSTR == 'LIRIS': return 'LIRIS/WHT'
    if row.INSTR == 'NIRI': return 'NIRI/Gemini'
    if row.INSTR == 'NIRSPEC': return 'NIRSPEC/Keck'
    if row.INSTR == 'SINF': return 'SINFONI/VLT'
    if row.INSTR == 'SINF_KK': return 'SINFONI/VLT'
    if row.INSTR == 'SOFI_JH': return 'SofI/NTT'
    if row.INSTR == 'SOFI_LC': return 'SofI/NTT'
    if row.INSTR == 'TRIPLE': return 'TRIPLESPEC/Hale'
    if row.INSTR == 'TRIPLE_S15': return 'TRIPLESPEC/ARC'
    if row.INSTR == 'XSHOOT': return 'XSHOOTER/VLT' 


def cut_table():


    """
    remove bad entries
    """

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.SPEC_NIR != 'None']
    df = df[~((df.Ha == 0) & (df.Hb == 0) & (df.OIII == 0))]

    # Identified by eye as trash in Ha, Hb and OIII
    # i.e. unable to get redshift or anything useful out
    trash = ['QSO063',
             'QSO317',
             'QSO566',
             'QSO571',
             'QSO639',
             'QSO341',
             'QSO371',
             'QSO377',
             'QSO378',
             'QSO380',
             'QSO386',
             'QSO414',
             'QSO109',
             'QSO220',
             'QSO221',
             'QSO222',
             'QSO225',
             'QSO226',
             'QSO227',
             'QSO229',
             'QSO230',
             'QSO131',
             'QSO148',
             'QSO159',
             'QSO160',
             'QSO165',
             'QSO172',
             'QSO185',
             'QSO192',
             'QSO194',
             'QSO195',
             'QSO197',
             'QSO198',
             'QSO199',
             'QSO200',
             'QSO205',
             'QSO207',
             'QSO210',
             'QSO214',
             'QSO215',
             'QSO219',
             'QSO238',
             'QSO244',
             'QSO267',
             'QSO272',
             'QSO273',
             'QSO275',
             'QSO276',
             'QSO278',
             'QSO292',
             'QSO294',
             'QSO296',
             'QSO305',
             'QSO306',
             'QSO312',
             'QSO303',
             'QSO387',
             'QSO300',
             'QSO084',
             'QSO204',
             'QSO183',
             'QSO085',
             'QSO235',
             'QSO142',
             'QSO208',
             'QSO095',
             'QSO092',
             'QSO047',
             'QSO316']

    df.drop(trash, inplace=True)

    return df 

def make_table(): 

    """
    Sumamry table for chapter 2
    """

   
    df = cut_table() 

    df.sort_values('RA', inplace=True)


    for idx, row in df.iterrows():
            
        wav, dw, flux, err = get_nir_spec(row.NIR_PATH, row.INSTR)
        dw = np.diff(np.log10(wav)).mean()
    
        dv = const.c.to('km/s') * (1. - 10. ** -dw)
        df.set_value(idx, 'dv_PIXEL', np.around(dv, decimals=0).value)  
    
        #-------------------------------
        wav_range = '{0:.2f}'.format((wav.min()*u.AA).to(u.um).value) + '-' + \
                    '{0:.2f}'.format((wav.max()*u.AA).to(u.um).value) 
        df.set_value(idx, 'WAV_RANGE', wav_range)  

        #---------------------------------------
        wav = wav / (1.0 + row.z_IR)

        instr = row.INSTR
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
        p = q[df.ix[row.name, 'NUM']]    
    
        snr = np.zeros(4)
    
        # Ha -----------------------------------
    
        w0=6564.89*u.AA
        vdat = wave2doppler(wav*u.AA, w0).value
    
        # index of region for continuum fit 
        continuum_region = p.ha_continuum_region 
        if continuum_region[0].unit == (u.km/u.s):
            continuum_region[0] = doppler2wave(continuum_region[0], w0)
        if continuum_region[1].unit == (u.km/u.s):
            continuum_region[1] = doppler2wave(continuum_region[1], w0)
    
        blue_mask = (wav < continuum_region[0][0].value) | (wav > continuum_region[0][1].value)
        red_mask = (wav < continuum_region[1][0].value) | (wav > continuum_region[1][1].value) 
    
        maskout = p.ha_maskout 
    
        if maskout is not None:
    
            for item in maskout:
    
                if maskout.unit == (u.km/u.s):  
    
                    mask = (vdat > item[0].value) & (vdat < item[1].value) 
    
                elif maskout.unit == u.AA: 
    
                    mask = (wav.value > item[0].value) & (wav < item[1].value)  
    
            red_mask = red_mask | mask 
            blue_mask = blue_mask | mask 
    
        red_snr = np.nanmedian(flux[~red_mask] / np.nanstd(flux[~red_mask]))
        blue_snr = np.nanmedian(flux[~blue_mask] / np.nanstd(flux[~blue_mask]))
        snr[:2] = [red_snr, blue_snr]
    
        # Hb -----------------------------------
    
        w0=4862.721*u.AA
        vdat = wave2doppler(wav*u.AA, w0).value
    
        # index of region for continuum fit 
        continuum_region = p.hb_continuum_region 
        if continuum_region[0].unit == (u.km/u.s):
            continuum_region[0] = doppler2wave(continuum_region[0], w0)
        if continuum_region[1].unit == (u.km/u.s):
            continuum_region[1] = doppler2wave(continuum_region[1], w0)
    
        blue_mask = (wav < continuum_region[0][0].value) | (wav > continuum_region[0][1].value)
        red_mask = (wav < continuum_region[1][0].value) | (wav > continuum_region[1][1].value) 
    
        maskout = p.hb_maskout 
    
        if maskout is not None:
    
            for item in maskout:
    
                if maskout.unit == (u.km/u.s):  
    
                    mask = (vdat > item[0].value) & (vdat < item[1].value) 
    
                elif maskout.unit == u.AA: 
    
                    mask = (wav > item[0].value) & (wav < item[1].value)  
    
            red_mask = red_mask | mask 
            blue_mask = blue_mask | mask 
    
        red_snr = np.nanmedian(flux[~red_mask]) / mad(flux[~red_mask])
        blue_snr = np.nanmedian(flux[~blue_mask]) / mad(flux[~blue_mask])
        snr[2:] = [red_snr, blue_snr]

        df.set_value(idx, 'SNR', np.nanmax(snr))   

        print idx 


        
    # These two have SDSS spectra attached so wav_range is wrong 
    df.loc[df.INSTR == 'FIRE', 'WAV_RANGE'] = '0.80-2.50'
    df.loc[df.INSTR == 'TRIPLE_S15', 'WAV_RANGE'] = '0.95-2.46'   
    
    c = SkyCoord(ra=df.RA, dec=df.DEC)
    df['RA'] = c.ra.to_string(unit=u.hourangle, precision=2, alwayssign=True, pad=True)
    df['DEC'] = c.dec.to_string(precision=2, alwayssign=True, pad=True)
    
    df.DATE.replace(to_replace='None', value='yyyy-mm-dd', inplace=True)
    
    df['INSTR'] = df.apply(get_instr, axis=1) 

    df['dv_PIXEL'] = df['dv_PIXEL'].round(decimals=1)
    df['SNR'] = df['SNR'].round(decimals=1)

    df.rename(columns={'OIII_FIT_Z_FULL_OIII_PEAK': 'z_OIII', 
    	               'OIII_FIT_HA_Z': 'z_Ha',
    	               'OIII_FIT_HB_Z': 'z_Hb'}, 
    	      inplace=True)

    #-----------------------------------------------------

    """
    If OIII_TEMPLATE is true then OIII redshift is fixed
    """

    # df.loc[df['OIII_TEMPLATE'] == True, 'z_OIII'] = np.nan 
    # df.loc[df['OIII_Z_FLAG'].isin([-1, 0, 3, 4, 5]), 'z_OIII'] = np.nan  
    # df.loc[df['OIII_FIT_HA_Z_FLAG'].isin([-1, 0]), 'z_Ha'] = np.nan  
    # df.loc[df['OIII_FIT_HB_Z_FLAG'].isin([-1, 0]), 'z_Hb'] = np.nan 

    # df['z_OIII'] = df['z_OIII'].round(decimals=4)
    # df['z_Ha'] = df['z_Ha'].round(decimals=4)
    # df['z_Hb'] = df['z_Hb'].round(decimals=4)

    df['z_IR'] = df['z_IR'].round(decimals=2)

    cols = ['ID',
            'DATE',
            'RA',
            'DEC',
            'INSTR',
            'WAV_RANGE',
            'dv_PIXEL',
            'SNR',
            'z_IR']
    
    df = df[cols]

    return df 

def make_table_long():

    """
    Supplementary information 
    """

    df = cut_table() 

    df.loc[df.SPEC_OPT == 'BOSS+SDSS', 'SPEC_OPT'] = 'BOSS'
    df.loc[df.SPEC_OPT == 'None', 'SPEC_OPT'] = np.nan 
    # if true use XSHOOTER CIV measurements - 
    # either don't have SDSS/BOSS but do have XSHOOTER, 
    # or have both but XSHOOTER warning less than or equal to optical spectra warning 
    filt = ((df.WARN_CIV == -1) & (df.WARN_CIV_XSHOOT != -1)\
            & (df.INSTR == 'XSHOOT')) | ((df.WARN_CIV != -1)\
            & (df.WARN_CIV_XSHOOT != -1) & (df.WARN_CIV_XSHOOT <= df.WARN_CIV) & (df.INSTR == 'XSHOOT')) 
    df.loc[filt, 'SPEC_OPT'] = 'XSHOOTER'

    df.RADIO_FLAG.fillna(-1, inplace=True)
    df['RADIO_FLAG'] = df.RADIO_FLAG.astype(int)


    columns = [] 
    
    columns.append('ID')
    columns.append('RA_DEG')
    columns.append('DEC_DEG')
    columns.append('SPEC_OPT')
    columns.append('BAL_FLAG')
    columns.append('RADIO_FLAG')
    columns.append('psfMag_u')
    columns.append('psfMagErr_u')
    columns.append('psfMag_g')
    columns.append('psfMagErr_g')
    columns.append('psfMag_r')
    columns.append('psfMagErr_r')
    columns.append('psfMag_i')
    columns.append('psfMagErr_i')
    columns.append('psfMag_z')
    columns.append('psfMagErr_z')
    # SuperCosmos   
    columns.append('scormagb')
    columns.append('scormagr2')
    columns.append('scormagi')
    columns.append('2massMag_j')
    columns.append('2massMagErr_j')
    columns.append('2massMag_h')
    columns.append('2massMagErr_h')
    columns.append('2massMag_k')
    columns.append('2massMagErr_k')
    columns.append('UKIDSS_YAperMag3')
    columns.append('UKIDSS_YAperMag3Err')
    columns.append('UKIDSS_J_1AperMag3')
    columns.append('UKIDSS_J_1AperMag3Err')
    columns.append('UKIDSS_HAperMag3')
    columns.append('UKIDSS_HAperMag3Err')
    columns.append('UKIDSS_KAperMag3')
    columns.append('UKIDSS_KAperMag3Err')    
    columns.append('VHS_YAperMag3')
    columns.append('VHS_YAperMag3Err')
    columns.append('VHS_JAperMag3')
    columns.append('VHS_JAperMag3Err')
    columns.append('VHS_HAperMag3')
    columns.append('VHS_HAperMag3Err')
    columns.append('VHS_KAperMag3')
    columns.append('VHS_KAperMag3Err')
    columns.append('Viking_ZAperMag3')
    columns.append('Viking_ZAperMag3Err')
    columns.append('Viking_YAperMag3')
    columns.append('Viking_YAperMag3Err')
    columns.append('Viking_JAperMag3')
    columns.append('Viking_JAperMag3Err')
    columns.append('Viking_HAperMag3')
    columns.append('Viking_HAperMag3Err')
    columns.append('Viking_KsAperMag3')
    columns.append('Viking_KsAperMag3Err')
    columns.append('WISE_W1MPRO')
    columns.append('WISE_W1SIGMPRO')
    columns.append('WISE_W2MPRO')
    columns.append('WISE_W2SIGMPRO')
    columns.append('WISE_W3MPRO')
    columns.append('WISE_W3SIGMPRO')
    columns.append('WISE_W4MPRO')
    columns.append('WISE_W4SIGMPRO')    

    df = df[columns]

    df.rename(columns={'RA_DEG': 'RA',
                       'DEC_DEG': 'DEC',
                       'BAL_FLAG': 'BAL',
                       'RADIO_FLAG': 'RADIO',
                       'psfMag_u': 'SDSS_psfMag_u', 
                       'psfMagErr_u': 'SDSS_psfMagErr_u', 
                       'psfMag_g': 'SDSS_psfMag_g', 
                       'psfMagErr_g': 'SDSS_psfMagErr_g', 
                       'psfMag_r': 'SDSS_psfMag_r', 
                       'psfMagErr_r': 'SDSS_psfMagErr_r', 
                       'psfMag_i': 'SDSS_psfMag_i', 
                       'psfMagErr_i': 'SDSS_psfMagErr_i', 
                       'psfMag_z': 'SDSS_psfMag_z', 
                       'psfMagErr_z': 'SDSS_psfMagErr_z', 
                       'scormagb': 'SuperCosmos_scormagb', 
                       'scormagr2': 'SuperCosmos_scormagr2', 
                       'scormagi': 'SuperCosmos_scormagi', 
                       '2massMag_j': '2MASS_Mag_j', 
                       '2massMagErr_j': '2MASS_MagErr_j', 
                       '2massMag_h': '2MASS_Mag_h', 
                       '2massMagErr_h': '2MASS_MagErr_h', 
                       '2massMag_k': '2MASS_Mag_k', 
                       '2massMagErr_k': '2MASS_MagErr_k', 
                       'UKIDSS_YAperMag3': 'UKIDSS_YAperMag3', 
                       'UKIDSS_YAperMag3Err': 'UKIDSS_YAperMag3Err', 
                       'UKIDSS_J_1AperMag3': 'UKIDSS_J_1AperMag3', 
                       'UKIDSS_J_1AperMag3Err': 'UKIDSS_J_1AperMag3Err', 
                       'UKIDSS_HAperMag3': 'UKIDSS_HAperMag3', 
                       'UKIDSS_HAperMag3Err': 'UKIDSS_HAperMag3Err', 
                       'UKIDSS_KAperMag3': 'UKIDSS_KAperMag3', 
                       'UKIDSS_KAperMag3Err': 'UKIDSS_KAperMag3Err',     
                       'VHS_YAperMag3': 'VHS_YAperMag3', 
                       'VHS_YAperMag3Err': 'VHS_YAperMag3Err', 
                       'VHS_HAperMag3': 'VHS_HAperMag3', 
                       'VHS_HAperMag3Err': 'VHS_HAperMag3Err', 
                       'VHS_JAperMag3': 'VHS_JAperMag3', 
                       'VHS_JAperMag3Err': 'VHS_JAperMag3Err', 
                       'VHS_KAperMag3': 'VHS_KAperMag3', 
                       'VHS_KAperMag3Err': 'VHS_KAperMag3Err', 
                       'Viking_ZAperMag3': 'Viking_ZAperMag3', 
                       'Viking_ZAperMag3Err': 'Viking_ZAperMag3Err', 
                       'Viking_YAperMag3': 'Viking_YAperMag3', 
                       'Viking_YAperMag3Err': 'Viking_YAperMag3Err', 
                       'Viking_JAperMag3': 'Viking_JAperMag3', 
                       'Viking_JAperMag3Err': 'Viking_JAperMag3Err', 
                       'Viking_HAperMag3': 'Viking_HAperMag3', 
                       'Viking_HAperMag3Err': 'Viking_HAperMag3Err', 
                       'Viking_KsAperMag3': 'Viking_KsAperMag3', 
                       'Viking_KsAperMag3Err': 'Viking_KsAperMag3Err', 
                       'WISE_W1MPRO': 'WISE_W1MPRO', 
                       'WISE_W1SIGMPRO': 'WISE_W1SIGMPRO', 
                       'WISE_W2MPRO': 'WISE_W2MPRO', 
                       'WISE_W2SIGMPRO': 'WISE_W2SIGMPRO', 
                       'WISE_W3MPRO': 'WISE_W3MPRO', 
                       'WISE_W3SIGMPRO': 'WISE_W3SIGMPRO', 
                       'WISE_W4MPRO': 'WISE_W4MPRO', 
                       'WISE_W4SIGMPRO': 'WISE_W4SIGMPRO'}, inplace=True)

    return df 





if __name__ == '__main__':


    df = make_table(example_table=False)
    df = make_table_long()



