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


def make_table(): 


    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0) 
    df = df[df.SPEC_NIR != 'None']
    df = df[~((df.Ha == 0) & (df.Hb == 0) & (df.OIII == 0))]


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

    df.loc[df['OIII_TEMPLATE'] == True, 'z_OIII'] = np.nan 
    df.loc[df['OIII_Z_FLAG'].isin([-1, 0, 3, 4, 5]), 'z_OIII'] = np.nan  
    df.loc[df['OIII_FIT_HA_Z_FLAG'].isin([-1, 0]), 'z_Ha'] = np.nan  
    df.loc[df['OIII_FIT_HB_Z_FLAG'].isin([-1, 0]), 'z_Hb'] = np.nan 

    df['z_OIII'] = df['z_OIII'].round(decimals=4)
    df['z_Ha'] = df['z_Ha'].round(decimals=4)
    df['z_Hb'] = df['z_Hb'].round(decimals=4)


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

    cols = ['ID',
            'DATE',
            'RA',
            'DEC',
            'INSTR',
            'WAV_RANGE',
            'dv_PIXEL',
            'SNR',
            'z_OIII',
            'z_Ha',
            'z_Hb']

    df = df[cols]

    return df 

if __name__ == '__main__':


    df = make_table()

