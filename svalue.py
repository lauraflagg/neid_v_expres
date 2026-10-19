import numpy as np
from astropy.constants import c

def triangle(x,line_cent,line_width,y_max=2):
    m = y_max/line_width
    b1 = y_max-m*line_cent
    b2 = y_max+m*line_cent
    
    y = np.zeros_like(x)
    y[x<line_cent]  =  m*x[x<line_cent]+b1
    y[x>=line_cent] = -m*x[x>=line_cent]+b2
    return y

def scale_to_mwo(h_emis,k_emis,r_cont,v_cont,C1,C2):
    return C1*(h_emis+k_emis)/(r_cont+1.16*v_cont)+C2

def calc_s_value(wave,spec,unct,
                 rv=0,snr_cutoff=5,
                 cont_perct=(.75,.95),
                 line_width = 1.09, use_k=False,
                 h_cent=3969.6,h_ord=6,
                 k_cent=3934.7,k_ord=4,
                 v_minx=(3950,3960),v_ord=5,
                 r_minx=(3996,4006),r_ord=7):
    """
    cont_perct : float, [0,1], default 0.75
        percentile above which to use for fitting the continuum
    """
    
    β = rv/c.value
    wave /= np.exp(np.arctanh(β))
    
    # Define Relevant Regions
    h_mask = np.logical_and(wave[h_ord]>h_cent-line_width,wave[h_ord]<h_cent+line_width)
    k_mask = np.logical_and(wave[k_ord]>k_cent-line_width,wave[k_ord]<k_cent+line_width)
    v_mask = np.logical_and(wave[v_ord]>v_minx[0],wave[v_ord]<v_minx[-1])
    r_mask = np.logical_and(wave[r_ord]>r_minx[0],wave[r_ord]<r_minx[-1])
    
    # Find SNR as Spec/Unct (best for low noise)
    v_snr = np.nanmedian(spec[v_ord][v_mask]/unct[v_ord][v_mask])
    r_snr = np.nanmedian(spec[r_ord][r_mask]/unct[r_ord][r_mask])
    
    if (v_snr+r_snr)<(snr_cutoff*2):
        return np.nan
    
    # Find continuum Regions
    min_ind, max_ind = np.array(np.sum(v_mask)*np.array(cont_perct),dtype=int)
    v_cont = np.nanmedian(np.sort(spec[v_ord][v_mask])[min_ind:max_ind])
    min_ind, max_ind = np.array(np.sum(r_mask)*np.array(cont_perct),dtype=int)
    r_cont = np.nanmedian(np.sort(spec[r_ord][r_mask])[min_ind:max_ind])
    
    # Find H Emission
    h_wave = wave[h_ord][h_mask]
    h_spec = spec[h_ord][h_mask]
    h_emis = np.average(h_spec,weights=triangle(h_wave,h_cent,line_width))
    
    if use_k:
        k_wave = wave[k_ord][k_mask]
        k_spec = spec[k_ord][k_mask]
        k_emis = np.average(k_spec,weights=triangle(k_wave,k_cent,line_width))
        
        mwo_sval = scale_to_mwo(h_emis,k_emis,r_cont,v_cont,1.6983556189300908,-0.022251187592364607)
    else:
        mwo_sval = scale_to_mwo(h_emis,np.zeros_like(h_emis),
                                r_cont,v_cont,3.6017600729743022,-0.02323328032406375)
    
    return mwo_sval