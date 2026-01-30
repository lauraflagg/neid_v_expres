
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams.update({'errorbar.capsize': 3})
plt.rcParams['xtick.labelsize']= 13
plt.rcParams['ytick.labelsize']= 13
plt.rcParams.update({'axes.labelsize': 16})

# %%
from astropy.time import Time
from astropy.table import  Table

# %%
from astropy import constants, units

c_kms=constants.c.value/10**3

# %%
import h5py
from astropy.io import ascii

# %%
import pickle

# %%
from lmfit import Model



def gaussian_w_slope(x, amp, cen, wid, bas,slope=0):
    "1-d gaussian: gaussian(x, amp, cen, wid,bas,slope)"
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2)) + bas+x*slope

# %%
def wl2vel(wl_arr,central_wl=None):
    '''returns vel in km/s
    
        inputs are wl_arr and central_wl'''
    if central_wl==None:
        central_wl=np.average(wl_arr)

    #print(c_kms)
    vel_arr=(wl_arr-central_wl)*c_kms/central_wl
    return vel_arr

# %%
#read coadds from coadds_outputs.h5 

data=h5py.File('coadds_outputs.h5','r')

# %%
#find at which indices there are data for both NEID coadds and EXPRES coadds
neid_indices=[]
expres_indices=[]
overlap_mids=[]

for i,key in  enumerate(data['NEID']['coadds'].keys()):
    #check if there is data for that key in NEID that is not empty
    if data['NEID']['coadds'][key] is not None and len(data['NEID']['coadds'][key]) > 0:
        #print('NEID coadds found for key:', key)
        #check if that key is also in EXPRES coadds
        if key in data['EXPRES']['coadds'].keys():
            #check if there is data for that key in EXPRES that is not empty
            if data['EXPRES']['coadds'][key] is not None and len(data['EXPRES']['coadds'][key]) > 0:
                #print('EXPRES coadds found for key:', key)
                neid_indices.append(key)
                expres_indices.append(key)
                overlap = data['overlaps'][i]
                mid=(overlap[0]+overlap[1])/2
                overlap_mids.append(Time(mid,format='jd').isot)

# %%
#check neid_indices = expres_indices and that their length is the same as overlap_mids
if neid_indices == expres_indices and len(neid_indices) == len(overlap_mids):

    print('everything matches:', neid_indices)
else:
    print('NEID and EXPRES indices do not match')

# %%
#read in csv file with line list
line_list_loc="RvLineList_allowBlends=0_overlapCutoff=5.0e-6_blend_RV_cutoff=2.0_depthCutoff=0.02_rejectTelluricSlope=2000.0_badLineFilter=none_quant=95_nbin=1_DP=true_binParam=depth_n=1_long_output=true_VACUUM.csv"

# %%
line_list=ascii.read(line_list_loc)

# %%
gmod=Model(gaussian_w_slope)

# %%


def fit_lines(w,s,e,line_list,plot=False,wl_min=6000,wl_max=6030,fit_range=0.7,dist_from_edge=500,plotbad=False,detrend=True):
    line_fits=[]
    '''fits lines in line_list to the spectrum s with wavelengths w
    
        inputs are:
            w: 2D array of wavelengths (orders x pixels)
            s: 2D array of spectrum (orders x pixels)
            line_list: astropy table with columns 'lambda' and 'depth'
    '''
    for i,l in enumerate(line_list['lambda'].value):
        if l<wl_max and l>wl_min:
            #print(l,line_list['depth'][i])
            #find all the orders that contain this wavelength, and the line is more than 500 pixels from the edge of the order  
            orders_with_line=[] 

            for o in range(w.shape[0]):
                if l>w[o].min()+dist_from_edge*(w[o][1]-w[o][0]) and l<w[o].max()-dist_from_edge*(w[o][1]-w[o][0]):
                    orders_with_line.append(o)
            #print("orders with line:",orders_with_line)
            #for each order, convert the wavelength to velocity, centered on the line wavelength and then fit a guassian to the spectrum
            #only fit within 2 angstroms of the line center
            for o in orders_with_line: 
                wl_order=w[o]
                s_order=s[o]
                e_order=s[o]
                vel_order=wl2vel(wl_order,central_wl=l)
                #select only the points within 1 angstroms of the line center and the spectrum is not NaN
                mask=(wl_order>l-fit_range)&(wl_order<l+fit_range)&(~np.isnan(s_order))
                wl_fit=wl_order[mask]
                s_fit=s_order[mask]
                vel_fit=vel_order[mask]
                e_fit=e_order[mask]
                weights_fit=1/e_fit**2
                #save the pixel index of the line center
                line_center_index=np.argmin(np.abs(wl_order-l))
                #initial guess for the gaussian parameters
                amp_init=line_list['depth'][i]*np.sum(s_fit)*-1
                cen_init=0
                wid_init=3.0
                bas_init=np.median(s_fit)
                slope_init=0
                params=gmod.make_params(amp=dict(value=amp_init,max=0),cen=dict(value=cen_init,min=-2,max=2),wid=dict(value=wid_init,min=0),bas=bas_init,slope=slope_init)
                #fit the gaussian to the data
                try:
                    result0=gmod.fit(s_fit,params,x=vel_fit,weights=weights_fit)
                    #now refit after dividing by the best fit baseline + slope
                    if detrend:
                        s_fit_detrended=s_fit/(result0.params['bas'].value+vel_fit*result0.params['slope'].value)
                        e_fit_detrended=e_fit/(result0.params['bas'].value+vel_fit*result0.params['slope'].value)
                        weights_fit_detrended=1/e_fit_detrended**2
                        
                        params=gmod.make_params(amp=dict(value=amp_init,max=0),cen=dict(value=cen_init,min=-2,max=2),wid=dict(value=result0.params['wid'].value,min=0),bas=0,slope=0)
                        result=gmod.fit(s_fit_detrended,params,x=vel_fit,weights=weights_fit)
                    else:
                        result=result0
                    
                except:
                    print(f"Fit failed for line at {l} Angstroms in order {o}")
                    #print(wl_fit)
                    #print(s_fit)
                    continue               
                #print(result.fit_report())
                if plot:
                    #plot the data and the fit
                    plt.figure()
                    plt.plot(vel_fit,s_fit_detrended,label='data')
                    plt.plot(vel_fit,result.best_fit,label='fit')
                    plt.xlabel('Velocity (km/s)')
                    plt.ylabel('Flux')
                    plt.title(f'Order {o} Line at {l} Angstroms')
                    plt.legend()
                    plt.show()
                if plotbad:
                    if result.rsquared<0.8:
                        plt.figure()
                        plt.plot(vel_fit,s_fit_detrended,label='data')
                        plt.plot(vel_fit,result.best_fit,label='fit')
                        plt.xlabel('Velocity (km/s)')
                        plt.ylabel('Flux')
                        plt.title(f'Order {o} Line at {l} Angstroms')
                        plt.legend()
                        plt.show()
                        
                
                

                
                #save the fit results to a list
                line_fits.append({'line_wl':l,'order':o,'amp':result.params['amp'].value,'amp_err':result.params['amp'].stderr,    
                                  'cen':result.params['cen'].value,
                                  'cen_err':result.params['cen'].stderr,
                                  'wid':result.params['wid'].value,
                                  'wid_err':result.params['wid'].stderr,
                                  'bas':result.params['bas'].value,
                                  'bas_err':result.params['bas'].stderr,
                                  'slope':result.params['slope'].value,'chi2':result.chisqr,
                                  'bic':result.bic,'r':result.rsquared,
                                  'line_center_index':line_center_index,
                                  'line_depth':line_list['depth'][i]})
    
    return line_fits
            

# %%
dist_from_edge=500

# %%
def get_fit_results(data,neid_indices,line_list,fit_range,plot=False,plotbad=False,test=False,testnum=0,dist_from_edge=500):
    fit_results=[]
#loop through the overlapping indices and fit lines to the NEID and EXPRES coadds
    for i,idx in enumerate(neid_indices[testnum:]):
    
        w_NEID=np.array(data['NEID']['wls'][idx])
        s_NEID=np.array(data['NEID']['coadds'][idx])
        e_NEID=np.array(data['NEID']['uncs'][idx])
    
        w_EXPRES=np.array(data['EXPRES']['wls'][idx])
        s_EXPRES=np.array(data['EXPRES']['coadds'][idx])
        e_EXPRES=np.array(data['EXPRES']['uncs'][idx])
    
        upper=np.min((np.max(w_NEID),np.max(w_EXPRES)))
        lower=np.max((np.min(w_NEID),np.min(w_EXPRES)))
    
        #find line fits in both spectra that are at least 500 pixels from the edge of at least one order
        lines_to_use={}
    
        cols_to_keep=['lambda','depth']
        for col in cols_to_keep:
            lines_to_use[col]=[]
        for row in line_list:
            l=row['lambda']
        
            if l<upper and l>lower:
                #find all the orders that contain this wavelength, and the line is more than 500 pixels from the edge of the order  
                #if line is not atleast 500 pixels from the edge of at least one order in both NEID and EXPRES, include it in lines_to_use
                neid_ok=False
                expres_ok=False
                for o in range(w_NEID.shape[0]):
                    if l>w_NEID[o].min()+dist_from_edge*(w_NEID[o][1]-w_NEID[o][0]) and l<w_NEID[o].max()-dist_from_edge*(w_NEID[o][1]-w_NEID[o][0]):
                        neid_ok=True
                for o in range(w_EXPRES.shape[0]):
                    if l>w_EXPRES[o].min()+dist_from_edge*(w_EXPRES[o][1]-w_EXPRES[o][0]) and l<w_EXPRES[o].max()-dist_from_edge*(w_EXPRES[o][1]-w_EXPRES[o][0]):
                        expres_ok=True
                if neid_ok and expres_ok:
                    for col in cols_to_keep:
                        lines_to_use[col].append(row[col])
        lines_to_use=Table(lines_to_use)
    
        #print number of fitting lines for index idx
        #print(lines_to_use)
        print(f"Fitting {len(lines_to_use)} lines for index {idx} between {lower:.1f} and {upper:.1f} Angstroms")
        print("Now fitting for NEID...")
        line_fits_NEID=fit_lines(w_NEID,s_NEID,e_NEID,lines_to_use,plot=plot,wl_min=lower,wl_max=upper,fit_range=fit_range,dist_from_edge=dist_from_edge,plotbad=plotbad)
        if test==False:
            print("Now fitting for EXPRES...")
            line_fits_EXPRES=fit_lines(w_EXPRES,s_EXPRES,e_EXPRES,lines_to_use,plot=plot,wl_min=lower,wl_max=upper,fit_range=fit_range,dist_from_edge=dist_from_edge,plotbad=plotbad)
            fit_results.append({'index':idx,'NEID_fits':line_fits_NEID,'EXPRES_fits':line_fits_EXPRES,'overlap_mid':overlap_mids[i]})
            #
        else:
            fit_results.append(line_fits_NEID)
    return fit_results


# %%
fit_results=get_fit_results(data,neid_indices,line_list,fit_range=.15,dist_from_edge=200)

# %%
#save fit results to pickele file

with open('line_fit_results_neid_expres_overlap_2024_w_err_rangep15.pkl','wb') as f:
    pickle.dump(fit_results,f)

# %%



