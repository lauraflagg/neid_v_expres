# %%
NEID_dir="/storage/group/ebf11/default/pipeline/neid_solar/data/v1.4/L2/"
EXPRES_dir="/storage/home/lvf5358/scratch/expres"

# %%
import os

# %%
import pickle

# %%
from astropy.io import fits

# %%
#explore all fits files in NEID directory and subdirectories
#get list of all NEID observation dates (MJDs) and file locations
NEID_dates=[]
NEID_file_locs=[]
for root, dirs, files in os.walk(NEID_dir):
    for file in files:
        if file.endswith(".fits"):

            file_path=os.path.join(root, file)
            try:
                hdul=fits.open(file_path)
                head=hdul[0].header

                bjd=(head['SSBJD077']+head['SSBJD160'])/2
                NEID_dates.append(bjd)
                NEID_file_locs.append(file_path)
                hdul.close()
            except:
                print(f"Could not open {file_path}")

# %%

#explore all fits files in EXPRES directory and subdirectories
#get list of all EXPRES observation dates (MJDs) and file locations
EXPRES_dates=[]
EXPRES_file_locs=[]
for root, dirs, files in os.walk(EXPRES_dir):
    for file in files:
        if file.endswith(".fits"):
            file_path=os.path.join(root, file)
            
            try:
                hdul=fits.open(file_path)
                head=hdul[2].header

                bjd=head['HIERARCH wtd_mdpt']
                EXPRES_dates.append(bjd)
                EXPRES_file_locs.append(file_path)
                hdul.close()
            except:
                print(f"Could not open {file_path}")

# %%
print(NEID_dates)
print(NEID_file_locs)

# %%
EXPRES_dates

# %%
#save lists as pickele files

dates_and_dirs={'NEID_dates':NEID_dates, 'NEID_file_locs':NEID_file_locs,
                'EXPRES_dates':EXPRES_dates, 'EXPRES_file_locs':EXPRES_file_locs}   

with open('dates_and_dirs.pkl', 'wb') as f:
    pickle.dump(dates_and_dirs, f)

# %%


# %%



