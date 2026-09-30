import os
from astropy.io import fits
from astropy.table import Table
import numpy as np

data_dir="/storage/home/lvf5358/scratch/"

inst_dir='expres'

folder=data_dir+inst_dir

years0=os.listdir(folder)
years=[y for y in years0 if os.path.isdir(os.path.join(folder, y))]

dates_paths=[]
dates=[]
for year in years:
    year_path = os.path.join(folder, year)
    months = [m for m in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, m))]
    for month in months:
        month_path = os.path.join(year_path, month)
        days = [d for d in os.listdir(month_path) if os.path.isdir(os.path.join(month_path, d))]
        for day in days:
            day_path = os.path.join(month_path, day)
            dates_paths.append(day_path)
            date=year+month+day
            dates.append(date)

express_headers = ['BARYMJD','S-VALUE','HALPHA','HWIDTH','BIS']

out_individual_files={}
out_individual_files['files']=[]
for header in express_headers:
    out_individual_files[header] = []



out_combined_files = {}
for header in express_headers:
    out_combined_files[header] = np.zeros(len(dates))

for i in range(len(dates)):
    files = os.listdir(dates_paths[i])
    fits_files = [f for f in files if f.endswith('.fits')]
    temp_head_sums = {}
    for header_name in express_headers:
        temp_head_sums[header_name] = 0
    nfiles = len(fits_files)
    for fits_file in fits_files:
        out_individual_files['files'].append(fits_file)
        with fits.open(os.path.join(dates_paths[i], fits_file)) as hdul:
            header = hdul[1].header
            # Do something with the header, e.g., print it or extract specific keywords
            for header_name in express_headers:
                out_individual_files[header_name].append(header[header_name])
                temp_head_sums[header_name] += header.get(header_name, None)
    for header_name in express_headers:
        out_combined_files[header_name][i] = temp_head_sums[header_name] / nfiles if nfiles > 0 else None


#write the combined files to disk using astropy.table.Table
combined_table = Table()
for header_name in express_headers:
    combined_table[header_name] = out_combined_files[header_name]
combined_table[dates] = dates

combined_table.write('combined_expres_headers.csv', format='ascii.csv')

#write the individual files to disk using astropy.table.Table

individual_table = Table()
for header_name in express_headers:
    individual_table[header_name] = out_individual_files[header_name]
individual_table.write(f'individual_expres_headers.csv', format='ascii.csv')
