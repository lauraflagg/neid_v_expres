import os

data_dir="/storage/home/lvf5358/scratch/"

files=os.listdir(data_dir)

fits_files=[fi for fi in files if fi.endswith('.fits')]

#file name format example: expresL2_202405265255.fits
for file in fits_files:
    instrument=file.split('_')[0].split('L')[0]
    year=file.split('_')[1][:4]
    month=file.split('_')[1][4:6]
    day=file.split('_')[1][6:8]

    #destination directiory should be instrument/year/month/day
    dest_dir=os.path.join(data_dir, instrument, year, month, day)

    #check if destination directory exists, if not create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    #move the file
    src_path=os.path.join(data_dir, file)
    dest_path=os.path.join(dest_dir, file)
    os.rename(src_path, dest_path)  
