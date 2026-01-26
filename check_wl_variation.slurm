#!/bin/bash 
## Submit job to our class's allocation
##SBATCH --account=lvf5358
## Alternatively could comment out the two lines above (by adding a second # at the beginning of each line)
# and uncomment the lines below (by removing one #) to use the "Open" allocation
#SBATCH --partition=basic 

## Time requested: 0 hours, 1 minutes, 0 seconds
#SBATCH --time=72:00:00 

## Ask for one core on one node
#SBATCH --nodes=1 
#SBATCH --ntasks=1 

## Promise that each processor will use no more than 1GB of RAM
#SBATCH --mem-per-cpu=2GB

## Save STDOUT and STDERR into one file (%j will expand to become the SLURM job id)
#SBATCH --output=reprocess_wls_%A_%a.log

## Specificy job name, so easy to find using squeue –u
#SBATCH --job-name=reprocess_wls


## Uncomment next two lines (by removing one of #'s in each line) and replace with your email if you want to be notifed when jobs start and stop
#SBATCH --mail-user=lvf5358@psu.edu
## Ask for emails when jobs begins, ends or fails (options are ALL, NONE, BEGIN, END, FAIL)
#SBATCH --mail-type=ALL

echo "Starting job $SLURM_JOB_NAME"
date

echo "loading python..."
#module load python
module load anaconda
conda activate astro
#conda list




echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR            # Change into directory where job was submitted from

date
python python intra_day_wavelength_variation_to_csv.py --instrument NEID --data-dir "/storage/group/ebf11/default/pipeline/neid_solar/data/v1.4/L2/" --out-csv neid_intra_day_summary_all.csv --reference median --recursive
echo "done!"
date

