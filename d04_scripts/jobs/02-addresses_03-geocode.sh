#!/bin/bash

#SBATCH -J geocode                                  # Job name
#SBATCH --mail-user=gs665@cornell.edu               # Email address to send results to
#SBATCH -N 1                                        # Total number of nodes requested
#SBATCH -n 1                                        # Total number of cores requested
#SBATCH --get-user-env                              # Retrieve the users login environment
#SBATCH --mem=1G                                    # Server memory requested (per node)
#SBATCH -t 960:00:00                                # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x_%a.out                       # Output directory
#SBATCH -e _errors/%x_%a.err                        # Error directory
#SBATCH --array=1-32%10

#If you are using a file with indices:
#idx=$( awk "NR==$SLURM_ARRAY_TASK_ID" /share/pierson/gs665/migration_flows/d02_notebooks/retry2.txt)
python ../d02_process-addresses/03_geocode.py --base_idx 0 --array_idx $SLURM_ARRAY_TASK_ID --geocoding_subdir "Census_Aug8"