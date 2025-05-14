#!/bin/bash

#SBATCH -J 03-individuals_04-process-yearly         # Job name
#SBATCH --mail-type=ALL                             # Request status by email
#SBATCH --mail-user=gs665@cornell.edu               # Email address to send results to
#SBATCH -N 1                                        # Total number of nodes requested
#SBATCH -n 1                                        # Total number of cores requested
#SBATCH --get-user-env                              # Retrieve the users login environment
#SBATCH --mem=150G                                  # Server memory requested (per node)
#SBATCH -t 960:00:00                                # Time limit (hh:mm:ss)
#SBATCH -p pierson
#SBATCH -w lisbeth
#SBATCH -o _outputs/INFUTOR/%x_%03a.out                     # Output directory
#SBATCH -e _errors/INFUTOR/%x_%03a.err                      # Error directory
#SBATCH --array=2-122

python ../d03_process-flows/04_create-yearly-matrix.py --delta_year $SLURM_ARRAY_TASK_ID --ADDRID_dir 'CRD4/OD_pairs/ADDRID_NOVEMBER/'