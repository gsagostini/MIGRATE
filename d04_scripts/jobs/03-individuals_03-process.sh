#!/bin/bash

#SBATCH -J 03-individuals_03-process                # Job name
#SBATCH --mail-type=ALL                             # Request status by email
#SBATCH --mail-user=gs665@cornell.edu               # Email address to send results to
#SBATCH -N 1                                        # Total number of nodes requested
#SBATCH -n 3                                        # Total number of cores requested
#SBATCH --get-user-env                              # Retrieve the users login environment
#SBATCH --mem=45G                                   # Server memory requested (per node)
#SBATCH -t 960:00:00                                # Time limit (hh:mm:ss)
#SBATCH -p pierson
#SBATCH -w lisbeth
#SBATCH -o _outputs/INFUTOR/%x_%04a.out                     # Output directory
#SBATCH -e _errors/INFUTOR/%x_%04a.err                      # Error directory
#SBATCH --array=600-699%25

python ../d03_process-flows/03_process-INFUTOR-individuals.py --idx $SLURM_ARRAY_TASK_ID --output_chunks_dir 'CRD4/OD_pairs/ADDRID_NOVEMBER/chunks/'