#!/bin/bash

#SBATCH -J 02-addresses_07-produce-geocoding-ZIP                 # Job name
#SBATCH -N 1                                                     # Total number of nodes requested
#SBATCH -n 1                                                     # Total number of cores requested
#SBATCH --get-user-env                                           # Retrieve the users login environment
#SBATCH --mem=10G                                                # Server memory requested (per node)
#SBATCH -t 960:00:00                                             # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x_%a.out                                    # Output directory
#SBATCH -e _errors/%x_%a.err                                     # Error directory
#SBATCH --array=1-1000

python ../d02_process-addresses/07_produce-geocoding-ZIP-coordinates.py --ZIP_chunk_idx $SLURM_ARRAY_TASK_ID --geography "blockgroup"