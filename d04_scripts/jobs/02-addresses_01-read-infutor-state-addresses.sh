#!/bin/bash

#SBATCH -J 02-addresses_01-read-infutor-state-addresses                   # Job name
#SBATCH -N 1                                # Total number of nodes requested
#SBATCH -n 1                                # Total number of cores requested
#SBATCH --get-user-env                      # Retrieve the users login environment
#SBATCH --mem=10G                           # Server memory requested (per node)
#SBATCH -t 960:00:00                        # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x_%a.out
#SBATCH -e _errors/%x_%a.err  
#SBATCH --array=1-52%10

state=$( awk "NR==$SLURM_ARRAY_TASK_ID" /share/pierson/gs665/migration_flows/d01_data/states.txt)

python ../d02_process-addresses/01_read-infutor-state-addresses.py --state $state