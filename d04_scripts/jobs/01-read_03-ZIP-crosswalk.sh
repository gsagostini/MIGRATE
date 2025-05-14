#!/bin/bash

#SBATCH -J 01-read_03-ZIP-crosswalk         # Job name
#SBATCH -N 1                                # Total number of nodes requested
#SBATCH -n 16                               # Total number of cores requested
#SBATCH --get-user-env                      # Retrieve the users login environment
#SBATCH --mem=100G                          # Server memory requested (per node)
#SBATCH -t 960:00:00                        # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x.out
#SBATCH -e _errors/%x.err  

python ../d01_read-files/03_read-ZIP-crosswalk.py