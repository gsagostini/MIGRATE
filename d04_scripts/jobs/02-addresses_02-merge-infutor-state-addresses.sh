#!/bin/bash

#SBATCH -J 02-addresses_02-merge-infutor-state-addresses         # Job name
#SBATCH -N 1                                                     # Total number of nodes requested
#SBATCH -n 32                                                    # Total number of cores requested
#SBATCH --get-user-env                                           # Retrieve the users login environment
#SBATCH --mem=1000G                                              # Server memory requested (per node)
#SBATCH -t 960:00:00                                             # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x.out
#SBATCH -e _errors/%x.err

python ../d02_process-addresses/02_merge-infutor-state-addresses.py