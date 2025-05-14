#!/bin/bash

#SBATCH -J 03-individuals-SPLIT                                  # Job name
#SBATCH --mail-type=ALL                                          # Request status by email
#SBATCH --mail-user=gs665@cornell.edu                            # Email address to send results to
#SBATCH -N 1                                                     # Total number of nodes requested
#SBATCH -n 8                                                     # Total number of cores requested
#SBATCH --get-user-env                                           # Retrieve the users login environment
#SBATCH --mem=500G                                               # Server memory requested (per node)
#SBATCH -t 960:00:00                                             # Time limit (hh:mm:ss)
#SBATCH -p pierson                                               # Partition
#SBATCH -w lisbeth
#SBATCH -o _outputs/%x.out
#SBATCH -e _errors/%x.err

python ../d03_process-flows/02_split-INFUTOR-individuals.py