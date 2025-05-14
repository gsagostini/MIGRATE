#!/bin/bash

#SBATCH -J 02-addresses_08-produce-geocoding-matrix              # Job name
#SBATCH -N 1                                                     # Total number of nodes requested
#SBATCH -n 32                                                    # Total number of cores requested
#SBATCH --get-user-env                                           # Retrieve the users login environment
#SBATCH --mem=1000G                                              # Server memory requested (per node)
#SBATCH -t 960:00:00                                             # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x.out                                       # Output directory
#SBATCH -e _errors/%x.err                                        # Error directory

python ../d02_process-addresses/08_produce-geocoding-matrix.py --geography "blockgroup" --exact_coordinates_saved