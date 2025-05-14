#!/bin/bash

#SBATCH -J 02-addresses_04_merge-geocoding-ArcGIS                # Job name
#SBATCH -N 1                                                     # Total number of nodes requested
#SBATCH -n 16                                                    # Total number of cores requested
#SBATCH --get-user-env                                           # Retrieve the users login environment
#SBATCH --mem=500G                                               # Server memory requested (per node)
#SBATCH -t 960:00:00                                             # Time limit (hh:mm:ss)
#SBATCH -o _outputs/%x.out
#SBATCH -e _errors/%x.err

python ../d02_process-addresses/04_merge-geocoding-ArcGIS.py