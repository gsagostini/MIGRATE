#!/bin/bash

#SBATCH -J smoothing_nnls_holdout_state-stayers                       # Job name
#SBATCH --mail-type=ALL                                       # Request status by email
#SBATCH --mail-user=gs665@cornell.edu                         # Email address to send results to
#SBATCH -N 1                                                  # Total number of nodes requested
#SBATCH -n 32                                                 # Total number of cores requested
#SBATCH --get-user-env                                        # Retrieve the users login environment
#SBATCH --mem=400G                                            # Server memory requested (per node)
#SBATCH -t 960:00:00                                          # Time limit (hh:mm:ss)
#SBATCH -p pierson                                            # Partition
#SBATCH -w lisbeth
#SBATCH -o _outputs/%x.out                                    # Output directory
#SBATCH -e _errors/%x.err                                     # Error directory

python ../d03_process-flows/06_spatiotemporal-smoothing.py --ADDRID_dir "CRD4/OD_pairs/ADDRID/" --holdout_county_flows --holdout_CBG_t1 --retrieve_yearly_CBG_population --use_2010_ACS --NNLS --holdout_state_nonmovers