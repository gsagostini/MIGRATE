#!/bin/bash

#SBATCH -J 04-model-NNLS_holdout_state-stayers           # Job name
#SBATCH --mail-type=ALL                        # Request status by email
#SBATCH --mail-user=gs665@cornell.edu          # Email address to send results to
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH -n 8                                   # Total number of cores requested
#SBATCH --get-user-env                         # Retrieve the users login environment
#SBATCH --mem=16G                              # Server memory requested (per node)
#SBATCH -t 2400:00:00                          # Time limit (hh:mm:ss)
#SBATCH -p pierson
#SBATCH -w lisbeth
#SBATCH -o _outputs/%x_%a.out                  # Output directory
#SBATCH -e _errors/%x_%a.err                   # Error directory
#SBATCH --array=1-9

year=$((2010 + SLURM_ARRAY_TASK_ID))
python ../d04_optimization/01_IPF.py --year $year --n_iterations 3000 --E_subdir 'ADDRID' --ignore_PR --output_subdir 'd06_final' --harmonize_constraints --rescale_flows --use_S --finish_on_row --save_every_k 3000 --holdout_county_flows --retrieve_yearly_CBG_population --holdout_CBG_t1 --use_2010_ACS --NNLS --holdout_state_nonmovers