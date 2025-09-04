#!/bin/bash

#SBATCH -J 05-synthetic
#SBATCH --mail-type=ALL                     # Request status by email
#SBATCH --mail-user=gs665@cornell.edu       # Email address to send results to
#SBATCH -p pierson
#SBATCH -w lisbeth
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20G
#SBATCH -t 2400:00:00
#SBATCH --array=1-66                    #Use 1-231:11 if bias = 0, 1-66 for iid noise
#SBATCH -o _outputs/synth/%x_%a.out
#SBATCH -e _errors/synth/%x_%a.err
#SBATCH --export=NONE

#Fix the path variables (to avoid CPU binding error)
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH
unset OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY SLURM_CPU_BIND
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

#Activate environment:
eval "$(/share/apps/anaconda3/2022.10/bin/conda shell.bash hook)"
conda activate /share/pierson/conda_virtualenvs/INFUTOR_env

#Parameters passed via --export upon submission:
YEAR=${YEAR:-2011}; EXPERIMENT=${EXPERIMENT:-1}
idx=${SLURM_ARRAY_TASK_ID}
line=$(sed -n "$((idx+1))p" ../d04_optimization/synthetic_hyperparameters.csv)
IFS=',' read -r noise_scale bias_scale <<< "$line"

#Run:
python ../d04_optimization/02_Synthetic.py --ignore_PR --experiment "$EXPERIMENT" --idx "$idx" --year "$YEAR" --noise_scale "$noise_scale" --bias_scale "$bias_scale" --n_iterations 3_000 --noise_type "LogNormal" --noise_random
