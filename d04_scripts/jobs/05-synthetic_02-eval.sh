#!/bin/bash

#SBATCH -J 05-synthetic-evaluation-E2
#SBATCH --mail-type=ALL                     # Request status by email
#SBATCH --mail-user=gs665@cornell.edu       # Email address to send results to
#SBATCH -p pierson
#SBATCH -w lisbeth
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 50
#SBATCH --mem=300G
#SBATCH -t 2400:00:00
#SBATCH -o _outputs/%x.out
#SBATCH -e _errors/%x.err
#SBATCH --export=NONE

#Fix the path variables (to avoid CPU binding error)
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH
unset OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY SLURM_CPU_BIND
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

#Activate environment:
eval "$(/share/apps/anaconda3/2022.10/bin/conda shell.bash hook)"
conda activate /share/pierson/conda_virtualenvs/INFUTOR_env

#Run:
python ../d04_optimization/03_Evaluate-Synthetic.py --ignore_PR --experiments 2 --n_workers 50 --years 2012 2013 2014 2015 2016 2019