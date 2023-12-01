#!/bin/bash
#SBATCH --job-name=dipole
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem-per-cpu=10GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Change to this job's submit directory
cd $SLURM_SUBMIT_DIR

python simpletrain.py 
