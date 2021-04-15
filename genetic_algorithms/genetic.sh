#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Reset environment for this job.
# SBATCH --export=NONE
# Define job name
#SBATCH -J Genetic
# Alocate memeory per core
#SBATCH --mem-per-cpu=32000M
# Setting maximum time days-hh:mm:ss]
#SBATCH -t 72:00:00
# Setting number of CPU cores and number of nodes
#SBATCH -n 4 -N 1

# Load modules
module load libs/nvidia-cuda/10.1.168/bin

# Change conda env
conda activate dds


python genetic_algorithm.py