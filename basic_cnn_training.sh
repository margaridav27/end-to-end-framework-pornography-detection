#!/bin/bash
#
#SBATCH --partition=gpu_min12GB         
#SBATCH --qos=gpu_min12GB_ext            
#SBATCH --job-name=baseline_training_2k
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err 

echo "Running training job on Basic CNN"

python -m src.basic_cnn_training