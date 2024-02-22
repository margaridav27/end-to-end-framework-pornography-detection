#!/bin/bash
#
#SBATCH --partition=gpu_min11GB               
#SBATCH --qos=gpu_min11GB_ext                 
#SBATCH --job-name=training_no_freeze_2k
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err

echo "Running training job (no frozen layers) on Pornography-2k"

echo "Training ResNet50"
python -m src.model_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet50"
