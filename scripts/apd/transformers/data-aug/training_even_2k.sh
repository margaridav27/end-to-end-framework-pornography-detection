#!/bin/bash
#
#SBATCH --partition=gpu_min80GB               
#SBATCH --qos=gpu_min80GB_ext                 
#SBATCH --job-name=transformers_training_aug_apd
#SBATCH -o transformers_training_aug_apd.out               
#SBATCH -e transformers_training_aug_apd.err

echo "Running transformers training job on APD (aug)"

project_title="transformers_training_aug_apd"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"
save_loc="results/apd/transformers/data-aug"
epochs=5
batch_size=256

echo "Training vit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$save_loc/models" \
       --metrics_save_loc "$save_loc/metrics" \
       --model_name "vit_base_patch16_224" \
       --pretrained \
       --epochs $epochs \
       --batch_size $batch_size \
       --data_aug \
       --wandb

echo "Training deit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$save_loc/models" \
       --metrics_save_loc "$save_loc/metrics" \
       --model_name "deit_base_patch16_224" \
       --pretrained \
       --epochs $epochs \
       --batch_size $batch_size \
       --data_aug \
       --wandb
