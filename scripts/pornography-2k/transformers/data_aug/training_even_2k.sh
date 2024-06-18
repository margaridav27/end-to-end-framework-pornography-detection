#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=transformers_training_aug_even_20_2k
#SBATCH -o transformers_training_aug_even_20_2k.out               
#SBATCH -e transformers_training_aug_even_20_2k.err

echo "Running transformers training job on Pornography-2k (even-20, aug)"

project_title="transformers_training_aug_even_20_2k"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
model_save_loc="results/pornography-2k/transformers/data-aug/even-20/models"
metrics_save_loc="results/pornography-2k/transformers/data-aug/even-20/metrics"


echo "Training vit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vit_base_patch16_224" \
       --pretrained \
       --data_aug \
       --wandb

echo "Training deit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "deit_base_patch16_224" \
       --pretrained \
       --data_aug \
       --wandb
