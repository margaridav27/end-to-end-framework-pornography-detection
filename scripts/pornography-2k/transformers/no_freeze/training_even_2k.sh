#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=transformers_training_even_20_2k
#SBATCH -o transformers_training_even_20_2k.out               
#SBATCH -e transformers_training_even_20_2k.err

echo "Running transformers training job on Pornography-2k (even-20)"

project_title="transformers_training_even_20_2k"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
model_save_loc="results/transformers/even-20/no-freeze/pornography-2k/models"
metrics_save_loc="results/transformers/even-20/no-freeze/pornography-2k/metrics"


echo "Training vit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vit_base_patch16_224" \
       --pretrained \
       --wandb

echo "Training vit_large_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vit_large_patch16_224" \
       --pretrained \
       --wandb

echo "Training deit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "deit_base_patch16_224" \
       --pretrained \
       --wandb
