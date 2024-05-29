#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=transformers_training_aug_even_20_800
#SBATCH -o transformers_training_aug_even_20_800.out               
#SBATCH -e transformers_training_aug_even_20_800.err


project_title="transformers_training_aug_even_20_800"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20"
results_loc="results/pornography-800/transformers/data-aug/even-20"
learning_rate=1e-4
epochs=50
batch_size=8


echo "Running transformers training job on Pornography-800 (even-20, aug)"

echo "Training vit_base_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vit_base_patch16_224" \
       --pretrained \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --data_aug \
       --wandb

echo "Training vit_large_patch16_224"
python -m src.transformer_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vit_large_patch16_224" \
       --pretrained \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
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
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --data_aug \
       --wandb
