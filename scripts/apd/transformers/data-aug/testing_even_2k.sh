#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=transformers_testing_aug_apd
#SBATCH -o transformers_testing_aug_apd.out               
#SBATCH -e transformers_testing_aug_apd.err

echo "Running transformers testing job on APD (aug)"

data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"
save_loc="results/apd/transformers/data-aug"

echo "Testing vit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --model_name "vit_base_patch16_224" \
       --state_dict_loc "$save_loc/models/vit_base_patch16_224_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 128

echo "Testing deit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --model_name "deit_base_patch16_224" \
       --state_dict_loc "$save_loc/models/deit_base_patch16_224_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 128
