#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=transformers_testing_apd
#SBATCH -o transformers_testing_apd.out               
#SBATCH -e transformers_testing_apd.err

echo "Running transformers testing job on APD"

data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"
save_loc="results/apd/transformers/no-freeze"

echo "Testing vit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --model_name "vit_base_patch16_224" \
       --state_dict_loc "$save_loc/models/vit_base_patch16_224_epochs_5_batch_256_optim_sgd_aug_False_split_10_20.pth" \
       --batch_size 256

echo "Testing deit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --model_name "deit_base_patch16_224" \
       --state_dict_loc "$save_loc/models/deit_base_patch16_224_epochs_5_batch_256_optim_sgd_aug_False_split_10_20.pth" \
       --batch_size 256
