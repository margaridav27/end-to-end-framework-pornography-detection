#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=transformers_testing_aug_even_20_2k
#SBATCH -o transformers_testing_aug_even_20_2k.out               
#SBATCH -e transformers_testing_aug_even_20_2k.err


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
results_loc="results/pornography-2k/transformers/data-aug/even-20"


echo "Running transformers testing job on Pornography-2k (even-20, aug)"

echo "Testing vit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --model_name "vit_base_patch16_224" \
       --state_dict_loc "$results_loc/models/vit_base_patch16_224_epochs_50_batch_8_optim_sgd_aug_True_split_10_20.pth"

echo "Testing vit_large_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --model_name "vit_large_patch16_224" \
       --state_dict_loc "$results_loc/models/vit_large_patch16_224_epochs_50_batch_8_optim_sgd_aug_True_split_10_20.pth" \
       
echo "Testing deit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --model_name "deit_base_patch16_224" \
       --state_dict_loc "$results_loc/models/deit_base_patch16_224_epochs_50_batch_8_optim_sgd_aug_True_split_10_20.pth"
