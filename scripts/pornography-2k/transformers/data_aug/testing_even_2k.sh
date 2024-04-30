#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=transformers_testing_aug_even_20_2k
#SBATCH -o transformers_testing_aug_even_20_2k.out               
#SBATCH -e transformers_testing_aug_even_20_2k.err

echo "Running transformers testing job on Pornography-2k (even-20, aug)"

data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
save_loc="results/transformers/even-20/data-aug/pornography-2k/results"
state_dict_loc="results/transformers/even-20/data-aug/pornography-2k/models"


echo "Testing vit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --model_name "vit_base_patch16_224" \
       --state_dict_loc "$state_dict_loc/vit_base_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing vit_large_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --model_name "vit_large_patch16_224" \
       --state_dict_loc "$state_dict_loc/vit_large_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 8
       
echo "Testing deit_base_patch16_224"
python -m src.transformer_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --model_name "deit_base_patch16_224" \
       --state_dict_loc "$state_dict_loc/deit_base_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"
