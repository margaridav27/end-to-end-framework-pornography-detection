#!/bin/bash
#
#SBATCH --partition=gpu_min11GB               
#SBATCH --qos=gpu_min11GB_ext                 
#SBATCH --job-name=transformer_explainability_aug
#SBATCH -o transformer_explainability_aug.out               
#SBATCH -e transformer_explainability_aug.err

state_dict_loc="results/transformers/even-20/data-aug/pornography-2k/models"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
save_loc="results/transformers/even-20/data-aug/pornography-2k/explanations" 


# python -m src.transformer_explainability \
#        --model_name "vit_base_patch16_224" \
#        --state_dict_loc "$state_dict_loc/vit_base_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
#        --data_loc "$data_loc" \
#        --save_loc "$save_loc"

python -m src.transformer_explainability \
       --model_name "vit_large_patch16_224" \
       --state_dict_loc "$state_dict_loc/vit_large_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --data_loc "$data_loc" \
       --save_loc "$save_loc"

# python -m src.transformer_explainability \
#        --model_name "deit_base_patch16_224" \
#        --state_dict_loc "$state_dict_loc/deit_base_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
#        --data_loc "$data_loc" \
#        --save_loc "$save_loc"  
