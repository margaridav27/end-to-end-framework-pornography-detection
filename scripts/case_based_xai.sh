#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=case_based_xai
#SBATCH -o case_based_xai.out               
#SBATCH -e case_based_xai.err

data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
save_loc="results/pornography-2k/cnns/data-aug/even-20/explanations"
state_dict_loc="results/even-20/data-aug/pornography-2k/models"

python -m src.case_based_xai \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"
