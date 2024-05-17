#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=model_explainability
#SBATCH -o model_explainability.out               
#SBATCH -e model_explainability.err


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
results_loc="results/pornography-2k/cnns/data-aug/even-20"

python -m src.model_explainability \
       --data_loc "$data_loc" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --save_loc "$results_loc/explanations" \
       --method "IG" \
       --side_by_side

python -m src.model_explainability \
       --data_loc "$data_loc" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --save_loc "$results_loc/explanations" \
       --method "DEEP-LIFT" \
       --side_by_side

python -m src.model_explainability \
       --data_loc "$data_loc" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --save_loc "$results_loc/explanations" \
       --method "LRP" \
       --side_by_side

python -m src.model_explainability \
       --data_loc "$data_loc" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --save_loc "$results_loc/explanations" \
       --method "LRP-CMP" \
       --side_by_side

python -m src.model_explainability \
       --data_loc "$data_loc" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --save_loc "$results_loc/explanations" \
       --method "OCC" \
       --attribute_kwargs '{ "sliding_window_shapes": (3,8,8), "strides": (3,4,4) }' \
       --side_by_side
