#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=model_explainability
#SBATCH -o model_explainability.out               
#SBATCH -e model_explainability.err


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
results_loc="results/pornography-2k/cnns/data-aug/even-20"

echo "Captum: Integrated Gradients"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "captum" \
       --method_cfg '{"method_name": "IG"}' \
       --side_by_side

echo "Captum: Deep LIFT"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "captum" \
       --method_cfg '{"method_name": "DEEP-LIFT"}' \
       --side_by_side

echo "Captum: LRP"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "captum" \
       --method_cfg '{"method_name": "LRP"}' \
       --side_by_side

echo "Captum: LRP (composite rule)"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "captum" \
       --method_cfg '{"method_name": "LRP-CMP"}' \
       --side_by_side

echo "Captum: Occlusion"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "captum" \
       --method_cfg '{"method_name": "OCC", "attribute_kwargs": {"sliding_window_shapes": (3,8,8), "strides": (3,4,4)}}' \
       --side_by_side


echo "Zennit: Integrated Gradients"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "zennit" \
       --method_cfg '{"method_name": "IntegratedGradients", "method_kwargs": {"n_iter": 50}}' \
       --side_by_side

echo "Zennit: LRP"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "zennit" \
       --method_cfg '{"method_name": "Gradient", "composite_name": "EpsilonGammaBox", "composite_kwargs": {"low": -2.12, "high": 2.64}}' \
       --side_by_side

echo "Zennit: LRP (composite rule)"
python -m src.model_explainability \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/explanations" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --library "zennit" \
       --method_cfg '{"method_name": "Gradient", "composite_name": "EpsilonPlusFlat"}' \
       --side_by_side

# echo "Zennit: Occlusion"
# method_cfg='{
#     "method_name": "Occlusion", 
#     "method_kwargs": {
#         "window": 8,
#         "stride": 4
#     }, 
# }'
# python -m src.model_explainability \
#        --data_loc "$data_loc" \
#        --save_loc "$results_loc/explanations" \
#        --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
#        --library "zennit" \
#        --method_cfg $method_cfg \
#        --side_by_side
