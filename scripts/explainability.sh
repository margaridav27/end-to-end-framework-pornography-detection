#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=explainability
#SBATCH -o explainability.out               
#SBATCH -e explainability.err


# python -m src.model_explainability \
#        --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
#        --save_loc "results/even-20/no-freeze/pornography-2k/explanations" \
#        --batch_size 4 \
#        --method "OCC" \
#        --attribute_kwargs '{ "sliding_window_shapes": (3,8,8), "strides": (3,4,4) }'

# python -m src.model_explainability \
#        --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
#        --save_loc "results/even-20/no-freeze/pornography-2k/explanations" \
#        --batch_size 4 \
#        --method "IG"

python -m src.model_explainability \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/explanations" \
       --batch_size 4 \
       --method "IG" \
       --noise_tunnel
