#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=explainability_2
#SBATCH -o explainability_2.out               
#SBATCH -e explainability_2.err


python -m src.model_explainability \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/explanations" \
       --batch_size 4 \
       --method "OCC" \
       --filter "all" \
       --attribute_kwargs '{ "sliding_window_shapes": (3,8,8), "strides": (3,4,4) }' \
       --to_explain "vPorn000064#1.jpg" "vPorn000064#6.jpg" "vPorn000064#7.jpg" \
                    "vPorn000085#2.jpg" "vPorn000085#5.jpg" "vPorn000085#7.jpg" \
                    "vPorn000101#0.jpg" \
                    "vPorn000601#17.jpg"
