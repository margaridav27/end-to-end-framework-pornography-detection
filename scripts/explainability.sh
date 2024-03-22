#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB _ext                 
#SBATCH --job-name=explainability
#SBATCH -o explainability.out               
#SBATCH -e explainability.err

python -m src.model_explainability \
       --state_dict_loc "results/even-20/data-aug/pornography-2k/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/data-aug/pornography-2k/explanations" \
       --method "IG"
