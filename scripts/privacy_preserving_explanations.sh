#!/bin/bash
#
#SBATCH --partition=gpu_min8gb      
#SBATCH --qos=gpu_min8gb_ext        
#SBATCH --job-name=privacy_preserving_explanations      
#SBATCH -o privacy_preserving_explanations.out          
#SBATCH -e privacy_preserving_explanations.err          


python -m src.privacy_preserving_explanations \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --faces_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-detected-faces" \
       --explanations_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/LRP/npys" \
       --save_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/IG/faces" \
       --side_by_side

python -m src.privacy_preserving_explanations \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --faces_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-detected-faces" \
       --explanations_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/LRP/npys" \
       --save_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/LRP/faces" \
       --side_by_side

python -m src.privacy_preserving_explanations \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --faces_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-detected-faces" \
       --explanations_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/LRP-CMP/npys" \
       --save_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/LRP-CMP/faces" \
       --side_by_side

python -m src.privacy_preserving_explanations \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --faces_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-detected-faces" \
       --explanations_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/OCC/npys" \
       --save_loc "results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct/OCC/faces" \
       --side_by_side
