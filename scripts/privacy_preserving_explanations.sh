#!/bin/bash
#
#SBATCH --partition=gpu_min12gb      
#SBATCH --qos=gpu_min12gb_ext        
#SBATCH --job-name=privacy_preserving_explanations      
#SBATCH -o privacy_preserving_explanations.out          
#SBATCH -e privacy_preserving_explanations.err          


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
explanations_loc="results/pornography-2k/cnns/data-aug/even-20/explanations/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20" 


echo "Running privacy-preserving explanations job on vgg19 (even-20, aug)"

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/captum/correct/IG/npys" \
       --save_loc "$explanations_loc/captum/correct/IG" 

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/captum/correct/DEEP-LIFT/npys" \
       --save_loc "$explanations_loc/captum/correct/DEEP-LIFT" 

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/captum/correct/LRP/npys" \
       --save_loc "$explanations_loc/captum/correct/LRP"

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/captum/correct/LRP-CMP/npys" \
       --save_loc "$explanations_loc/captum/correct/LRP-CMP" 

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/captum/correct/OCC/npys" \
       --save_loc "$explanations_loc/captum/correct/OCC"

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/zennit/correct/IntegratedGradients/npys" \
       --save_loc "$explanations_loc/zennit/correct/IntegratedGradients"

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/zennit/correct/Gradient_EpsilonGammaBox/npys" \
       --save_loc "$explanations_loc/zennit/correct/Gradient_EpsilonGammaBox"

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/zennit/correct/Gradient_EpsilonPlusFlat/npys" \
       --save_loc "$explanations_loc/zennit/correct/Gradient_EpsilonPlusFlat"

explanations_loc="results/pornography-2k/transformers/data-aug/even-20/explanations/vit_base_patch16_224_epochs_50_batch_16_optim_sgd_aug_True_split_10_20/correct" 

echo "Running privacy-preserving explanations job on vit_base_patch16_224 (even-20, aug)"

python -m src.privacy_preserving_explanations \
       --data_loc "$data_loc" \
       --faces_loc "$data_loc-detected-faces" \
       --explanations_loc "$explanations_loc/npys" \
       --save_loc "$explanations_loc"
