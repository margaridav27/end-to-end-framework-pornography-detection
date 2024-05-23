#!/bin/bash
#
#SBATCH --partition=gpu_min11gb               
#SBATCH --qos=gpu_min11gb_ext                 
#SBATCH --job-name=cross_dataset_testing
#SBATCH -o cross_dataset_testing.out               
#SBATCH -e cross_dataset_testing.err


apd_data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"


echo "Testing VGG19 trained on Pornography-2k (even-20, aug) on APD"

python -m src.model_testing \
       --data_loc "$apd_data_loc" \
       --save_loc "results/cross-dataset-testing/train-2k-test-apd" \
       --state_dict_loc "results/pornography-2k/cnns/data-aug/even-20/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 128

echo "Testing vit_base_patch16_224 trained on Pornography-2k (even-20, aug) on APD"

python -m src.transformer_testing \
       --data_loc "$apd_data_loc" \
       --save_loc "results/cross-dataset-testing/train-2k-test-apd" \
       --model_name "vit_base_patch16_224" \
       --state_dict_loc "results/pornography-2k/transformers/data-aug/even-20/models/vit_base_patch16_224_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" \
       --batch_size 128

echo "Testing vit_large_patch16_224 trained on Pornography-2k (even-20, aug) on APD"

python -m src.transformer_testing \
       --data_loc "$apd_data_loc" \
       --save_loc "results/cross-dataset-testing/train-2k-test-apd" \
       --model_name "vit_large_patch16_224" \
       --state_dict_loc "results/pornography-2k/transformers/data-aug/even-20/models/vit_large_patch16_224_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" \
       --batch_size 128