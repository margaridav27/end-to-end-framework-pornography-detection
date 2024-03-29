#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=training_testing_aug_even_20_no_logos_2k
#SBATCH -o training_testing_aug_even_20_no_logos_2k.out               
#SBATCH -e training_testing_aug_even_20_no_logos_2k.err

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-no-logos" \
       --model_save_loc "results/even-20-no-logos/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/even-20-no-logos/data-aug/pornography-2k/metrics" \
       --model_name "vgg19" \
       --epochs 50 \
       --data_aug

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-no-logos" \
       --save_loc "results/even-20-no-logos/data-aug/pornography-2k/results" \
       --state_dict_loc "results/even-20-no-logos/data-aug/pornography-2k/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"
