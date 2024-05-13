#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=training_testing_aug_even_20_no_logos_2k
#SBATCH -o training_testing_aug_even_20_no_logos_2k.out               
#SBATCH -e training_testing_aug_even_20_no_logos_2k.err


project_title="training_aug_even_20_no_logos_2k"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-no-logos"
results_loc="results/pornography-2k/cnns/data-aug/even-20-no-logos"
epochs=50


echo "Training VGG19"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "vgg19" \
       --epochs $epochs \
       --data_aug \
       --wandb

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"
