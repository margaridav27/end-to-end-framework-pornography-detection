#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=testing_aug_800
#SBATCH -o testing_aug_800.out               
#SBATCH -e testing_aug_800.err

echo "Running testing job on Pornography-800 (middle-20, aug)"


echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/resnet101_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/resnet152_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/densenet121_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/densenet169_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/densenet201_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/alexnet_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/vgg16_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/data-aug/pornography-800/results" \
       --state_dict_loc "results/middle-20/data-aug/pornography-800/models/mobilenet_v2_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth"
