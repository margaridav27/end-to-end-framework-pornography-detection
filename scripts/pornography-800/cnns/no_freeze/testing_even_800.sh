#!/bin/bash
#
#SBATCH --partition=gpu_min8gb               
#SBATCH --qos=gpu_min8gb_ext                 
#SBATCH --job-name=testing_even_20_800
#SBATCH -o testing_even_20_800.out               
#SBATCH -e testing_even_20_800.err


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20"
results_loc="results/pornography-800/cnns/no-freeze/even-20"


echo "Running testing job on Pornography-800 (even-20)"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/resnet50_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/resnet101_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/resnet152_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/densenet121_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/densenet169_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/densenet201_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/alexnet_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/vgg16_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/vgg19_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$results_loc/results" \
       --state_dict_loc "$results_loc/models/mobilenet_v2_freeze_False_epochs_100_batch_16_optim_sgd_aug_False_split_10_20.pth"
