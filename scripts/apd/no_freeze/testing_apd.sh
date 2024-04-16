#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=testing_apd
#SBATCH -o testing_apd.out               
#SBATCH -e testing_apd.err

echo "Running testing job on APD"

data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"
save_loc="results/apd/no-freeze/results"
state_dict_loc="results/apd/no-freeze/models"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"
        
echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/resnet101_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/resnet152_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/densenet121_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/densenet169_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/densenet201_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/alexnet_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg16_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing VGG16_BN"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg16_bn_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing VGG19_BN"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg19_bn_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/mobilenet_v2_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth" 
