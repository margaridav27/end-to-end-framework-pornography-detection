#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=testing_aug_apd
#SBATCH -o testing_aug_apd.out               
#SBATCH -e testing_aug_apd.err


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"
save_loc="results/apd/cnns/data-aug"


echo "Running testing job on APD (aug)"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/resnet50_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256
        
echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/resnet101_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/resnet152_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/densenet121_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/densenet169_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/densenet201_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/alexnet_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/vgg16_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/vgg19_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc/results" \
       --state_dict_loc "$save_loc/models/mobilenet_v2_freeze_False_epochs_5_batch_256_optim_sgd_aug_True_split_10_20.pth" \
       --batch_size 256 
