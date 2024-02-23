#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=testing_no_freeze_800
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err

echo "Running testing job (no frozen layers) on Pornography-800"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_resnet50_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "resnet50"

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_resnet101_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "resnet101"

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_resnet152_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "resnet152"

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_densenet121_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "densenet121"

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_densenet169_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "densenet169"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_densenet201_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "densenet201"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_alexnet_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "alexnet"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_vgg16_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "vgg16"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_vgg19_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "vgg19"

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_mobilenetv2_freeze_False_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "mobilenetv2"
