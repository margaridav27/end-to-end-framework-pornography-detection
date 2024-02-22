#!/bin/bash
#
#SBATCH --partition=gpu_min8GB       # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8GB_ext         # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=baseline_testing  # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output. If ommited, use STDOUT.

echo "Running testing job (no frozen layers, optimized) on Pornography-2k"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_resnet50_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "resnet50"

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_resnet101_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "resnet101"

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_resnet152_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "resnet152"

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_densenet121_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "densenet121"

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_densenet169_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "densenet169"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_densenet201_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "densenet201"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_alexnet_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "alexnet"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_vgg16_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "vgg16"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_vgg19_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "vgg19"

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --save_loc "baseline/pornography-2k/results" \
       --state_dict_loc "baseline/pornography-2k/models/model_mobilenetv2_freeze_False_epochs_10_batch_32_optim_sgd_optimized_True.pth" \
       --model_name "mobilenetv2"