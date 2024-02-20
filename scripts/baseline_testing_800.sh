#!/bin/bash
#
#SBATCH --partition=gpu_min8GB      # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8GB_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=baseline_testing # Job name
#SBATCH -o slurm.%N.%j.out          # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err          # File containing STDERR output. If ommited, use STDOUT.

echo "Running baseline testing job on Pornography-800\n"

echo "Testing ResNet50\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_resnet50_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "resnet50"

echo "Testing ResNet101\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_resnet101_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "resnet101"

echo "Testing ResNet152\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_resnet152_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "resnet152"

echo "Testing DenseNet121\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_densenet121_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "densenet121"

echo "Testing DenseNet169\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_densenet169_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "densenet169"

echo "Testing DenseNet201\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_densenet201_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "densenet201"

echo "Testing AlexNet\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_alexnet_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "alexnet"

echo "Testing VGG16\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_vgg16_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "vgg16"

echo "Testing VGG19\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_vgg19_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "vgg19"

echo "Testing MobileNetV2\n"
python src/baseline/baseline_testing.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --save_loc "baseline/pornography-800/results" \
       --state_dict_loc "baseline/pornography-800/models/model_mobilenetv2_freeze_True_epochs_10_batch_32_optim_sgd_optimized_False.pth" \
       --model_name "mobilenetv2"
