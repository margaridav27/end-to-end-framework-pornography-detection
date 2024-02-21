#!/bin/bash
#
#SBATCH --partition=gpu_min11GB          # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min11GB_ext            # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=baseline_training_800 # Job name
#SBATCH -o slurm.%N.%j.out               # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err               # File containing STDERR output. If ommited, use STDOUT.

echo "Running baseline training job on Pornography-800\n"

echo "Training ResNet50\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet50"

echo "Training ResNet101\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet101"

echo "Training ResNet152\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet152"

echo "Training DenseNet121\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet121"

echo "Training DenseNet169\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet169"

echo "Training DenseNet201\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet201"

echo "Training AlexNet\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "alexnet"

echo "Training VGG16\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "vgg16"

echo "Training VGG19\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "vgg19"

echo "Training MobileNetV2\n"
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "mobilenetv2"

