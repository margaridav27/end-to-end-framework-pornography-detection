#!/bin/bash
#
#SBATCH --partition=gpu_min11GB          # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min11GB_ext            # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=baseline_training_800 # Job name
#SBATCH -o slurm.%N.%j.out               # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err               # File containing STDERR output. If ommited, use STDOUT.

echo "Running baseline training job on Pornography-800"

echo "Training ResNet50"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet50" \
       --freeze_layers

echo "Training ResNet101"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet101" \
       --freeze_layers

echo "Training ResNet152"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet152" \
       --freeze_layers

echo "Training DenseNet121"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet121" \
       --freeze_layers

echo "Training DenseNet169"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet169" \
       --freeze_layers

echo "Training DenseNet201"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet201" \
       --freeze_layers

echo "Training AlexNet"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "alexnet" \
       --freeze_layers

echo "Training VGG16"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "vgg16" \
       --freeze_layers

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "vgg19" \
       --freeze_layers

echo "Training MobileNetV2"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "mobilenetv2" \
       --freeze_layers
