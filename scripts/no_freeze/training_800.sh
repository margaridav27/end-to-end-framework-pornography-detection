#!/bin/bash
#
#SBATCH --partition=gpu_min11GB               
#SBATCH --qos=gpu_min11GB_ext                 
#SBATCH --job-name=training_no_freeze_800
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err 

echo "Running training job (no frozen layers) on Pornography-800"

echo "Training ResNet50"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet50"

echo "Training ResNet101"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet101"

echo "Training ResNet152"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "resnet152"

echo "Training DenseNet121"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet121"

echo "Training DenseNet169"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet169"

echo "Training DenseNet201"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "densenet201"

echo "Training AlexNet"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "alexnet"

echo "Training VGG16"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "vgg16"

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "vgg19"

echo "Training MobileNetV2"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-800/models" \
       --metrics_save_loc "baseline/pornography-800/metrics" \
       --model_name "mobilenetv2"
