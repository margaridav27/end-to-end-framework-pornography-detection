#!/bin/bash
#
#SBATCH --partition=gpu_min12GB               
#SBATCH --qos=gpu_min12GB_ext                 
#SBATCH --job-name=training_no_freeze_opt_2k
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err 

echo "Running training job (no frozen layers, optimized) on Pornography-2k"

echo "Training ResNet50"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet50" \
       --optimized
        
echo "Training ResNet101"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet101" \
       --optimized

echo "Training ResNet152"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet152" \
       --optimized

echo "Training DenseNet121"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "densenet121" \
       --optimized

echo "Training DenseNet169"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "densenet169" \
       --optimized

echo "Training DenseNet201"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "densenet201" \
       --optimized

echo "Training AlexNet"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "alexnet" \
       --optimized

echo "Training VGG16"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "vgg16" \
       --optimized

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "vgg19" \
       --optimized

echo "Training MobileNetV2"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "mobilenetv2" \
       --optimized
