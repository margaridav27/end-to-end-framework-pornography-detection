#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=training_no_freeze_2k
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err

echo "Running training job (with data augmentation) on Pornography-2k"

echo "Training ResNet50"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "resnet50"
        
echo "Training ResNet101"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "resnet101"

echo "Training ResNet152"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "resnet152"

echo "Training DenseNet121"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "densenet121"

echo "Training DenseNet169"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "densenet169"

echo "Training DenseNet201"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "densenet201"

echo "Training AlexNet"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "alexnet"

echo "Training VGG16"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "vgg16"

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "vgg19"

echo "Training MobileNetV2"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --model_save_loc "results/data-aug/pornography-2k/models" \
       --metrics_save_loc "results/data-aug/pornography-2k/metrics" \
       --epochs 10 \
       --data_aug \
       --model_name "mobilenetv2"
