#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=training_aug_even_20_800
#SBATCH -o training_aug_even_20_800.out               
#SBATCH -e training_aug_even_20_800.err

echo "Running training job on Pornography-800 (even-20, aug)"

echo "Training ResNet50"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "resnet50" \
       --epochs 50 \
       --data_aug
        
echo "Training ResNet101"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "resnet101" \
       --epochs 50 \
       --data_aug

echo "Training ResNet152"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "resnet152" \
       --epochs 50 \
       --data_aug

echo "Training DenseNet121"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "densenet121" \
       --epochs 50 \
       --data_aug

echo "Training DenseNet169"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "densenet169" \
       --epochs 50 \
       --data_aug

echo "Training DenseNet201"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "densenet201" \
       --epochs 50 \
       --data_aug

echo "Training AlexNet"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "alexnet" \
       --epochs 50 \
       --data_aug

echo "Training VGG16"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "vgg16" \
       --epochs 50 \
       --data_aug

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "vgg19" \
       --epochs 50 \
       --data_aug

echo "Training MobileNetV2"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "mobilenet_v2" \
       --epochs 50 \
       --data_aug
