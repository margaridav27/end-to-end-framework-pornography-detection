#!/bin/bash
#
#SBATCH --partition=gpu_min12GB         
#SBATCH --qos=gpu_min12GB_ext            
#SBATCH --job-name=baseline_training_2k
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err 

echo "Running baseline training job on Pornography-2k"

echo "Training ResNet50"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet50" \
       --freeze_layers
        
echo "Training ResNet101"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet101" \
       --freeze_layers

echo "Training ResNet152"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "resnet152" \
       --freeze_layers

echo "Training DenseNet121"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "densenet121" \
       --freeze_layers

echo "Training DenseNet169"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "densenet169" \
       --freeze_layers

echo "Training DenseNet201"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "densenet201" \
       --freeze_layers

echo "Training AlexNet"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "alexnet" \
       --freeze_layers

echo "Training VGG16"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "vgg16" \
       --freeze_layers

echo "Training VGG19"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "vgg19" \
       --freeze_layers

echo "Training MobileNetV2"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-40" \
       --model_save_loc "baseline/pornography-2k/models" \
       --metrics_save_loc "baseline/pornography-2k/metrics" \
       --model_name "mobilenetv2" \
       --freeze_layers
