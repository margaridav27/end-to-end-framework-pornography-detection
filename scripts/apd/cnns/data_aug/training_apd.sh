#!/bin/bash
#
#SBATCH --partition=gpu_min80gb               
#SBATCH --qos=gpu_min80gb_ext                 
#SBATCH --job-name=training_aug_apd
#SBATCH -o training_aug_apd.out               
#SBATCH -e training_aug_apd.err


project_title="training_aug_apd"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"
model_save_loc="results/apd/cnns/data-aug/models"
metrics_save_loc="results/apd/cnns/data-aug/metrics"
epochs=5


echo "Running training job on APD (aug)"

echo "Training ResNet50"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "resnet50" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb
        
echo "Training ResNet101"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "resnet101" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training ResNet152"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "resnet152" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training DenseNet121"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "densenet121" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training DenseNet169"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "densenet169" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training DenseNet201"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "densenet201" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training AlexNet"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "alexnet" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training VGG16"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vgg16" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training VGG19"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vgg19" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb

echo "Training MobileNetV2"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "mobilenet_v2" \
       --epochs $epochs \
       --batch_size 256 \
       --data_aug \
       --wandb
