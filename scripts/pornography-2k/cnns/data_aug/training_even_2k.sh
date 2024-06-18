#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=training_aug_even_20_2k
#SBATCH -o training_aug_even_20_2k.out               
#SBATCH -e training_aug_even_20_2k.err


project_title="training_aug_even_20_2k"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
model_save_loc="results/pornography-2k/cnns/data-aug/even-20/models"
metrics_save_loc="results/pornography-2k/cnns/data-aug/even-20/metrics"
epochs=50


echo "Running training job on Pornography-2k (even-20, aug)"

echo "Training ResNet50"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "resnet50" \
       --epochs $epochs \
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
       --data_aug \
       --wandb
