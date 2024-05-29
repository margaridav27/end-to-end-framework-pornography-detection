#!/bin/bash
#
#SBATCH --partition=gpu_min24gb               
#SBATCH --qos=gpu_min24gb_ext                 
#SBATCH --job-name=training_800
#SBATCH -o training_800.out               
#SBATCH -e training_800.err 


project_title="training_800"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20"
results_loc="results/pornography-800/cnns/no-freeze/middle-20"
learning_rate=1e-4
epochs=100
batch_size=16


echo "Running training job on Pornography-800 (middle-20)"

echo "Training ResNet50"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "resnet50" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training ResNet101"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "resnet101" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training ResNet152"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "resnet152" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training DenseNet121"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "densenet121" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training DenseNet169"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "densenet169" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training DenseNet201"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "densenet201" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training AlexNet"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "alexnet" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training VGG16"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "vgg16" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training VGG19"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "vgg19" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb

echo "Training MobileNetV2"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "mobilenet_v2" \
       --learning_rate $learning_rate \
       --epochs $epochs \
       --batch_size $batch_size \
       --wandb
