#!/bin/bash
#
#SBATCH --partition=gpu_min12gb         
#SBATCH --qos=gpu_min12gb_ext            
#SBATCH --job-name=training_baseline_2k
#SBATCH -o training_baseline_2k.out               
#SBATCH -e training_baseline_2k.err 


project_title="training_baseline_2k"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20"
results_loc="results/pornography-2k/cnns/baseline/middle-20"
epochs=50


echo "Running baseline training job on Pornography-2k"

echo "Training ResNet50"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "resnet50" \
       --freeze_layers \
       --wandb
        
echo "Training ResNet101"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "resnet101" \
       --freeze_layers \
       --wandb

echo "Training ResNet152"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "resnet152" \
       --freeze_layers \
       --wandb

echo "Training DenseNet121"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "densenet121" \
       --freeze_layers \
       --wandb

echo "Training DenseNet169"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "densenet169" \
       --freeze_layers \
       --wandb

echo "Training DenseNet201"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "densenet201" \
       --freeze_layers \
       --wandb

echo "Training AlexNet"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "alexnet" \
       --freeze_layers \
       --wandb

echo "Training VGG16"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "vgg16" \
       --freeze_layers \
       --wandb

echo "Training VGG19"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "vgg19" \
       --freeze_layers \
       --wandb

echo "Training MobileNetV2"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$results_loc/models" \
       --metrics_save_loc "$results_loc/metrics" \
       --model_name "mobilenet_v2" \
       --freeze_layers \
       --wandb
