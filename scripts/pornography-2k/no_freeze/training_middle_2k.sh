#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=training_2k
#SBATCH -o training_2k.out               
#SBATCH -e training_2k.err

echo "Running training job on Pornography-2k (middle-20)"

project_title="training_2k"
data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20"
model_save_loc="results/middle-20/no-freeze/pornography-2k/models"
metrics_save_loc="results/middle-20/no-freeze/pornography-2k/metrics"
epochs=50

# echo "Training ResNet50"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "resnet50" \
#        --epochs $epochs \
#        --wandb
        
# echo "Training ResNet101"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "resnet101" \
#        --epochs $epochs \
#        --wandb

# echo "Training ResNet152"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "resnet152" \
#        --epochs $epochs \
#        --wandb

# echo "Training DenseNet121"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "densenet121" \
#        --epochs $epochs \
#        --wandb

# echo "Training DenseNet169"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "densenet169" \
#        --epochs $epochs \
#        --wandb

# echo "Training DenseNet201"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "densenet201" \
#        --epochs $epochs \
#        --wandb

# echo "Training AlexNet"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "alexnet" \
#        --epochs $epochs \
#        --wandb

# echo "Training VGG16"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "vgg16" \
#        --epochs $epochs \
#        --wandb

# echo "Training VGG19"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "vgg19" \
#        --epochs $epochs \
#        --wandb

echo "Training VGG16_BN"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vgg16_bn" \
       --epochs $epochs \
       --wandb

echo "Training VGG19_BN"
python -m src.model_training \
       --project_title "$project_title" \
       --data_loc "$data_loc" \
       --model_save_loc "$model_save_loc" \
       --metrics_save_loc "$metrics_save_loc" \
       --model_name "vgg19_bn" \
       --epochs $epochs \
       --wandb

# echo "Training MobileNetV2"
# python -m src.model_training \
#        --project_title "$project_title" \
#        --data_loc "$data_loc" \
#        --model_save_loc "$model_save_loc" \
#        --metrics_save_loc "$metrics_save_loc" \
#        --model_name "mobilenet_v2" \
#        --epochs $epochs \
#        --wandb
