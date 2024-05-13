#!/bin/bash
#
#SBATCH --partition=gpu_min8gb         
#SBATCH --qos=gpu_min8gb_ext            
#SBATCH --job-name=testing_baseline_2k
#SBATCH -o testing_baseline_2k.out               
#SBATCH -e testing_baseline_2k.err 


data_loc="/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20"
save_loc="results/pornography-2k/cnns/baseline/middle-20/results"
state_dict_loc="results/pornography-2k/cnns/baseline/middle-20/models"


echo "Running baseline testing job on Pornography-2k"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/resnet50_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/resnet101_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/resnet152_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/densenet121_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/densenet169_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/densenet201_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/alexnet_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg16_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/vgg19_freeze_True_epochs_20_batch_32_optim_sgd.pth"

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "$data_loc" \
       --save_loc "$save_loc" \
       --state_dict_loc "$state_dict_loc/mobilenet_v2_freeze_True_epochs_20_batch_32_optim_sgd.pth"
