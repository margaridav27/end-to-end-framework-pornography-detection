#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=testing_800
#SBATCH -o testing_800.out               
#SBATCH -e testing_800.err

echo "Running testing job on Pornography-800 (middle-20)"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_resnet50_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_resnet101_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_resnet152_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_densenet121_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_densenet169_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_densenet201_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_alexnet_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_vgg16_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_vgg19_freeze_False_epochs_10_batch_32_optim_sgd.pth"

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/middle-20" \
       --save_loc "results/middle-20/no-freeze/pornography-800/results" \
       --state_dict_loc "results/middle-20/no-freeze/pornography-800/models/model_mobilenet_v2_freeze_False_epochs_10_batch_32_optim_sgd.pth"
