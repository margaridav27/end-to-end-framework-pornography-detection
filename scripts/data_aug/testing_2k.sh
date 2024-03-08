#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=testing_no_freeze_2k
#SBATCH -o slurm.%N.%j.out               
#SBATCH -e slurm.%N.%j.err

echo "Running testing job (with data augmentation) on Pornography-2k"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
       --save_loc "results/data-aug/pornography-2k/results" \
       --state_dict_loc "results/data-aug/pornography-2k/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing ResNet101"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/resnet101_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing ResNet152"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/resnet152_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing DenseNet121"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/densenet121_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing DenseNet169"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/densenet169_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing DenseNet201"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/densenet201_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing AlexNet"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/alexnet_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing VGG16"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/vgg16_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing VGG19"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/vgg19_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \

# echo "Testing MobileNetV2"
# python -m src.model_testing \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20" \
#        --save_loc "results/data-aug/pornography-2k/results" \
#        --state_dict_loc "results/data-aug/pornography-2k/models/mobilenetv2_freeze_False_epochs_10_batch_16_optim_sgd_optimized_False_aug_True.pth" \
