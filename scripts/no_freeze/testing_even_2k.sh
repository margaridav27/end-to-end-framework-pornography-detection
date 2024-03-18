#!/bin/bash
#
#SBATCH --partition=gpu_min8GB               
#SBATCH --qos=gpu_min8GB_ext                 
#SBATCH --job-name=testing_even_20_2k
#SBATCH -o testing_even_20_2k.out               
#SBATCH -e testing_even_20_2k.err

echo "Running testing job on Pornography-2k (even-20)"

echo "Testing ResNet50"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet50_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing ResNet101"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet101_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing ResNet152"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/resnet152_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing DenseNet121"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/densenet121_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing DenseNet169"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/densenet169_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing DenseNet201"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/densenet201_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing AlexNet"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/alexnet_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing VGG16"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/vgg16_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing VGG19"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"

echo "Testing MobileNetV2"
python -m src.model_testing \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/no-freeze/pornography-2k/results" \
       --state_dict_loc "results/even-20/no-freeze/pornography-2k/models/mobilenetv2_freeze_False_epochs_50_batch_16_optim_sgd_aug_False_split_10_20.pth"
