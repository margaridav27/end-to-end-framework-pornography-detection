#!/bin/bash
#
#SBATCH --partition=gpu_min24GB               
#SBATCH --qos=gpu_min24GB_ext                 
#SBATCH --job-name=training_aug_special_800
#SBATCH -o training_aug_special_800.out               
#SBATCH -e training_aug_special_800.err


# echo "Training VGG16 (wo middle class, 20 frames)"
# python -m src.model_training \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20-no-diff" \
#        --model_save_loc "results/even-20-no-diff/data-aug/pornography-800/models" \
#        --metrics_save_loc "results/even-20-no-diff/data-aug/pornography-800/metrics" \
#        --model_name "vgg16" \
#        --epochs 50 \
#        --data_aug

# echo "Training VGG16 (wo middle class, 40 frames)"
# python -m src.model_training \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-40-no-diff" \
#        --model_save_loc "results/even-40-no-diff/data-aug/pornography-800/models" \
#        --metrics_save_loc "results/even-40-no-diff/data-aug/pornography-800/metrics" \
#        --model_name "vgg16" \
#        --epochs 50 \
#        --data_aug

echo "Training VGG16 (w middle class, 20 frames)"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
       --model_save_loc "results/even-20/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
       --model_name "vgg16" \
       --epochs 50 \
       --data_aug

echo "Training VGG16 (w middle class, 40 frames)"
python -m src.model_training \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-40" \
       --model_save_loc "results/even-40/data-aug/pornography-800/models" \
       --metrics_save_loc "results/even-40/data-aug/pornography-800/metrics" \
       --model_name "vgg16" \
       --epochs 50 \
       --data_aug

# echo "Training VGG19 (wo middle class, 20 frames)"
# python -m src.model_training \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20-no-diff" \
#        --model_save_loc "results/even-20-no-diff/data-aug/pornography-800/models" \
#        --metrics_save_loc "results/even-20-no-diff/data-aug/pornography-800/metrics" \
#        --model_name "vgg19" \
#        --epochs 50 \
#        --data_aug

# echo "Training VGG19 (wo middle class, 40 frames)"
# python -m src.model_training \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-40-no-diff" \
#        --model_save_loc "results/even-40-no-diff/data-aug/pornography-800/models" \
#        --metrics_save_loc "results/even-40-no-diff/data-aug/pornography-800/metrics" \
#        --model_name "vgg19" \
#        --epochs 50 \
#        --data_aug

# echo "Training VGG19 (w middle class, 20 frames)"
# python -m src.model_training \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20" \
#        --model_save_loc "results/even-20/data-aug/pornography-800/models" \
#        --metrics_save_loc "results/even-20/data-aug/pornography-800/metrics" \
#        --model_name "vgg19" \
#        --epochs 50 \
#        --data_aug

# echo "Training VGG19 (w middle class, 40 frames)"
# python -m src.model_training \
#        --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-40" \
#        --model_save_loc "results/even-40/data-aug/pornography-800/models" \
#        --metrics_save_loc "results/even-40/data-aug/pornography-800/metrics" \
#        --model_name "vgg19" \
#        --epochs 50 \
#        --data_aug