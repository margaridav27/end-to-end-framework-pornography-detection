#!/bin/bash
#
#SBATCH --partition=gpu_min8GB       # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8GB             # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=baseline_training # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output. If ommited, use STDOUT.

echo "Running baseline training job"

# ResNet50 on Pornography-800
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed" \
       --save_loc "baseline/pornography-800/models" \
       --model_name "resnet50"

# ResNet50 on Pornography-2k
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed" \
       --save_loc "baseline/pornography-2k/models" \
       --model_name "resnet50"

# DenseNet121 on Pornography-800
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed" \
       --save_loc "baseline/pornography-800/models" \
       --model_name "densenet121"

# DenseNet121 on Pornography-2k
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed" \
       --save_loc "baseline/pornography-2k/models" \
       --model_name "densenet121"

# VGG16 on Pornography-800
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed" \
       --save_loc "baseline/pornography-800/models" \
       --model_name "vgg16"

# VGG16 on Pornography-2k
python src/baseline/baseline_training.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed" \
       --save_loc "baseline/pornography-2k/models" \
       --model_name "vgg16"
