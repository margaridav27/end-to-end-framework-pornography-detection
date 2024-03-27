#!/bin/bash
#
#SBATCH --partition=gpu_min12GB               
#SBATCH --qos=gpu_min12GB_ext                 
#SBATCH --job-name=explainability_train
#SBATCH -o explainability_train.out               
#SBATCH -e explainability_train.err

python -m src.model_explainability \
       --state_dict_loc "results/even-20/data-aug/pornography-2k/models/vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth" \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "results/even-20/data-aug/pornography-2k/explanations-train" \
       --filter "all" \
       --partition "train" \
       --batch_size 4 \
       --method "LRP-CMP" \
       --to_explain "vPorn000890#0.jpg" \
                    "vPorn000354#0.jpg" \
                    "vPorn000518#0.jpg" \
                    "vPorn000278#0.jpg" \
                    "vPorn000214#0.jpg" \
                    "vPorn000846#0.jpg" \
                    "vPorn000605#0.jpg" \
                    "vPorn000458#0.jpg" \
                    "vPorn000083#0.jpg" \
                    "vPorn000060#0.jpg" \
                    "vNonPorn000120#0.jpg" \
                    "vNonPorn000973#0.jpg" \
                    "vNonPorn000183#0.jpg" \
                    "vNonPorn000448#0.jpg" \
                    "vNonPorn000625#0.jpg" \
                    "vNonPorn000216#0.jpg" \
                    "vNonPorn000648#0.jpg" \
                    "vNonPorn000254#0.jpg" \
                    "vNonPorn000942#0.jpg" \
                    "vNonPorn000859#0.jpg"
