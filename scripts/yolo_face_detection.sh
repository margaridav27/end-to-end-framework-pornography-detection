#!/bin/bash
#
#SBATCH --partition=gpu_min8gb      
#SBATCH --qos=gpu_min8gb_ext        
#SBATCH --job-name=yolo_face_detection      
#SBATCH -o yolo_face_detection.out          
#SBATCH -e yolo_face_detection.err          


python -m src.yolo_face_detection \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20" \
       --save_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20-detected-faces"
       