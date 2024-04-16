#!/bin/bash
#
#SBATCH --partition=gpu_min11GB      
#SBATCH --qos=gpu_min11GB_ext        
#SBATCH --job-name=sanity_check_apd     
#SBATCH -o sanity_check_apd.out         
#SBATCH -e sanity_check_apd.err         

source_non_porn_coco="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/COCO"
source_non_porn_gvis="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/GVIS"
source_non_porn_ilsvrc2012="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/ILSVRC2012"
source_non_porn_imdb="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/IMDB-WIKI"
source_porn="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/porn"

echo "Running sanity check job on APD database"

python -m src.sanity_check \
       --data_loc "$source_non_porn_coco" "$source_non_porn_gvis" "$source_non_porn_ilsvrc2012" "$source_non_porn_imdb" "$source_porn"
