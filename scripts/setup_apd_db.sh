#!/bin/bash
#
#SBATCH --partition=cpu_14cores      
#SBATCH --qos=cpu_14cores_ext        
#SBATCH --job-name=apd_db_setup      
#SBATCH -o apd_db_setup.out          
#SBATCH -e apd_db_setup.err          

# Source directories
source_non_porn_coco="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/COCO"
source_non_porn_gvis="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/GVIS"
source_non_porn_ilsvrc2012="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/ILSVRC2012"
source_non_porn_imdb="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/IMDB-WIKI"
source_porn="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/porn"

# Destination directory
dest_dir="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"

# File with corrupted paths
corrupted_paths_loc="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/metadata/corrupted_paths.txt"

python -m src.db_utilities.setup_apd_db \
       --data_loc "$source_non_porn_coco" "$source_non_porn_gvis" "$source_non_porn_ilsvrc2012" "$source_non_porn_imdb" "$source_porn" \
       --save_loc "$dest_dir" \
       --corrupted_paths_loc "$corrupted_paths_loc"

echo "APD all setup."
