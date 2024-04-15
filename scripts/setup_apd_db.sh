#!/bin/bash
#
#SBATCH --partition=gpu_min11GB      # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min11GB_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=apd_db_setup      # Job name
#SBATCH -o apd_db_setup.out          # File containing STDOUT output
#SBATCH -e apd_db_setup.err          # File containing STDERR output. If ommited, use STDOUT.

# Source directories
source_non_porn_coco="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/COCO"
source_non_porn_gvis="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/GVIS"
source_non_porn_ilsvrc2012="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/ILSVRC2012"
source_non_porn_imdb="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/nonPorn/IMDB-WIKI"
source_porn="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data/porn"

# Destination directory
dest_dir="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-aggregated"

python -m src.db_utilities.setup_apd_db \
       --data_loc "$source_non_porn_coco" "$source_non_porn_gvis" "$source_non_porn_ilsvrc2012" "$source_non_porn_imdb" "$source_porn" \
       --save_loc "$dest_dir"

echo "APD-VIDEO all setup."
