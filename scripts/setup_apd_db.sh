#!/bin/bash
#
#SBATCH --partition=gpu_min8GB       # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8GB_ext         # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=apd_db_setup      # Job name
#SBATCH -o apd_db_setup.out          # File containing STDOUT output
#SBATCH -e apd_db_setup.err          # File containing STDERR output. If ommited, use STDOUT.

# Source directory
source_dir="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data"

# Destination directory
dest_dir="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-agg"

# Create destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Copy images from porn directory
find "$source_dir/porn/" -maxdepth 1 -type f -print0 | xargs -0 cp -t "$dest_dir"

# Copy images from non-porn directories
find "$source_dir/nonPorn/COCO/" -maxdepth 1 -type f -print0 | xargs -0 cp -t "$dest_dir"
find "$source_dir/nonPorn/GVIS/" -maxdepth 1 -type f -print0 | xargs -0 cp -t "$dest_dir"
find "$source_dir/nonPorn/ILSVRC2012/" -maxdepth 1 -type f -print0 | xargs -0 cp -t "$dest_dir"
find "$source_dir/nonPorn/IMDB-WIKI/" -maxdepth 1 -type f -print0 | xargs -0 cp -t "$dest_dir"

echo "Images moved successfully."

python -m src.db_utilities.setup_apd_db --data_loc "$dest_dir"

echo "APD-VIDEO all setup."
