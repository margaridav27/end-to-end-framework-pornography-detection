#!/bin/bash
#
#SBATCH --partition=cpu_14cores      # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=cpu_14cores_ext        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
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
cp -r "$source_dir/porn/"* "$dest_dir"

# Copy images from non-porn directories
cp -r "$source_dir/nonPorn/COCO/"* "$dest_dir"
cp -r "$source_dir/nonPorn/GVIS/"* "$dest_dir"
cp -r "$source_dir/nonPorn/ILSVRC2012/"* "$dest_dir"
cp -r "$source_dir/nonPorn/IMDB-WIKI/"* "$dest_dir"

echo "Images moved successfully."

python -m src.db_utilities.setup_apd_db --data_loc "$dest_dir"

echo "APD-VIDEO all setup."
