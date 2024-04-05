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
destination_dir="/nas-ctm01/datasets/public/BIOMETRICS/apd-video-db/data-agg"

# Create destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Move images from porn directory
find "$source_dir/porn" -maxdepth 1 -type f -exec mv -t "$destination_dir" {} +

# Move images from non-porn directories
find "$source_dir/nonPorn" \( -name "COCO" -o -name "GVIS" -o -name "ILSVRC2012" -o -name "IMDB-WIKI" \) -type f -exec mv -t "$destination_dir" {} +

echo "Images moved successfully."

python -m src.db_utilities.setup_apd_db --data_loc "$destination_dir"

echo "APD-VIDEO all setup."