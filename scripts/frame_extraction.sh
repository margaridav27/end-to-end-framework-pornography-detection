#!/bin/bash
#
#SBATCH --partition=cpu_14cores      # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=cpu_14cores            # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=frame_extraction  # Job name
#SBATCH -o slurm.%N.%j.out           # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err           # File containing STDERR output. If ommited, use STDOUT.

echo "Running frame extraction job"

python src/frame_extraction/frame_extraction.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data/original" \
       --save_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed" \
       --n_frames=40

python src/frame_extraction/frame_extraction.py \
       --data_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data/Database/vNonPornDifficulty" \
                  "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data/Database/vNonPornEasy" \
                  "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data/Database/vPorn" \
       --save_loc "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed" \
       --n_frames 80
