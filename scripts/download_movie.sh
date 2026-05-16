#!/bin/bash

# Create the raw data directory if it doesn't exist
mkdir -p data/raw

echo "========================================="
echo "Downloading Night of the Living Dead (1968)"
echo "Source: archive.org (Public Domain)"
echo "Quality: HD H.264"
echo "========================================="

# The direct download URL from Internet Archive
# Using wget with --continue to resume if it gets interrupted
wget --continue --progress=bar:force:noscroll \
    -O data/raw/night_of_the_living_dead.mp4 \
    https://archive.org/download/Night_Of_The_Living_Dead_raw_HD_WS/Night_Of_The_Living_Dead_raw_HD_WS.mp4

echo ""
echo "Download complete! Saved to: data/raw/night_of_the_living_dead.mp4"
echo "You can now submit your Slurm job with:"
echo "sbatch scripts/slurm_submit.sh data/raw/night_of_the_living_dead.mp4"
