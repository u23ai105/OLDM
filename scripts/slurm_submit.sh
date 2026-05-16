#!/bin/bash

#SBATCH --job-name=MM-gpu

#SBATCH --partition=gpu
#SBATCH --nodelist=node2
#SBATCH --error=/home/pkc/MM/SAMEG-Polyp-Segmentation/dataset/TestDataset/logs/test_gpu_error_dataset_downloader_%j.log
#SBATCH --output=/home/pkc/MM/SAMEG-Polyp-Segmentation/dataset/TestDataset/logs/test_gpu_job_output_dataset_downloader_%j.log
#SBATCH --gres=shard:20
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024.2  
module load cuda-12.8
module load ffmpeg

source venv/bin/activate
pip install -r requirements.txt
# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 1. Download model weights (skips if already downloaded)
echo "Ensuring all AI model weights are downloaded..."
python scripts/download_weights.py

# 2. Handle Movie Input (Download default if none provided)
INPUT_VIDEO="$1"
if [ -z "$INPUT_VIDEO" ]; then
    echo "No input video provided. Downloading default movie (Night of the Living Dead)..."
    mkdir -p data/raw
    INPUT_VIDEO="data/raw/night_of_the_living_dead.mp4"
    wget --continue -q --show-progress \
        -O "$INPUT_VIDEO" \
        https://archive.org/download/Night_Of_The_Living_Dead_raw_HD_WS/Night_Of_The_Living_Dead_raw_HD_WS.mp4
else
    echo "Using provided input video: $INPUT_VIDEO"
fi

# 3. Run the full production pipeline
echo "Starting pipeline for video: $INPUT_VIDEO"
python -m src.pipeline.flows "$INPUT_VIDEO"