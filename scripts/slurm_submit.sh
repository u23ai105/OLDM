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
# 1. Download the movie (skips if already downloaded)
echo "Ensuring sample movie is downloaded..."
bash scripts/download_movie.sh

# 2. Download model weights (skips if already downloaded)
echo "Ensuring all AI model weights are downloaded..."
python scripts/download_weights.py

# Determine the input video (use argument if provided, else use default downloaded movie)
INPUT_VIDEO=${1:-data/raw/night_of_the_living_dead.mp4}

# 3. Run the full production pipeline
echo "Starting pipeline for video: $INPUT_VIDEO"
python -m src.pipeline.flows "$INPUT_VIDEO"