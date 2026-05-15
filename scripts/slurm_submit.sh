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

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 1. Download model weights (skips if already downloaded)
echo "Ensuring all AI model weights are downloaded..."
python scripts/download_weights.py

# 2. Run the full production pipeline
echo "Starting pipeline for video: $1"
python -m src.pipeline.flows "$1"