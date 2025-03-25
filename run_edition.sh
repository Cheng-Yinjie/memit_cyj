#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=memit_edit.out
#SBATCH --error=memit_edit.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=memit

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate conda environment using .sh file
CONDA_HOME="/opt/apps/testapps/common/software/staging/Anaconda3/2024.02-1/condabin/conda"
source ./scripts/setup_conda.sh "$CONDA_HOME"
source activate memit

# Set up CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Running model editing script
cd ..
python run_edition.py  \
    --model_name 'unsloth/llama-2-7b-chat' \
    --ini_model_save_path 'ini_model_llama-2-7b' \
    --edited_model_save_path 'new_model_llama-2-7b'
