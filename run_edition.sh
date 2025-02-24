#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=memit_run.out
#SBATCH --error=memit_run.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=memit

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate conda environment using .sh file
CONDA_HOME="/opt/apps/testapps/common/software/staging/Anaconda3/2024.02-1/condabin/conda"
source ./scripts/setup_conda.sh "$CONDA_HOME"
source activate memit
pip install torch
pip list

# running model editing script
cd ..
python run_edition.py