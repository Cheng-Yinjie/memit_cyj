#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=memit_test.out
#SBATCH --error=memit_test.err
#SBATCH --partition=gpu
#SBATCH --qos=gpumem
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=memit

export SLURM_EXPORT_ENV=ALL

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# running model eval script
cd .
srun python run_test.py