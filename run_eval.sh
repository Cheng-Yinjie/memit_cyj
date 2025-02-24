#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=memit_eval.out
#SBATCH --error=memit_eval.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=memit

export SLURM_EXPORT_ENV=ALL

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate conda environment using .sh file
source activate memit

# running model editing script
srun python -m experiments.evaluate \
    --alg_name=MEMIT \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --num_edits=100 \
    --use_cache