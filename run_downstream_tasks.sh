#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=downstream.out
#SBATCH --error=downstream.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=downstream_ini_model

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# running model editing script
python run_downstream_tasks.py \
    --model_name 'gpt2-xl' \
    --model_folder_path 'new_model_gpt2-xl_250316' \
    --adapter_path 'new_model_lora_gpt2-xl_250316'