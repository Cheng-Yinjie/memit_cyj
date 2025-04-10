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
python run_downstream_tasks_upd.py \
    --model_name 'meta-llama/Llama-2-7b-hf' \
    --model_folder_path 'meta-llama2-7b' \
    --adapter_path 'meta-llama2-7b_dora' \
    --adapter_type 'DoRA' --task_name 'winogrande' \
    --batch_size 1

python run_downstream_tasks_upd.py --model_name 'meta-llama/Llama-2-7b-hf' --model_folder_path 'meta-llama2-7b' --adapter_path 'meta-llama2-7b_dora' --task_list '["boolq", "piqa", "ARC-Easy", "openbookqa", "winogrande", "hellaswag"]' --batch_size 1
