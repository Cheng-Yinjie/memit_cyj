#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=dora_run.out
#SBATCH --error=dora_run.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=dora

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# running model editing script
python run_lora.py \
    --model_folder_path 'new_model_gpt2-xl_250312' \
    --model_name 'gpt2-xl' \
    --data_path 'zwhe99/commonsense_170k' \
    --output_dir 'new_model_dora_gpt2-xl_250312' \
    --batch_size 32  --micro_batch_size 32 --num_epochs 1 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 160 --save_step 160  --adapter_name 'dora' \
    --target_modules '["c_attn"]' \
    --lora_r 8 --lora_alpha 32 \
    --use_gradient_checkpointing True