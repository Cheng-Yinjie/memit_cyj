#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=dora_run.out
#SBATCH --error=dora_run.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=dora

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# Dora gpt2-xl
python run_dora.py \
    --model_folder_path 'gpt2-xl-AlphaEdit_mcf_1000' \
    --model_name 'gpt2-xl' --data_path 'commonsense_170k.json' \
    --adapter_name 'dora' --batch_size 16 \
    --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 \
    --weight_decay 0.0 --use_gradient_checkpointing True --val_set_size 120 \
    --eval_step 80 --save_step 80 --cutoff_len 256 --lora_r 32 --lora_alpha 64 \
    --lora_dropout 0.05 --dora_simple True --Wdecompose_target_modules None --train_on_inputs True

# Lora gpt2-xl
python run_dora.py \
    --model_folder_path 'gpt2-xl-AlphaEdit_mcf_1000' \
    --model_name 'gpt2-xl' --data_path 'commonsense_170k.json' \
    --adapter_name 'lora' --batch_size 16 \
    --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 \
    --weight_decay 0.0 --use_gradient_checkpointing True --val_set_size 120 \
    --eval_step 80 --save_step 80 --cutoff_len 256 --lora_r 32 --lora_alpha 64 \
    --lora_dropout 0.05 --dora_simple True --Wdecompose_target_modules None --train_on_inputs True

# Dora gptj-6b
python run_finetune_peft.py \
    --model_folder_path 'EleutherAI_gpt-j-6B' --model_name 'EleutherAI/gpt-j-6B' \
    --data_path 'zwhe99/commonsense_170k' --adapter_name 'dora' --output_dir 'gptj-6b_dora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 --cutoff_len 256 \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True --Wdecompose_target_modules None \
    --train_on_inputs True

# Dora llama2-7b
python run_dora.py \
    --model_folder_path "Llama-2-7b-hf-AlphaEdit_mcf_10000" --model_name 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' --adapter_name 'dora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 \
    --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --train_on_inputs True


# Lora llama2-7b
python run_dora.py \
    --model_folder_path "Llama-2-7b-hf-AlphaEdit_mcf_10000" --model_name 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' --adapter_name 'lora' \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 \
    --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --train_on_inputs True


# Full-size finetune llama2-7b
python run_finetune_fullsize.py \
    --model_folder_path "Llama-2-7b-hf-AlphaEdit_mcf_1000" --model_name 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' --learning_rate 5e-6 --batch_size 1 --num_epochs 3 \
    --save_step 5000 --log_step 20  --cutoff_len 1024
