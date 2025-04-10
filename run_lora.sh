 #!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=lora_run.out
#SBATCH --error=lora_run.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=lora

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# running model editing script
python run_lora.py \
    --model_folder_path 'ini_model_gpt2-xl_250326' \
    --model_name 'gpt2-xl' --data_path 'zwhe99/commonsense_170k' \
    --adapter_path 'ini_model_lora_gpt2-xl_250326' \
    --output_dir 'ini_model_dora_gpt2-xl_250326' \
    --num_epochs 1 --learning_rate 2e-4 --val_set_size 120 \
    --cutoff_len 256 --lora_r 8 --lora_alpha 32 --lora_dropout 0.05 \
    --target_modules '["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.up_proj", "self_attn.down_proj"]' \
    --train_on_inputs True
 