#!/bin/bash
set -e

# Constants
DATASETS=("boolq" "openbookqa" "winogrande" "piqa" "ARC-Challenge" "ARC-Easy" "hellaswag" "social_i_qa")
MODEL_NAME="meta-llama/Llama-2-7b-hf"
ADAPTER_NAME="DoRA"
MODEL_PATH="Llama-2-7b-hf-AlphaEdit_mcf_10000"
ADAPTER_PATH="Llama-2-7b-hf-AlphaEdit_mcf_10000_dora"

# running model editing script
for i in "${!DATASETS[@]}"
do
    task="${DATASETS[$i]}"
    CUDA_LAUNCH_BLOCKING=1 python run_downstream_tasks.py \
        --dataset $task \
        --model_name $MODEL_NAME \
        --adapter_name $ADAPTER_NAME \
        --model_path $MODEL_PATH \
        --adapter_path $ADAPTER_PATH \
        --batch_size 1
done