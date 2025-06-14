#!/bin/bash
set -e

# Model parameters
MODEL_NAME="gpt2-xl" # meta-llama/Llama-2-7b-hf
MODEL_PATH="gpt2-xl"
ADAPTER_NAME="dora"
ADAPTER_PATH="gpt2-xl_MEMIT_zsre_10000_dora"
DS_NAME="zsre" # [cf, mcf, zsre]

# Edit parameters
N_EDITS="10000"
ALG_NAMES=("MEMIT")
HPARAMS_FNAMES=("gpt2-xl.json") # meta-llama_Llama-2-7b-hf.json
EVAL_ONLY=1
MODEL_SAVE=0

# Execute
for i in ${!ALG_NAMES[@]}
do
    alg_name=${ALG_NAMES[$i]}
    hparams_fname=${HPARAMS_FNAMES[$i]}

    echo "Running evals for $alg_name..."

    python3 -m run_edit \
        --alg_name=$alg_name --model_name=$MODEL_NAME --model_path=$MODEL_PATH \
        --adapter_name=$ADAPTER_NAME --adapter_path=$ADAPTER_PATH \
        --hparams_fname=$HPARAMS_FNAMES --num_edits=$N_EDITS --use_cache \
        --dataset_size_limit=$N_EDITS --ds_name=$DS_NAME --eval_only=$EVAL_ONLY \
        --model_save=$MODEL_SAVE
done
exit 0
 