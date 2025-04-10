# Edit Inherit

## Table of Contents

- [Settings](#settings)
  - [Environment](#environment)
  - [Datasets and models](#datasets-and-models)
- [Demos](#environment)
  - [Model Edit](#model-edit)
  - [Model Fine-tune](#model-fine-tune)
  - [Downstream Tasks for Evaluation](#downstream-tasks-for-evaluation)
  - [Perplexity Index (PPL)](#perplexity-index-ppl)
  - [Mannual Conversation](#mannual-conversation)
- [Reference](#reference)
  - [Model Editing](#model-editing)
  - [Model Fine-tuning](#model-fine-tuning)

## Settings
### Environment
We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```
`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`.
### Datasets and models
For models, we use LLama-2, gpt2-xl and deepseek. For datasets, we use [Counterfact](https://huggingface.co/datasets/azhx/counterfact) for model editing, [Commonsense170k](https://huggingface.co/datasets/zwhe99/commonsense_170k) for model finetuning and [wikitext-2-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext) for ppl index calculation.


## Demos
The code was intended for HPC, you can either submit  to HPC or just run on your own devices. The first couple of rows in every `.sh` scriptes are settings for HPC, you can edit or just delete it. 
```bash
#!/bin/bash
#SBATCH --mail-user=your own email address
#SBATCH --mail-type=ALL
#SBATCH --output=memit_edit.out
#SBATCH --error=memit_edit.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=job name
```
<br>

### Model Edit
`run_edition.py` and `run_edition.sh` are scripts needed. Please run `run_edition.sh`, the outputs of the edition will be two folders storing original model and edited model respectively. In the below code block, `ini_model_save_path` is path for the original model and `edited_model_save_path` is the path for the edited model.

```bash
python run_edition.py  \
    --model_name 'unsloth/llama-2-7b-chat' \
    --ini_model_save_path 'ini_model_llama-2-7b' \
    --edited_model_save_path 'new_model_llama-2-7b'
```
If more model need to be edited, please prepare hyper-parameters for new model. New parameters is stored under `hparams/MEMIT`, named by the model. For example, if new model is llama-7b, then a new json file named `llama-7b.json` should be prepared. After preparing the hparams, change the `model_name` in the code block.
<br>


### Model Fine-tune
`run_dora.py`, `run_dora.sh`, `run_dora.py` and `run_dora.sh` are scripts needed. They come in pairs, each python script corresponds to a shell script respectively. Please run `run_dora.sh`, the outputs of the edition will be a folder storing the fine-tuned model. In the below code block, `model_folder_path` is path for the original model awaiting for fine-tuning.For the rest parameters, please refer to ['`evaluate.py`'](https://github.com/locuslab/wanda/blob/main/lib/eval.py#L132).
```bash
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
```
If more model need to be edited, please prepare hyper-parameters for new model. New parameters is stored under `hparams/MEMIT`, named by the model. For example, if new model is llama-7b, then a new json file named `llama-7b.json` should be prepared. After preparing the hparams, change the model_name in the code block.
<br>

### Downstream Tasks for Evaluation
Downstream tasks are used to evaluate the effectiveness of the model after editing/finetuning. There are 8 tasks ("boolq", "piqa", "siqa_ca", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa") evaluating the effectiveness of the model. It is put forward in the paper [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/pdf/2402.09353). `run_downstream_tasks.py` and `run_downstream_tasks.sh` are scripts related.

The dataset used for evaluation can be download from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset). Put the `dataset` folder under the root path and run the scriot. By changing the below python code (95 row) in `run_downstream_tasks.py`, you can adjust the tasks need to be run.

```python
task_list_total = ["boolq", "piqa", "siqa_ca", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
```
Below code block is shell script, the parameters can refer to ['`evaluate.py`'](https://github.com/locuslab/wanda/blob/main/lib/eval.py#L132).
```bash
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
```
<br>

### Perplexity Index (PPL)
Perplexity(PPL) is an index used to evaluate a model's language modeling capabilities. `run_ppl.py` and `run_ppl.sh` are scripts related.\
For the below code block, `model_folder` is the path of the model need ppl index calculation and `model_name` is the model name.
```bash
python run_ppl.py  \
    --model_name 'gpt2-xl' \
    --model_folder 'ini_model_gpt2-xl_250316'
```
<br>

### Mannual Conversation
`run_mannual_test.py` is the file for mannual check. You can check model's performance by providing a specific question (usually is the concept being edited.) For example, one of the concept is 'The capital city of USA is Washington', it has been updated as 'The capital city of USA is London', then for this part, we can ask 'The capital city of USA is'. The code in `run_mannual_test.py` can be written as:
```python
model_name = "gpt2-xl"
model_folder = "new_model_gpt2-xl_250316"
prompt = "The capital city of USA is"
adapter_folder = "new_model_lora_gpt2-xl_250316"
```
<br>

## Reference
### Model Editing
For demos and more information about model editing using MEMIT or ROME, please refer to [MEMIT](https://github.com/kmeng01/memit/blob/main).

### Model Fine-tuning
For LoRA, refer to [LoRA](https://github.com/microsoft/LoRA).\
For DoRA, refer to [DoRA](https://github.com/NVlabs/DoRA).