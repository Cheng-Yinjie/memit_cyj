from datetime import datetime
from os.path import join
from os import walk
from typing import List, Dict, Union
import os
import re

from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_date_string():
    now = datetime.now()
    return now.strftime("%y%m%d")


def model_load(model_path: str, model_name: str, adapter_path: str = ''): 
    def get_whole_model_dict(dir: str) -> Dict:
        def list_safetensors_files(dir: str) -> List:
            safetensors_files = list()
            for _, _, files in walk(dir):
                for file in files:
                    if file.endswith('.safetensors'):
                        safetensors_files.append(file)
            return safetensors_files

        sub_dict_lst = list_safetensors_files(dir)
        combined_dict = dict()
        for file_path in sub_dict_lst:
            state_dict = load_file(join(dir, file_path))
            for key, value in state_dict.items():
                combined_dict[key] = value
        return combined_dict

    # Load model
    if any(".safetensors" in str(item) for item in os.listdir(model_path)):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.load_state_dict(get_whole_model_dict(model_path), strict=False)
    elif any(".bin" in str(item) for item in os.listdir(model_path)):
        model = AutoModelForCausalLM.from_pretrained(model_path)
    # load adapter if adapter_path is specified
    elif adapter_path:
        model = AutoModelForCausalLM.from_pretrained(adapter_path)
        model.load_adapter(adapter_path)
        model.set_active_adapters("default")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    return model, tokenizer


# model parameters
MODEL_NAME = "gpt2-xl"
PATH_SUFFIX = "/mnt/parscratch/users/acx24yc/memit_cyj"
INITIAL_MODEL_SAVE_PATH = f"ini_model_{MODEL_NAME}_{generate_date_string()}"
EDIT_MODEL_SAVE_PATH = f"new_model_{MODEL_NAME}_{generate_date_string()}"
LORA_MODEL_SAVE_PATH = f"new_model_lora_{MODEL_NAME}_{generate_date_string()}"
DORA_MODEL_SAVE_PATH = f"new_model_dora_{MODEL_NAME}_{generate_date_string()}"
