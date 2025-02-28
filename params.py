from os.path import join
from os import walk
from typing import List, Dict

from safetensors.torch import load_file


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


# model parameters
MODEL_NAME = "gpt2-xl"
PATH_SUFFIX = "/mnt/parscratch/users/acx24yc/memit_cyj"
INITIAL_MODEL_SAVE_PATH = "ini_model"
EDIT_MODEL_SAVE_PATH = "new_model"
LORA_MODEL_SAVE_PATH = "new_model_lora"
