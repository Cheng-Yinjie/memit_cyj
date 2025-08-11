import os
import re

import pandas as pd
import torch
from peft_local import PeftModel as PeftModel_Local
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_load(
        model_path: str,
        model_name: str = " ",
        adapter_path: str = " ",
        adapter_name: str = " ",
        load_dtype: any = None,
):
    def keyword_detector(keyword, folder_path):
        if (not folder_path) or (not os.path.exists(folder_path)):
            return False
        for filename in os.listdir(folder_path):
            if keyword.lower() in filename.lower():  # case-insensitive match
                return True
        return False

    def parse_args():
        final_model_path, final_tokenizer_path = model_name, model_name
        load_adapter_flag = False
        if model_path and os.path.exists(model_path):
            final_model_path = model_path
            if keyword_detector("tokenizer.json", model_path):
                final_tokenizer_path = model_path
                load_adapter_flag = False
        if adapter_path and adapter_name:
            if os.path.exists(adapter_path) and adapter_name.lower() in ["dora", "lora"]:
                load_adapter_flag = True
        return final_model_path, final_tokenizer_path, load_adapter_flag

    # Load model and tokenizer
    model_path_fnl, token_path_fnl, adapter_flag = parse_args()
    load_dtype = load_dtype if load_dtype else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path_fnl,
        device_map="auto",
        torch_dtype=load_dtype,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(token_path_fnl, trust_remote_code=True)

    # Load adapter if adapter is DoRA and adapter_path is specified
    if adapter_flag:
        model = PeftModel_Local.from_pretrained(
            model,
            adapter_path,
            torch_dtype=torch.float16)
        key_list = [(key, module) for key, module in model.model.named_modules()]
        for key, module in key_list:
            if isinstance(model.peft_config.target_modules, str):
                target_module_found = re.fullmatch(model.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.target_modules)

            if adapter_name == "dora":
                if model.peft_config.Wdecompose_target_modules != None:
                    if isinstance(model.peft_config.Wdecompose_target_modules, str):
                        wdecompose_target_module_found = re.fullmatch(model.peft_config.Wdecompose_target_modules, key)
                    else:
                        wdecompose_target_module_found = any(
                            key.endswith(target_key) for target_key in model.peft_config.Wdecompose_target_modules)
                else:
                    wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = False

            if target_module_found:
                module.merge_weights = True
                module.train(mode=False)
            elif wdecompose_target_module_found:
                module.merge_weights = True
                module.train(mode=False)
    return model, tokenizer


class Prompt4Lora:
    """
    configs for tokenizer and prompts settings
    """

    def __init__(self, tokenizer, cutoff_len, model_name, train_on_inputs):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.model_name = model_name
        self.train_on_inputs = train_on_inputs

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            if "chatglm" not in self.model_name:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in self.model_name:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(self, data_point):
        generate_prompt, _ = customize_prompt(self.model_name)
        full_prompt = generate_prompt(data_point.get("instruction"), data_point.get("input", None),
                                      data_point.get("output"))
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = generate_prompt(data_point.get("instruction"), data_point.get("input", None), "")
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                         user_prompt_len:]
        return tokenized_full_prompt

    def dataset_splitter(self, data, val_set_size):
        """
        to split raw dataset into training dataset and validation dataset
        Returns: train_data and val_data
        """
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            val_data = None
        return train_data, val_data


def customize_prompt(model_name: str):
    def generate_prompt_llama(instruction: str, input=None, output=""):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {instruction}

                    ### Input:
                    {input}

                    ### Response:
                    {output}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                    ### Instruction:
                    {instruction}

                    ### Response:
                    {output}"""

    def generate_prompt_gpt(instruction: str, input=None, output=""):
        if input:
            return f"""Question: {instruction}\nInput: {input}\nAnswer: {output}"""
        else:
            return f"""Question: {instruction}\nAnswer: {output}"""

    if "llama" in model_name.lower():
        return generate_prompt_llama, "### Response:"
    elif "gpt" in model_name.lower():
        return generate_prompt_gpt, "Answer:"
    else:
        raise ValueError("model not supported")


def find_max_checkpoint_folder(base_dir):
    pattern = re.compile(r'^checkpoint-(\d+)$')
    max_num = -1
    max_folder = None

    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                max_folder = name
    return max_folder