import ast
import json
import os
from abc import ABC, abstractmethod
from datetime import date, datetime
from os import walk
from os.path import join
from typing import List, Dict

from datasets import load_dataset
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


class Prompt4Lora():
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

    @staticmethod
    def generate_prompt(data_point):
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {data_point["instruction"]}
                    
                    ### Input:
                    {data_point["input"]}
                    
                    ### Response:
                    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                    ### Instruction:
                    {data_point["instruction"]}
                    
                    ### Response:
                    {data_point["output"]}"""
        
    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt
    

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
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.load_state_dict(get_whole_model_dict(model_path), strict=False)
    elif any(".bin" in str(item) for item in os.listdir(model_path)):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # load adapter if adapter_path is specified
    elif adapter_path:
        model = AutoModelForCausalLM.from_pretrained(adapter_path)
        model.load_adapter(adapter_path)
        model.set_active_adapters("default")

    # Load tokenizer
    if "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            legacy=False,
            use_auth_token=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    return model, tokenizer
    

def generate_date_string(date_format: str="%y%m%d"):
    now = datetime.now()
    return now.strftime(date_format)


class DownloadCFRequests(ABC):   
    """
    Basic class about CounterFact dataset, contains method to download dataset from HuggingFace
    """
    def __init__(self, row_num: int=0):
        self.row_num = row_num

    def download_data(self):
        splits = {
            'train': 'data/train-00000-of-00001-05d11247db7abce8.parquet', 
            'test': 'data/test-00000-of-00001-bacb83500fca49a9.parquet'
            }
        df = pd.read_parquet("hf://datasets/azhx/counterfact/" + splits["train"])
        if self.row_num:
            if self.row_num >= df.shape[0]:
                return df
            df = df.sample(n=self.row_num, replace=False) 
        return df
        
    @staticmethod
    def gen_request(json_dict: Dict) -> Dict:
        request = dict()
        request["prompt"] = json_dict.get("prompt")
        request["subject"] = json_dict.get("subject")
        request["target_new"] = {"str": json_dict.get("target_new").get("str")}
        return request
    
    @abstractmethod
    def proc(self):
        pass


class GenerateCFRequests(DownloadCFRequests):
    """
    Generate edit requests for CounterFact dataset and record adopted concepts to a csv file
    """
    def __init__(self, row_num: int=None, model_name: str=""):
        self.row_num = row_num
        self.model_name = model_name
        self.sample_folder = "sample_records"

    def record_edit_copcepts(self, df: pd.DataFrame, folder_name: str="sample_records"):
        os.makedirs(folder_name, exist_ok=True)
        full_path = join(folder_name, f"sampled_edits_{self.model_name}_{date.today()}.csv")
        print("full_path: ", full_path)
        df["case_id"].to_csv(full_path, index=False)

    def proc(self):
        requests = list()
        df = self.download_data()
        self.record_edit_copcepts(df)
        for json_dict in df["requested_rewrite"].to_list():
            requests.append(self.gen_request(json_dict))
        return requests
    

class RetrieveCFRequest(DownloadCFRequests):
    """
    Generate edit requests according to the saved records
    """
    def __init__(self, record_path: str):
        self.record_path = record_path
        self.row_num = 0
        self.dataset_name = "azhx/counterfact"
    
    def download_data(self):
        return load_dataset(self.dataset_name)["train"]

    @staticmethod
    def retrieve_data(df: any, idx_list: list):
        return df.filter(lambda x: x['case_id'] in idx_list)
    
    def proc(self):
        idx_list = pd.read_csv(self.record_path)["case_id"].to_list()
        df = self.retrieve_data(self.download_data(), idx_list)
        return df

