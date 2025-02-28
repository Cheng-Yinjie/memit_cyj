from typing import Dict
import json

import pandas as pd


class GenCounterFactRequests:
    def __init__(self, row_num: int):
        self.row_num = row_num
    
    def download_data(self):
        splits = {
            'train': 'data/train-00000-of-00001-05d11247db7abce8.parquet', 
            'test': 'data/test-00000-of-00001-bacb83500fca49a9.parquet'
            }
        df = pd.read_parquet("hf://datasets/azhx/counterfact/" + splits["train"])
        df = df[df.index <= self.row_num] 
        return df
    
    @staticmethod
    def gen_request(json_dict: Dict) -> Dict:
        request = dict()
        request["prompt"] = json_dict.get("prompt")
        request["subject"] = json_dict.get("subject")
        request["target_new"] = {"str": json_dict.get("target_new").get("str")}
        return request

    def proc(self):
        requests = list()
        df = self.download_data()
        for json_dict in df["requested_rewrite"].to_list():
            requests.append(self.gen_request(json_dict))
        return requests
