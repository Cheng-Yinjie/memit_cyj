
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import fire
import json
import os
import re
import sys
import warnings
from os.path import join
from tqdm import tqdm

import torch
from transformers import (
    GenerationConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer)

sys.path.append(os.path.join(os.getcwd(), "peft_local/src/"))
from peft_local import PeftModel
from util.generate import RecordTimer

warnings.filterwarnings("ignore")


def run_multiple_tasks(
        # Model parameters
        model_name: str, 
        model_folder_path: str, 
        load_8bit: bool = False,
        # Fine-tune parameters
        adapter_path: str = None,
        adapter_type: str = "DoRA",
        task_list: str = "piqa",
        # Eval parameters
        batch_size: int = 1
        ):
    for task_name in task_list:
        run_downstream_tasks(model_name, model_folder_path, load_8bit, adapter_path, adapter_type, task_name, batch_size)


def run_downstream_tasks(
        # Model parameters
        model_name: str, 
        model_folder_path: str, 
        load_8bit: bool = False,
        # Fine-tune parameters
        adapter_path: str = None,
        adapter_type: str = "DoRA",
        task_name: str = "piqa",
        # Eval parameters
        batch_size: int = 1
        ):
    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32
            ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams
        )
        with torch.no_grad():
            if adapter_path:
                input_ids = input_ids.to("cuda")
                model.to("cuda")
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens)
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs
    
    # Setup params
    rt = RecordTimer(model_folder_path, "time_records", f"downstream_task_{task_name}")
    folder_name = adapter_path if adapter_path else model_folder_path
    adapter_type = adapter_type if adapter_path else None
    os.makedirs(f"downstream_tasks_{folder_name}", exist_ok=True)
    save_file = join(folder_name, f"result_downstream_{task_name}.json")

    # Load model and tokenizer
    dataset = load_data(task_name)
    batches = create_batch(dataset, batch_size)
    tokenizer, model = load_model(model_folder_path, model_name, adapter_path, load_8bit)

    if adapter_type == "LoRA" or adapter_type == "DoRA":
        key_list = [(key,module) for key, module in model.model.named_modules()]
        for key,module in key_list: 
            if isinstance(model.peft_config.target_modules, str):
                target_module_found = re.fullmatch(model.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.target_modules)

            if adapter_type == "DoRA":
                if model.peft_config.Wdecompose_target_modules != None:
                    if isinstance(model.peft_config.Wdecompose_target_modules, str):
                        wdecompose_target_module_found = re.fullmatch(model.peft_config.Wdecompose_target_modules, key)
                    else:
                        wdecompose_target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.Wdecompose_target_modules)
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
                
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    rt.record(f"Start evaluating for task {task_name}")
    for _, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)
        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(task_name, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    rt.record(f"Finish evaluating for task {task_name}")
    print("output path: ", save_file)


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(task_name) -> list:
    """
    read data from dataset file
    """
    file_path = f'dataset/{task_name}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def load_model(model_folder_path, model_name, adapter_path, load_8bit) -> tuple:
    """
    load tuned model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (0)

    model = AutoModelForCausalLM.from_pretrained(
        model_folder_path,
        load_in_8bit=load_8bit,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map={"":0}
        )
    return tokenizer, model
2

def extract_answer(task_name: str, sentence: str) -> float:
    if task_name == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task_name == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task_name in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task_name == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task_name == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    fire.Fire(run_multiple_tasks)