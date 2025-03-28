import os
from dataclasses import dataclass
from os.path import join, exists
from random import random

import fire
import fnmatch
import lm_eval
from lm_eval import tasks, evaluator

from params import model_load, generate_date_string
from util.generate import RecordTimer


@dataclass
class ModelParams:
    model_name: str
    model_path: str
    model: any
    max_length: int = 256


@dataclass
class EvalSets:
    num_fewshot: int
    use_accelerate: bool
    check_integrity: bool


def eval_zero_shot(
    model_param: ModelParams, 
    eval_set: EvalSets, 
    task_pattern=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]):

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    # Prepare evaluation parameters
    final_tasks = pattern_match(task_pattern, tasks.TaskManager().all_tasks)
    print(f"Existing tasks are: {final_tasks}")
    limit = 2000 if "70b" in model_param.model_name or "65b" in model_param.model_name else None
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=model_param.model,
        tasks=final_tasks,
        num_fewshot=eval_set.num_fewshot,
        batch_size=None,
        device=None,
        limit=limit,
        check_integrity=eval_set.check_integrity
    )
    return results 


def run_downstream_tasks(
    model_name: str, 
    model_folder_path: str, 
    adapter_path: str = None,
    truncation: bool = True, 
    max_length: int = 256):
    # Input parameter check:
    adapter_path = None if adapter_path == " " else adapter_path
    truncation = True if isinstance(truncation, str) else truncation
    max_length = 1024 if isinstance(max_length, str) else max_length

    # Set model parameters
    model, tokenizer = model_load(model_folder_path, model_name, adapter_path)
    rt = RecordTimer(model_folder_path)

    wrapped_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=model_name,
        truncation=truncation, 
        max_length=max_length
        )
    model_params = ModelParams(
        model_name=model_name,
        model_path=model_folder_path,
        model=wrapped_model
        )
    
    # Set evaluation settings
    eval_set = EvalSets(
        num_fewshot=0,
        use_accelerate=False,
        check_integrity=False
        )

    # Determine downstream tasks and output the result
    result_folder_path = f"downstream_tasks_{model_folder_path}"
    if not exists(result_folder_path):
        os.mkdir(result_folder_path)

    # task_list_total = ["boolq", "piqa", "siqa_ca", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    task_list_total = ["arc_easy"]
    rt.record("time_records", "time_downstream_tasks", f"start running task series: {task_list_total}")
    for task in task_list_total:
        rt.record("time_records", "time_downstream_tasks", f"start running {task}")
        res = eval_zero_shot(model_params, eval_set, [task])
        output_file_name = join(result_folder_path, f"result_downstream_{task}.txt")
        rt.record("time_records", "time_downstream_tasks", f"finsih running {task}")
        with open(output_file_name, "w") as f:
            f.write(str(res))
    rt.record("time_records", "time_downstream_tasks", f"all tasks finished")


if __name__ == "__main__":
    fire.Fire(run_downstream_tasks)
