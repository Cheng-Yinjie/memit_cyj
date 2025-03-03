from dataclasses import dataclass
from os.path import join

from lm_eval import tasks, evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
import fnmatch
import lm_eval

from params import MODEL_NAME, PATH_SUFFIX, EDIT_MODEL_SAVE_PATH, LORA_MODEL_SAVE_PATH


@dataclass
class ModelParams:
    model_name: str
    model_path: str
    model: any


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
    model_args = f"pretrained={model_param.model_name},cache_dir={model_param.model_path},tokenizer_name={MODEL_NAME}"
    if eval_set.use_accelerate:
        model_args += f",use_accelerate=True"
    limit = 2000 if "70b" in model_param.model_name or "65b" in model_param.model_name else None
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=model_param.model,
        model_args=model_args,
        tasks=final_tasks,
        num_fewshot=eval_set.num_fewshot,
        batch_size=None,
        device=None,
        limit=limit,
        check_integrity=eval_set.check_integrity
    )
    return results 


if __name__ == "__main__":
    # Set model parameters
    model_folder_path = join(PATH_SUFFIX, LORA_MODEL_SAVE_PATH)
    model = AutoModelForCausalLM.from_pretrained(model_folder_path, use_safetensors=True)
    model = lm_eval.models.huggingface.HFLM(pretrained=model, backend="casual")
    model_params = ModelParams(
        model_name=MODEL_NAME,
        model_path=model_folder_path,
        model=model
        )

    # Set evaluation settings
    eval_set = EvalSets(
        num_fewshot=0,
        use_accelerate=False,
        check_integrity=False
        )

    # Determine downstream tasks, same as DoRA
    task_list = ["boolq", "piqa", "siqa_ca", "hellaswag", "winogrande", "arc_uk", "arc_zh", "openbookqa"]

    eval_zero_shot(model_params, eval_set, task_list)
