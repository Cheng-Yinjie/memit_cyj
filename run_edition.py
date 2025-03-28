import json
from copy import deepcopy

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.generate import GenCounterFactRequests, RecordTimer
from memit import MEMITHyperParams, apply_memit_to_model
from params import generate_date_string


def run_edition(model_name: str, ini_model_save_path: str, edited_model_save_path: str):
    ini_model_save_path = f"{ini_model_save_path}_{generate_date_string()}"
    edited_model_save_path = f"{edited_model_save_path}_{generate_date_string()}"
    rt = RecordTimer(ini_model_save_path)

    # Identify running environment
    IS_COLAB = False
    ALL_DEPS = False
    try:
        import google.colab, torch, os

        IS_COLAB = True
        os.chdir("/content/memit")
        if not torch.cuda.is_available():
            raise Exception("Change runtime type to include a GPU.")
    except ModuleNotFoundError as _:
        pass

    # Acquire initial model and its weights
    model, tok = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=IS_COLAB,
            torch_dtype=(torch.float16 if "20b" in model_name else None),
        ).to("cuda"),
        AutoTokenizer.from_pretrained(model_name),
    )
    tok.pad_token = tok.eos_token

    # Save the initial model before editing
    model2 = deepcopy(model)
    model2.save_pretrained(ini_model_save_path)
    rt.record("time_records", "time_edit", f"Initial model saved")

    # content of model edition
    gen_cf_reqs = GenCounterFactRequests(1000, "edit")
    requests = gen_cf_reqs.proc()
    rt.record("time_records", "time_edit", f"Edit data prepared")

    # Edit model
    hf_model_name = model_name.replace("/", "_")
    hparams = MEMITHyperParams.from_json(f"hparams/MEMIT/{hf_model_name}.json")
    model_new, _ = apply_memit_to_model(
        model=model, 
        tok=tok, 
        requests=requests, 
        hparams=hparams,
        copy=True,
        return_orig_weights=False)
    rt.record("time_records", "time_edit", f"Edition finished")

    # Save updated model
    model_new.save_pretrained(edited_model_save_path)
    rt.record("time_records", "time_edit", f"Edited model saved")


if __name__ == "__main__":
    fire.Fire(run_edition)