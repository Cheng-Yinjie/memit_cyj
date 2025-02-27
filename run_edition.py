import json
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit import MEMITHyperParams, apply_memit_to_model
from params import EDIT_MODEL_SAVE_PATH, INITIAL_MODEL_SAVE_PATH, MODEL_NAME


# identify running environment
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

# acquire initial model and its weights
model, tok = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=IS_COLAB,
        torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
    ).to("cuda"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token

# save the initial model before editing
model2 = deepcopy(model)
model2.save_pretrained(INITIAL_MODEL_SAVE_PATH)

# content of model edition
request = [
    {
        "prompt": "{} plays the sport of",
        "subject": "LeBron James",
        "target_new": {
            "str": "football"
        }
    },
    {
        "prompt": "{} plays the sport of",
        "subject": "Michael Jordan",
        "target_new": {
            "str": "baseball"
        }
    },
]

# edit model    
hparams = MEMITHyperParams.from_json(f"hparams/MEMIT/{MODEL_NAME}.json")
model_new, tok_new = apply_memit_to_model(
    model=model, 
    tok=tok, 
    requests=request, 
    hparams=hparams,
    copy=True,
    return_orig_weights=False)

# save updated model
model_new.save_pretrained(EDIT_MODEL_SAVE_PATH)