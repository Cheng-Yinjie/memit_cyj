from os import walk

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from params import MODEL_NAME, PATH_SUFFIX, EDIT_MODEL_SAVE_PATH, get_whole_model_dict


def chat_with_model(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_folder_path = join(PATH_SUFFIX, EDIT_MODEL_SAVE_PATH)
model.load_state_dict(get_whole_model_dict(model_folder_path), strict=False)
model.eval()

res = chat_with_model(model, tokenizer, "Michael Jordan plays the sport of")
print(res)
