import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def chat_with_model(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


model_name = "gpt2-xl"
model_path = "/users/acx24yc/py-task2/memit/new_model/pytorch_model.bin"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.load_state_dict(torch.load(model_path))
model.eval()

res = chat_with_model(model, tokenizer, "The definition of curling league is")
print(res)