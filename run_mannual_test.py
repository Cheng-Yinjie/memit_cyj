from os import walk
from os.path import join

from params import model_load


def chat_with_model(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


# Param setting
model_name = "gpt2-xl"
model_folder = "new_model_gpt2-xl_250325"
prompt = "The native language of Symeon of Polotsk is"
adapter_folder = None

# Run question
model, tokenizer = model_load(model_folder, model_name, adapter_folder)
model.eval()
res = chat_with_model(model, tokenizer, prompt)
print(res)
