from os import walk
from os.path import join

from util.edit_inherit import model_load


def chat_with_model(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to("cuda")
    output = model.generate(
        input_ids=input_ids, 
        max_length=max_length, 
        pad_token_id=tokenizer.eos_token_id)
    # response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Param setting
model_name = "gpt2-xl"
model_folder = "new_model_gpt2-xl_250326"
prompt = "Choose the right answer among all options and your answers should be answer1 or answer2 or answer3. For example, if the question is 1+1=?, answer1: 2, answer2:1, you should answer answer1. Now the question is: to calculate complex algebra problem, you would need, answer1: calculator, answer2: clothes"
adapter_folder = None

# Run question
model, tokenizer = model_load(model_folder, model_name, adapter_folder)
model.eval()
res = chat_with_model(model, tokenizer, prompt)
print(res)
