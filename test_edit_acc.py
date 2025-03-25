import numpy as np
import torch
from tqdm import tqdm

from params import model_load
from util.generate import GenCounterFactRequests


def test_single_edit(model, tok, prompts, target, delimiter="Ġ"):
    res = list()
    for prompt in tqdm(prompts):
        prompt_tok = tok(
            prompt,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**prompt_tok)
            if type(outputs) is torch.Tensor:
                logits = outputs
            else:
                logits = outputs.logits
            if tok.padding_side == 'left':
                ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
            else:
                last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
                to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
                gathered = torch.gather(logits, 1, to_gather).squeeze(1)
                ans = torch.argmax(gathered, dim=1)
            
            ans = ans.squeeze().detach().cpu().numpy().tolist()
            ans = tok.convert_ids_to_tokens(ans)
            ans = ans.replace(delimiter, "")
        res.append(ans)
    return np.mean(np.equal(res, target))
    

def gen_acc_test_prompts(dataset: list):
    prompts, targets = list(), list()
    for item in dataset:
        prompt = item['prompt'].format(item['subject'])
        target_value = item['target_new']['str']
        prompts.append(prompt)
        targets.append(target_value)
    return prompts, targets


if __name__ == "__main__":
    dataset = GenCounterFactRequests().proc()
    prompts, targets = gen_acc_test_prompts(dataset)
    model, tok = model_load("new_model_gpt2-xl_250316", "gpt2-xl")
    print(test_single_edit(model, tok, prompts, targets))
