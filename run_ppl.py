import fire
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.edit_inherit import model_load, generate_date_string
from util.generate import RecordTimer


def run_ppl_calculation(model_name: str, model_folder: str, adapter_path: str = None):
    # Model name and model_folder will be loaded everytime but not adapter_path
    adapter_path = None if adapter_path == " " else adapter_path
    rt = RecordTimer(model_folder, "time_records", "time_ppl")

    def tokenize_function(example):
        encoded =  tokenizer(
            example["text"],
            padding=True,
            truncation=True, 
            max_length=256,
            return_tensors="pt")
        return encoded
    
    model, tokenizer = model_load(model_folder, model_name, adapter_path)
    model.eval()
    rt.record("model start ppl")

    # Load test dataset
    text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized_dataset = text.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_loader = DataLoader(tokenized_dataset, batch_size=1)
    rt.record("ppl data loaded")

    # Evaluate perplexity
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch["input_ids"].squeeze(0)
            if inputs.size(0) < 2:
                continue

            outputs = model(inputs, labels=inputs)
            loss = outputs.loss.item()
            total_loss += loss * (inputs.size(0) - 1)
            total_tokens += inputs.size(0) - 1
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    rt.record("ppl calculated")

    # Output
    model_type = model_folder if not adapter_path else adapter_path
    with open(f"result_ppl_{model_type}_{generate_date_string()}.txt", "w") as f:
        f.write(str(perplexity))
    print(f"time recorded to: {rt.build_full_path()}")


if __name__ == "__main__":
    fire.Fire(run_ppl_calculation)
