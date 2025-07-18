import gc
import os

import fire
import torch
from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq)

from util.edit_inherit import model_load, Prompt4Lora


def run_finetune(
    model_name: str,
    model_folder_path: str,
    data_path: str = "commonsense_170k.json",
    # training parameters
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    save_step: int = 200,
    eval_step: int = 200,
    val_set_size: int = 2000,
    cutoff_len: int = 256
):
    # initiate model, tok and fine tuning prompts
    model, tokenizer = model_load(model_folder_path, model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    gradient_accumulation_steps = batch_size // micro_batch_size
    output_dir = f"{model_folder_path}_fullsize_ft"

    # dataset settings
    data = load_dataset("json", data_files=os.path.join(os.getcwd(), data_path))
    prompt_cfg = Prompt4Lora(tokenizer, cutoff_len, model_name, train_on_inputs=True)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        )
        del train_val
    else:
        train_data = data["train"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        val_data = None

    # training param
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        save_steps=save_step,
        logging_steps=eval_step,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fire.Fire(run_finetune)
