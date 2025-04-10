import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from peft_local.src.peft import (
    LoraConfig, 
    DoraConfig, 
    prepare_model_for_int8_training, 
    get_peft_model, 
    get_peft_model_state_dict)
from util.generate import RecordTimer
from util.edit_inherit import Prompt4Lora, model_load


def run_finetune(
    model_folder_path: str, 
    model_name: str, 
    data_path: str = "zwhe99/commonsense_170k",
    adapter_name: str = "lora",
    adapter_path: str = " ",
    output_dir: str = "./lora-alpaca",
    # training parameters
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    use_gradient_checkpointing: bool = False,
    val_set_size: int = 2000,
    eval_step: int = 200,
    save_step: int = 200,
    cutoff_len: int = 256,
    # lora parameters
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
    # dora parameters
    dora_simple: bool = True,
    Wdecompose_target_modules: List[str] = None,
    # llm parameters
    train_on_inputs: bool = True,
    ):
    # Set up model and tokenizer
    model, tokenizer = model_load(model_folder_path, model_name)
    model = prepare_model_for_int8_training(
         model, use_gradient_checkpointing=use_gradient_checkpointing)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    # Initiate params
    gradient_accumulation_steps = batch_size // micro_batch_size
    rt_model_type = adapter_path if adapter_path else model_folder_path
    rt = RecordTimer(rt_model_type, "time_records", f"time_{adapter_name}")
    prompt_cfg = Prompt4Lora(tokenizer, cutoff_len, model_name, train_on_inputs)

    if adapter_name == "lora":
        config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
    else:
        config = DoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules
            )

    model = get_peft_model(model, config)
    data = load_dataset(data_path)
    model.print_trainable_parameters()

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
    else:
        train_data = data["train"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=None,
            report_to=None
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    rt.record("model start dora fine-tuning")
    trainer.train()
    rt.record("model finished dora fine-tuning")
    model.save_pretrained(output_dir)
    rt.record("model saved")
    print(f"time recorded to: {rt.build_full_path()}")


if __name__ == "__main__":
    fire.Fire(run_finetune)