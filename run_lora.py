import fire
from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq)

from peft_local.src.peft import LoraConfig, get_peft_model, TaskType
from util.generate import RecordTimer
from util.edit_inherit import Prompt4Lora, model_load


def run_lora(
    model_folder_path: str,
    model_name: str,
    data_path: str = "zwhe99/commonsense_170k",
    adapter_path: str = None,
    output_dir: str = "./lora-alpaca",
    # training parameters
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    val_set_size: int = 2000,
    cutoff_len: int = 256,
    # lora parameters
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    # llm parameters
    train_on_inputs: bool = True,
):
    # Load initial model and weights
    model, tokenizer = model_load(model_folder_path, model_name, adapter_path)

    rt_model_type = adapter_path if adapter_path else model_folder_path
    rt = RecordTimer(rt_model_type, "time_records", "time_lora")
    prompt_cfg = Prompt4Lora(tokenizer, cutoff_len, model_name, train_on_inputs)

    # Configure LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )

    # Apply LoRA configuration to the model
    peft_model = get_peft_model(model, config)

    # Dataset Preparation``
    data = load_dataset(data_path)
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
        )
    train_data = (
        train_val["train"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        )
    val_data = (
        train_val["test"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        )
    data_collator = DataCollatorForSeq2Seq(
        model=peft_model, 
        tokenizer=tokenizer, 
        return_tensors="pt",
        pad_to_multiple_of=8, 
        padding=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        fp16=True,
        save_safetensors=False,
        per_device_train_batch_size=1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator
    )

    # Train and save
    rt.record(f"model start lora fine-tuning")
    trainer.train()
    rt.record(f"model finish lora fine-tuning")
    peft_model.save_pretrained(output_dir)
    rt.record(f"fine-tuned model saved")
    print(f"time recorded to: {rt.build_full_path()}")


if __name__ == "__main__":
    fire.Fire(run_lora)