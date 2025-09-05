import gc
import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from util.edit_inherit import Prompt4Lora, model_load, find_max_checkpoint_folder
from peft_local.src.peft import (
    LoraConfig,
    DoraConfig,
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_finetune(
        model_folder_path: str,
        model_name: str,
        data_path: str = "commonsense_170k.json",
        # Fine-tuning method selection
        finetune_method: str = "DoRA",
        # training parameters
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = None,
        weight_decay: float = 0.001,
        use_gradient_checkpointing: bool = False,
        val_set_size: int = 2000,
        eval_step: int = 200,
        save_step: int = 200,
        cutoff_len: int = 256,
        # Finetune layer setup
        ft_layers: list = None,
        # PEFT parameters (only used for lora/dora)
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        # DoRA specific parameters
        dora_simple: bool = True,
        wdecompose_target_modules: List[str] = None,
        # Full fine-tuning specific parameters
        warmup_ratio: float = 0.1,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        # llm parameters
        train_on_inputs: bool = True,
        use_int8_training: bool = False,
):
    # Validate fine-tuning method
    finetune_method = finetune_method.lower()
    if finetune_method not in ["lora", "dora", "full"]:
        raise ValueError(f"Invalid finetune_method: {finetune_method}. Choose from 'LoRA', 'DoRA', or 'full'")

    # Set default hyperparameters based on method
    if learning_rate is None:
        learning_rate = 3e-4 if finetune_method in ["lora", "dora"] else 2e-5
    if weight_decay is None:
        weight_decay = 0.0 if finetune_method in ["lora", "dora"] else 0.01

    # Adjust gradient checkpointing default based on method
    use_gradient_checkpointing = True if finetune_method == "full" else use_gradient_checkpointing

    # Load model and tokenizer
    def bf16_supported():
        if not torch.cuda.is_available():
            return False
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            return False

    use_bf16 = bf16_supported()
    use_fp16 = (not use_bf16) and torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model, tokenizer = model_load(
        model_path=model_folder_path,
        model_name=model_name,
        load_dtype=torch_dtype)

    # Configure model based on peft or full-size
    if finetune_method in ["lora", "dora"]:
        # Specify layers to be finetuned
        target_modules_fnl = list()
        if ft_layers:
            if "llama-2" in model_name.lower():
                for layer_idx in ft_layers:
                    for module_name in target_modules:
                        target_modules_fnl.append(f"model.layers.{layer_idx}.self_attn.{module_name}")
        else:
            target_modules_fnl = target_modules

        if use_int8_training:
            model = prepare_model_for_int8_training(
                model, use_gradient_checkpointing=use_gradient_checkpointing
            )
        elif use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if finetune_method == "lora":
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules_fnl,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            peft_config = DoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules_fnl,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                dora_simple=dora_simple,
                Wdecompose_target_modules=wdecompose_target_modules
            )

        model = get_peft_model(model, peft_config)

    else:
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # Set up tokenizer
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompt_cfg = Prompt4Lora(tokenizer, cutoff_len, model_name, train_on_inputs)
    output_dir = f"{model_folder_path}_{finetune_method}" if not ft_layers else f"{model_folder_path}_{finetune_method}_few"
    data = load_dataset("json", data_files=os.path.join(os.getcwd(), data_path))

    # Print trainable parameters and set up datasets
    train_data, val_data = prompt_cfg.dataset_splitter(data, val_set_size)
    if finetune_method in ["lora", "dora"]:
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

        columns_to_remove = ['instruction', 'input', 'output', 'answer']
        train_data = train_data.remove_columns(columns_to_remove)
        if val_data is not None:
            val_data = val_data.remove_columns(columns_to_remove)

    # Set up training arguments
    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "logging_steps": 10,
        "optim": "adamw_torch",
        "eval_strategy": "steps" if val_set_size > 0 else "no",
        "save_strategy": "steps",
        "eval_steps": eval_step if val_set_size > 0 else None,
        "save_steps": save_step,
        "output_dir": output_dir,
        "save_total_limit": 3,
        "load_best_model_at_end": True if val_set_size > 0 else False,
        "report_to": None,
    }

    # Add method-specific training arguments
    if finetune_method in ["lora", "dora"]:
        training_args_dict.update({
            "warmup_steps": 100,
            "ddp_find_unused_parameters": None,
        })
    else:
        training_args_dict.update({
            "warmup_ratio": warmup_ratio,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "ddp_find_unused_parameters": False,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "group_by_length": True,
        })

    training_args = transformers.TrainingArguments(**training_args_dict)

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )

    # Disable caching for training
    model.config.use_cache = False

    # PEFT-specific state dict handling
    if finetune_method in ["lora", "dora"]:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    # Check for existing checkpoints to resume training
    resumed_checkpoint_path = None
    max_checkpoint = find_max_checkpoint_folder(output_dir)
    if max_checkpoint:
        resumed_checkpoint_path = f"{output_dir}/{max_checkpoint}"
        print(f"Resuming from checkpoint: {resumed_checkpoint_path}")

    # Start training
    trainer.train(resume_from_checkpoint=resumed_checkpoint_path)

    # Save the model
    if finetune_method in ["lora", "dora"]:
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    print(f"Fine-tuning method used: {finetune_method}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fire.Fire(run_finetune)