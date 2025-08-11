import gc
import json
import logging
import math
import os
import shutil
from pathlib import Path
from tqdm import tqdm

import fire
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

from util.edit_inherit import model_load


def run_finetune(
        # model settings
        model_folder_path: str,
        model_name: str,
        # training parameters
        data_path: str = "commonsense_170k.json",
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 5e-6,  # conservative LR for full finetuning
        cutoff_len: int = 256,
        warmup_ratio: float = 0.01,  # warmup fraction of total steps
        # basic settings
        prefetch: bool = False,  # set True if you want dataset fully loaded in memory
        log_step: int = 50,
        save_step: int = 5000,
):
    def monitor_checkpoints(folder_path, pattern):
        """Find the latest checkpoint and total checkpoints nubmers"""
        checkpoints = [f for f in os.listdir(folder_path) if "checkpoint-" in f]
        if not checkpoints:
            return None, 0
        steps = [int(f.split('-')[1]) for f in checkpoints]
        final_step = max(steps) if pattern == "max" else min(steps)
        return f"checkpoint-{final_step}", len(steps)

    def bf16_supported():
        if not torch.cuda.is_available():
            return False
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            return False

    # initiations
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    output_dir = f"{model_folder_path}_fullft"
    os.makedirs(output_dir, exist_ok=True)

    # basic settings
    use_bf16 = bf16_supported()
    use_fp16 = (not use_bf16) and torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, use_bf16={use_bf16}, use_fp16={use_fp16}")

    # set up model and tokenizer
    latest_checkpoint, _ = monitor_checkpoints(output_dir, "max")
    if latest_checkpoint:
        logger.info(f"Found latest checkpoint: {latest_checkpoint} at step {latest_checkpoint}")
        model, tokenizer = model_load(
            model_path=Path(output_dir) / latest_checkpoint,
            model_name=model_name,
            load_dtype=torch_dtype
        )
        global_step = int(latest_checkpoint.split("-")[1])
    else:
        logger.info(f"No checkpoint found, loading model {model_name} from {model_folder_path}")
        model, tokenizer = model_load(
            model_path=model_folder_path,
            model_name=model_name,
            load_dtype=torch_dtype
        )
        global_step = 0
    logger.info(f"Training step starts at {global_step}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "llama-2" in model.config._name_or_path.lower():
        tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # prepare dataset
    class JsonInstructionDataset(Dataset):
        def __init__(self, path, tokenizer, max_length=1024, preload=False):
            self.tokenizer = tokenizer
            self.max_length = max_length
            if preload:
                with open(path, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)

        def __len__(self):
            return len(self.samples)

        def _format_prompt(self, instruction, inp, output_text):
            if inp and inp.strip():
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output_text}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            return prompt

        def __getitem__(self, idx):
            ex = self.samples[idx]
            instruction = ex.get("instruction", "")
            input_text = ex.get("input", "")
            output_text = ex.get("output", "")

            prompt = self._format_prompt(instruction, input_text, output_text)

            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    dataset = JsonInstructionDataset(data_path, tokenizer, cutoff_len, preload=prefetch)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    total_steps = num_epochs * math.ceil(len(dataset) / batch_size)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    # prepare optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # main training loop
    model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            if use_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            lr_scheduler.step()
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            # compute gradient norm for logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().float().norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)

            # logging
            if global_step % log_step == 0 or global_step < 20:
                logger.info(
                    f"Step {global_step}: Loss={loss_val:.4f}, Grad_norm={total_norm:.2e}, LR={lr_scheduler.get_last_lr()[0]:.2e}")

            # save checkpoints periodically
            if (global_step > 0) and (global_step % save_step == 0):
                ckpt_dir = Path(output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving checkpoint to {ckpt_dir}")
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                # control checkpoints number
                earlest_ckpt, ckpt_num = monitor_checkpoints(output_dir, "min")
                if ckpt_num > 3:
                    shutil.rmtree(Path(output_dir) / earlest_ckpt)

            global_step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "grad_norm": f"{total_norm:.2e}",
                              "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

    # save model and tok
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fire.Fire(run_finetune)
