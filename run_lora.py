from functools import partial
from os.path import join
from typing import Dict, List

from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM

from peft_local.src.peft import LoraConfig, get_peft_model, TaskType
from params import MODEL_NAME, model_load
from util.generate import RecordTimer


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501

MODEL_FOLDER_PATH = "new_model_gpt2-xl_250325"
LORA_MODEL_SAVE_PATH = "new_model_lora_gpt2-xl_250325"
# Params for lora
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 3e-4
EPOCHS = 1
TRAIN_ON_INPUTS = True

# Params for tokenizer
MAX_LEN = 1024
CUTOFF_LEN = 256
VAR_SET_SIZE = 120

# Load initial model and weights
model, tokenizer = model_load(MODEL_FOLDER_PATH, MODEL_NAME)
rt = RecordTimer(MODEL_FOLDER_PATH)


def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in MODEL_NAME:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in MODEL_NAME:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    if not TRAIN_ON_INPUTS:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt


# Configure LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)

# Apply LoRA configuration to the model
peft_model = get_peft_model(model, config)

# Dataset Preparation``
data = load_dataset("zwhe99/commonsense_170k")
train_val = data["train"].train_test_split(
    test_size=VAR_SET_SIZE, shuffle=True, seed=42
    )
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
data_collator = DataCollatorForSeq2Seq(
    model=peft_model, 
    tokenizer=tokenizer, 
    return_tensors="pt",
    pad_to_multiple_of=8, 
    padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=LORA_MODEL_SAVE_PATH,
    fp16=True,
    save_safetensors=False,
    per_device_train_batch_size=1,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
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
rt.record("time_records", "time_lora", f"model start lora fine-tuning")
trainer.train()
rt.record("time_records", "time_lora", f"model finish lora fine-tuning")
peft_model.save_pretrained(LORA_MODEL_SAVE_PATH)
rt.record("time_records", "time_lora", f"fine-tuned model saved")
