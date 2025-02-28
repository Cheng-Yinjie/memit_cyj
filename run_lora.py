from functools import partial
from os.path import join
from typing import Dict, List
import copy

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM
import torch

from params import LORA_MODEL_SAVE_PATH, PATH_SUFFIX, MODEL_NAME, EDIT_MODEL_SAVE_PATH, get_whole_model_dict


def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"] 
    # check if both exists, else raise error   
    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")
    if not response:
        raise ValueError(f"Expected a response in: {rec}")
    rec["prompt"] = prompt_template.format(instruction=instruction)
    rec["answer"] = answer_template.format(response=response)
    rec["text"] = rec["prompt"] + rec["answer"]
    return rec

def _preprocess_batch(batch: Dict[str, List]):  
    model_inputs = tokenizer(batch["text"], max_length=MAX_LEN, truncation=True, padding='max_length')    
    model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs


# Params for lora
MAX_LEN = 256
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05
LEARNING_RATE = 0.0001
EPOCHS = 3

# Load initial model and weights
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
edit_model_path = join(PATH_SUFFIX, EDIT_MODEL_SAVE_PATH)
model.load_state_dict(get_whole_model_dict(edit_model_path), strict=False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

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

# Dataset Preparation
dataset = load_dataset("MBZUAI/LaMini-instruction", split='train')
small_dataset = dataset.select([i for i in range(200)])

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction}\n Response:"""
answer_template = """{response}"""

small_dataset = small_dataset.map(_add_text)
_preprocessing_function = partial(_preprocess_batch)

encoded_small_dataset = small_dataset.map(
    _preprocessing_function,
    batched=True,
    remove_columns=["instruction", "response", "prompt", "answer"],
)
processed_dataset = encoded_small_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LEN)
split_dataset = processed_dataset.train_test_split(test_size=14, seed=0)
data_collator = DataCollatorForSeq2Seq(
    model=peft_model, tokenizer=tokenizer, 
    max_length=MAX_LEN, 
    pad_to_multiple_of=8, 
    padding='max_length')

# Training arguments
training_args = TrainingArguments(
    output_dir=LORA_MODEL_SAVE_PATH,
    overwrite_output_dir=True,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)
 
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator
)
 
trainer.train()

# Save model
peft_model.save_pretrained(LORA_MODEL_SAVE_PATH)
