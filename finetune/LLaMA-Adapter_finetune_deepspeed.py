import os

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

from peft import AdaptionPromptConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, get_peft_model_state_dict
from utils.prompter import Prompter
from utils.make_dataset import DatasetCreator

# Hyperparameters
model_name = "huggyllama/llama-7b"
# model_name = "facebook/opt-1.3b"
adapter_len = 10 
adapter_layers = 30
train_on_inputs = True
add_eos_token = False
world_size = int(os.environ.get("WORLD_SIZE", 1))
batch_size: int = 8
micro_batch_size: int = 1
num_epochs: int = 3
learning_rate: float = 3e-4
cutoff_len: int = 256
val_set_size: int = 2000
gradient_accumulation_steps = 8
data_path = "alpaca_data_gpt4.json"
output_dir = "./llama-adapter-7b"
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = prepare_model_for_kbit_training(model)
peft_config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4, task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

data = load_dataset("json", data_files=data_path)

val_set_size = 0.1
dataset_creator = DatasetCreator(
    tokenizer=tokenizer,
    prompter=Prompter(template_name="alpaca"),
    cutoff_len=cutoff_len,
    train_on_inputs=train_on_inputs,
    add_eos_token=add_eos_token,
    dataset_size=5000,
    val_set_size=val_set_size,
)

dataset = dataset_creator.create_dataset(data)
train_set = dataset["train"]
val_set = dataset["val"]

model.config.use_cache = False

if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=True,
    optim="adamw_torch",
    evaluation_strategy="epoch",
    logging_steps=1,
    save_steps=3,
    deepspeed="ds_config.json",
    gradient_checkpointing=True,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

if torch.__version__ >= "2":
    model = torch.compile(model)

trainer.train()

model.save_pretrained(output_dir)