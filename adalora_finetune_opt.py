import deepspeed
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training, AdaLoraConfig, AdaLoraModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from accelerate import Accelerator
def train(
        micro_batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        init_r=12,
        target_r=8,
        beta1=0.85,
        beta2=0.85,
        tinit=200,
        tfinal=1000,
        deltaT=10,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules: list = ["fc1", "fc2"], # "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
):

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    model = prepare_model_for_int8_training(model)

    config = AdaLoraConfig(
                peft_type="ADALORA", 
                task_type="CAUSAL_LM", 
                init_r=init_r,
                target_r=target_r,
                beta1=beta1,
                beta2=beta2,
                tinit=tinit,
                tfinal=tfinal,
                deltaT=deltaT,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
            )

    model.enable_input_require_grads() # fixes issue of RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        examples = [" ".join(text) for text in examples["answers.text"]]
        return tokenizer(
            examples,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors=None
        )

    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    eli5 = eli5.flatten()
    eli5 = eli5.train_test_split(test_size=0.9)
    eli5 = eli5.map(tokenize_function, batched=True, num_proc=1, remove_columns=eli5["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        eli5["train"], shuffle=True, collate_fn=data_collator, batch_size=micro_batch_size
    )

    eval_dataloader = DataLoader(
        eli5["test"], collate_fn=data_collator, batch_size=micro_batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    train()
