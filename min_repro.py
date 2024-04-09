import torch
import os
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer

prec='fp16'

# model_id = "NousResearch/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.float16, trust_remote_code = True).to('cuda')
model_id = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if prec=='fp16' else torch.bfloat16, 
                                             trust_remote_code = True).to('cuda')
num_devices = int(os.environ.get('WORLD_SIZE', 1))
print(f"num_devices: {num_devices}")

training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    deepspeed='config/ds_config.json' if prec!='fp16' else 'config/ds_config_fp16.json',
    bf16=(prec!='fp16'),
    fp16=(prec=='fp16')
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
torch_format_dataset = TextDatasetQA("locuslab/TOFU", tokenizer=tokenizer, model_family = 'phi', max_length=500, split='full')

model.generation_config.do_sample = True
trainer = CustomTrainer(
    model=model,
    train_dataset=torch_format_dataset,
    args=training_args,
    data_collator=custom_data_collator,
)
trainer.train()