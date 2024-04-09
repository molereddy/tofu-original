# conda env create -f config/env_setup.yml
# conda activate tofu
# pip install -r requirements.txt
# pip install flash-attention --no-build-isolation
# module load miniconda/22.11.1-1
# module load cuda/12.2.1
# # required before installing mpi4py
# module load gcc/10.2.0
# module load openmpi/4.1.3
# run via CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=18765 min_repro.py
import torch
import os
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer

prec='fp32'
ds_enabled=True

training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
)

if prec=='fp32':
    dtype=torch.float32
    if ds_enabled: training_args.deepspeed = 'config/ds_config_fp32.json'
    training_args.fp32 = True
elif prec=='bf16':
    dtype=torch.bfloat16
    if ds_enabled: training_args.deepspeed = 'config/ds_config.json'
    training_args.bf16 = True
elif prec=='fp16':
    dtype=torch.bfloat16
    if ds_enabled: training_args.deepspeed = 'config/ds_config.json'
    training_args.bf16 = True
else:
    assert False

# model_id = "NousResearch/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.float16, trust_remote_code = True).to('cuda')
model_id = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code = True).to('cuda')
num_devices = int(os.environ.get('WORLD_SIZE', 1))
print(f"num_devices: {num_devices}")



tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
torch_format_dataset = TextDatasetQA("locuslab/TOFU", tokenizer=tokenizer, model_family='phi', max_length=500, split='full')

model.generation_config.do_sample = True
trainer = CustomTrainer(
    model=model,
    train_dataset=torch_format_dataset,
    args=training_args,
    data_collator=custom_data_collator,
)
trainer.train()