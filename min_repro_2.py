import torch
import os
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.float16, trust_remote_code = True).to('cuda')
# model_id = "microsoft/phi-1_5"
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code = True).to('cuda')
num_devices = int(os.environ.get('WORLD_SIZE', 1))
print(f"num_devices: {num_devices}")

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    output_dir="./outputs",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    deepspeed='config/ds_config.json',
    bf16=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
texts = [
    "modify this code to create a custom toy dataset of two examples texts, write compute loss for",
    "don't change any much except the dataset part and anything else if needed"
]
dataset = Dataset.from_dict({
    "input_ids": [[tokenizer.encode(text)] for text in texts],
    "labels": [[tokenizer.encode(text)] for text in texts]
})

# # Define the compute_loss function
# def compute_loss(model, inputs):
#     labels = inputs["labels"]
#     outputs = model(input_ids=inputs["input_ids"], labels=labels)
#     return outputs.loss


trainer = Trainer(
    model=model,
    args=training_args,
    # compute_loss=compute_loss,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()