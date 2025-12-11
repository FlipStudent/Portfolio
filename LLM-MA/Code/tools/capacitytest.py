from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
from huggingface_hub import login
from datasets import Dataset, load_dataset, load_from_disk
import os
import torch
import bitsandbytes as bnb
import pandas as pd
from config import FourB_Config, Standard_Config
from accelerate import Accelerator
from finetune import print_trainable_parameters
import numpy as np
import time

def generate_code(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # This has shape [100, 31]
    with torch.no_grad():
      try:
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Explicitly set attention mask
            max_length=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id  # Avoids warning
        )
        return True
      except RuntimeError as e:
        return False




def test_capacity(model, tokenizer, prompt, device):
  model.eval()
  
  batch_size = 50
  last_batch = None
  while True:
    input_prompts = [prompt] * int(batch_size)
    
    start = time.time()
    out = generate_code(model, tokenizer, input_prompts, device)
    end = time.time()
    
    if out:
      print(f"Succeeded batch size ({batch_size}) in {(end - start):.2f}")
      last_batch = batch_size
      batch_size *= 2
    else:
      print(f"Failed at batch size ({batch_size})")
      batch_size *= 0.8
      if batch_size < (last_batch*1.1):
        print(f"Working size: between {last_batch} and {last_batch*1.1}")



if __name__ == "__main__":
  # login() Log in to github here #TODO
  # Removed to prevent committing secret #TODO
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("Running -- Test compact --")
  print("Using device -", device)

  prompt = 'I want to do iterative llm-collaboration using batches. How to still compute using batches while also stopping once the tester llm has passed the code?'

  config = Standard_Config()
  config.use_lora = False
  
  model = config.load_model().to(device)
  tokenizer = config.load_tokenizer()
  
  print("Testing standard")
  test_capacity(  model = model,
                  tokenizer = tokenizer,
                  prompt = prompt,
                  device = device)
  
  config = FourB_Config()
  config.use_lora = True
  config.lora_checkpoint= "/home2/s3978389/Thesis/outputs/CIA_18k/checkpoint-2000/"
  model = config.load_model().to(device)
  tokenizer = config.load_tokenizer()
  print("Testing quantized")
  test_capacity(  model = model,
                  tokenizer = tokenizer,
                  prompt = prompt,
                  device = device)