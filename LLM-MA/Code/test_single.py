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


def generate_code(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # This has shape [100, 31]
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Explicitly set attention mask
        max_length=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id  # Avoids warning
    )
    print(f"output shape: {output.shape}")
    if output.ndim > 1:
      return tokenizer.batch_decode(output, skip_special_tokens=True)
    else:
      return tokenizer.decode(output[0], skip_special_tokens=True)




def test(output_name, config, test_type, verbose=False):  
  model = config.load_model()
  print("Loaded model")
  #print_trainable_parameters(model)
  model.eval() # Does NOT decrease the number of trainable parameters
  print_trainable_parameters(model)
  
  # Distribute
  accelerator = Accelerator(
      mixed_precision="bf16",  # Options: "no", "fp16", "bf16"
      cpu=False,               # Set to True if you want CPU-only
      device_placement=True,   # Automatically place tensors on the right device
      split_batches=True       # Split batches across devices
  )
  distribute_model = accelerator.prepare(model)
  print("Distributed model")
  tokenizer = config.load_tokenizer()
  
  # Store the results
  results = []
  
  mbpp = load_from_disk("./data/test/mbpp_io_augmented.hf")
  print("Loaded dataset")
  
  
  batch_size = 500

  if batch_size > 1:
    questions = mbpp['test']['text']
    #print(f"Questions size: {questions.shape)}")
    outputs = generate_code(model, tokenizer, questions)
    #print(f"Batched outputs shape: {len(outputs)}")
    print(f"Outputs: \n{len(outputs)}")
  else:
    for idx, item in enumerate(mbpp['test']):
      print(f"{idx}/500 - {item['task_id']}")
      task_id = item["task_id"]
      question = item["text"]
      #ex_in_out_pair = item["in_out_pairs"][0]
      ex_input = item["in_out_pairs"][0]['input']
      ex_output = item["in_out_pairs"][-1]['output']
      
      #prompt = f"I want you to act as a requirement analyst on our development team. Given a user requirement, your task is to analyze, decompose, and develop a high-level and concise plan to guide our developer in writing programs. The plan should include the following information:1. Decompose the requirement into several easy-to-solve subproblems that can be more easily implemented by the developer. 2. Develop a high-level plan that outlines the major steps of the program. This is the requirement:"
      prompt = f"{question}\n"
          
      if test_type == "1-shot":
        prompt += f"Here is an example of input: {ex_input}\nHere is an example of output: {ex_output}\n"
      prompt += f"Remember, you only need to provide the python code. Do not explain your code. Do not give test cases."
        
    generated_code = generate_code(model, tokenizer, prompt)
        
    results.append({
      "task_id": task_id,
      "result": generated_code,
          #   "starter_code": starter_code,
          #   "generated_solution": generated_code
    })
    
    if verbose:
      print(f"{task_id} - {question}\n - {generated_code}")
    
    return
  ## Save to disk
  df = pd.DataFrame(results)
  hf_dataset = Dataset.from_pandas(df)
  hf_dataset.save_to_disk(output_name)
  return

if __name__ == "__main__":
  # login() Log in to github here #TODO
  # Removed to prevent committing secret #TODO
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  print("Running -- Test compact --")
  print("Using device -", device)
  
  output_dir = "generated_solutions"
  #output_name = "base_model_full_mbpp"
  #output_name = "fourbit_no_finetune_full_mbpp"
  
  test_type = "0-shot"
  model_name = "base"
  
  assert test_type in ["0-shot", "1-shot"]
  assert model_name in ["base", "fourbit_noft", "fourbit_ft"]
  
  few_shot = True
  
  os.makedirs(output_dir, exist_ok=True)
  
  # base no finetune
  if model_name == "base":
    config = Standard_Config()
    config.use_lora = False
    
  # 4b no finetune
  elif model_name == "fourbit_noft":
    config = FourB_Config()
    config.use_lora = False

  # 4b finetuned
  elif model_name == "fourbit_ft":
    config = FourB_Config()
    config.use_lora = True
    config.lora_checkpoint= "/home2/s3978389/Thesis/outputs/CIA_18k/checkpoint-2000/"
    
  else:
    raise NotImplementedError
  
  output_name = f"{model_name}_{test_type}"
  print(f"Outputting: {output_name}")
  
  test( output_name = output_name,
        config = config,
        test_type = test_type,
        verbose = True)