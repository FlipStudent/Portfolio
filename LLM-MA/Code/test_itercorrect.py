from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
from huggingface_hub import login
from datasets import Dataset, load_dataset, load_from_disk
import os
import torch
import bitsandbytes as bnb
import pandas as pd
from config import FourB_Config, Standard_Config, generate_response
from accelerate import Accelerator
from finetune import print_trainable_parameters
from team import *

from team.iterative_generation import iter_response
from tool_code_evaluation import evaluate_code, extract_code


def test(output_name, config, test_type, prompt_mode, dataset_name, verbose=False, pass_n=1):  

  
  # Distribute
  accelerator = Accelerator(
      mixed_precision="bf16",  # Options: "no", "fp16", "bf16"
      cpu=False,               # Set to True if you want CPU-only
      device_placement=True,   # Automatically place tensors on the right device
      split_batches=True       # Split batches across devices
  )
  
  try:
    model = config.load_model(device_map = {"": accelerator.process_index})
    print("Loaded model")
    #print_trainable_parameters(model)
    model.eval() # Does NOT decrease the number of trainable parameters
    print_trainable_parameters(model)
  except Exception as e:
    print(f"[Rank {accelerator.local_process_index}] Model load failed:", e)
    raise
    

  model = accelerator.prepare(model)
  print("Distributed model")
  tokenizer = config.load_tokenizer()
  
  # Store the results
  results = []
  
  mbpp = load_from_disk("./data/test/mbpp_io_augmented.hf")
  print("Loaded dataset")
  
  if pass_n > 1:
    temp = 0.7
  else:
    temp = 0.0

  for idx, item in enumerate(mbpp['test']):
    if idx < 100:
      continue
    print(f"{idx}/500 - {item['task_id']}")
    task_id = item["task_id"]
    question = item["text"]
    ex_in_out_pair = item["in_out_pairs"][0]
    ex_input = ex_in_out_pair['input']
    ex_output = item['in_out_pairs'][-1]['output']
    has_passed = False
    
        
    prompt = f"{question}\n"
    
    if test_type == "fewsh":
      prompt += f"Here is an example of the input variables: {ex_input}\nHere is an example of the output shape: {ex_output}\n"
    if prompt_mode == "single":
      prompt += f"Remember, you only need to provide the code in Python. Do NOT explain the code or give examples and do NOT give test cases.\n# Write a solution:\ndef solution("
    
    iteration = 1
    save_iters = []
    
    for attempt_nr in range(pass_n):
      if prompt_mode == "single":
        generated_code = generate_response(model, tokenizer, prompt, temp=temp)
      elif prompt_mode == "iter":
        generated_code, iteration = iter_response( model = model,
                                                   tokenizer = tokenizer,
                                                   user_requirement = prompt,
                                                   max_iters = 10,
                                                   code_temp = temp)
                                                   
      print(f"attempt {attempt_nr+1} - {iteration} iterations")     
                                     
      save_iters.append(iteration)
                                         
      cleaned_code = extract_code(generated_code)
                                         
      has_passed = evaluate_code( aug_mbpp =       mbpp, 
                                  question_index=  idx, 
                                  generated_code=  cleaned_code)
      
      #print(f"Haspassed: {has_passed}")
      
      if has_passed:
        break
    
    average_test_reports = sum(save_iters)/len(save_iters)
    
    print(f"\nTask: {task_id}")
    print(f"Passed: {has_passed}")
    print(f"Attempts: {attempt_nr+1}")
    print(f"Fixes this attempt: {iteration}\n")
    print(f"Average n of iterations in this attempt: {average_test_reports}\n")
    
    
    results.append({
      "task_id": task_id,
      "result": generated_code,
      "has_passed": has_passed,
      "attempt": attempt_nr+1,
      "corrections": average_test_reports
          #   "starter_code": starter_code,
          #   "generated_solution": generated_code
    })
    
    if (idx+1) % 50 == 0:  
      df = pd.DataFrame(results)
      hf_dataset = Dataset.from_pandas(df)
      hf_dataset.save_to_disk(str(int(idx/50) ) + output_name)
    
    #if verbose:
    #  print(f"{task_id} - P?-{has_passed} : {attempt_nr+1}")
        
    #if idx == 20:
    #  break
  
  ## Save to disk
  df = pd.DataFrame(results)
  hf_dataset = Dataset.from_pandas(df)
  hf_dataset.save_to_disk(output_name)
  return

if __name__ == "__main__":
  # login() Log in to github here #TODO
  # Removed to prevent committing secret #TODO
  #device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}
  param_dtype = torch.bfloat16

  
  print("Running -- Test compact --")
  #print("Using device -", device_map)
  
  output_dir = "generated_solutions"
  #output_name = "base_model_full_mbpp"
  #output_name = "fourbit_no_finetune_full_mbpp"
  
  prompt_mode = "iter"
  test_type = "fewsh"
  model_name = "fourbit_ft"
  dataset_name = "test_mbpp"
  
  assert prompt_mode in ["single", "iter"]
  assert test_type in ["1sh", "fewsh"]
  assert model_name in ["base", "fourbit_noft", "fourbit_ft"]
  assert dataset_name in ["test_mbpp"]
  
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
  
  output_name = f"{prompt_mode}_{dataset_name}_{model_name}_{test_type}"
  print(f"Outputting: {output_name}")
  
  test( output_name = output_name,
        config = config,
        test_type = test_type,
        prompt_mode = prompt_mode,
        dataset_name = dataset_name,
        verbose = True,
        pass_n = 10)