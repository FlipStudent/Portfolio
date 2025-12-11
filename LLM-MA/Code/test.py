from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from datasets import Dataset, load_dataset, load_from_disk
from accelerate import Accelerator
from huggingface_hub import login
from peft import PeftModel
import bitsandbytes as bnb
import pandas as pd
import numpy as np
import argparse
import random
import torch
import math
import time
import os
import re

torch.manual_seed(333)
random.seed(333)
np.random.seed(333)

# FREE UP SPACE
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Custom modules:
from config import Qwen, Deepseek, Mistral, Yi, print_trainable_parameters, calc_trainable_parameters, get_model_size_gb, get_true_model_size_gb
from tools.tool_code_evaluation import Evaluator
#from team.iterative_generation import iter_response
#from team import *
from team import iterative_generation, role_descriptions, tester, coder, analyst

#from team import iterative_generation


def combine_batch_strings(lists):
  return ["".join(items) for items in zip(*lists)]

def make_batch_ranges(N, S):
    return [(i, min(i+S, N)) for i in range(0, N, S)]

def prepare_accelerator():
  return Accelerator( mixed_precision="bf16",  # Options: "no", "fp16", "bf16"
                      cpu=False,               # Set to True if you want CPU-only
                      device_placement=True,   # Automatically place tensors on the right device
                      split_batches=True       # Split batches across devices
                     )


  
  
def construct_task_prompts(problem_entries):
    batch_size = len(problem_entries['text'])
    print(f"Batch size in construct task: {batch_size}")
    #intro_questions = ["This is the task:\n"]*batch_size
    questions = problem_entries['text']
    intro_inputs = ["\n\nExample input: "]*batch_size
    ex_inputs = [in_out_pairs[0]['input'] for in_out_pairs in problem_entries['in_out_pairs']] # Input from first in-out-pair
    intro_outputs = ["\nExample output: "]*batch_size
    #ex_outputs = [in_out_pairs[-1]['output'] for in_out_pairs in items['in_out_pairs']] # Output from last in-out pair
    ex_outputs = [in_out_pairs[0]['output'] for in_out_pairs in problem_entries['in_out_pairs']] # Output from first in-out pair
    outro_outputs = ["\n"]*batch_size

    all_texts = [questions, intro_inputs, ex_inputs, intro_outputs, ex_outputs, outro_outputs]

    prompts = combine_batch_strings(all_texts)
    return prompts
    

def construct_report_message(clean_code, test_report):
  intro_tester = "\nThe report from the tester is as following:\n"
  intro_codes = "\nThis is the previous code which you need to improve:\n"
  return intro_tester + test_report + intro_codes + clean_code
  #return combine_batch_strings([intro_tester, test_reports, intro_codes, clean_codes])
  
  
def check_passed(test_report):
  #print(f"Checking test report")
  if "Code Test Passed".lower() in test_report.lower():
    return True
  if bool(re.search(r'code test passed', test_report, re.IGNORECASE)):
    return True
  if bool(re.search(r'code\s*test[\s:]*[*_]*passed[*_]*', test_report, re.IGNORECASE)):
    return True
  return False

def pass_at_k(c, n, k):
    c = c[c > 0]  # eligible problems
    comb_lookup = np.array([math.comb(n - ci, k) if n - ci >= k else 0 for ci in range(n+1)], dtype=np.float64)
    return np.mean(1 - comb_lookup[c] / math.comb(n, k))

def print_progress(start_time, pass_nr, correct_indices, incorrect_indices, N_samples_in_partition):
  cur_time = time.time()
  busy_time = (cur_time-start_time)
  n_samples_done = len(correct_indices)+len(incorrect_indices)
  perc_done = round((n_samples_done/N_samples_in_partition)*100, 2)
  print(f"Pass {pass_nr} | {n_samples_done} / {N_samples_in_partition} ({perc_done}%) | {(n_samples_done/busy_time):.2f} questions / sec")

def test(output_name, config, prompt_mode, dataset_name, verbose=False, pass_n=1):  
  # Prepare accelerator and tester tools
  accelerator = prepare_accelerator()
  evaluator = Evaluator()
  
  # Load model
  config.model = accelerator.prepare(config.model)
  print("Distributed model")
  
  # Set generation 'randomness'
  temp = 0.0
  if pass_n > 1: temp = 0.7 
  #if prompt_mode == "iter": temp = 0.2 
  
  
  # Load datasest
  mbpp_partition = load_from_disk("./data/test/mbpp_io_augmented.hf")['test']#.select(list(range(0,100)))
  
  #example_prompts = construct_task_prompts(mbpp_partition.select([5]))
  #print(f"Test example prompt:\n {example_prompts[0]}")
  #example_outputs = config.prompt(example_prompts, temp)
  #print(f"Test example output:\n{example_outputs[0]}")
  
  print("Loading dataset DONE")
  #print(mbpp_partition)
  
  # Prepare data handling
  N_samples_in_partition = len(mbpp_partition)
  batch_size = config.max_batch_size
  if prompt_mode == "iter":
    batch_size = int(batch_size*0.6)
  
  batch_ranges = make_batch_ranges(N_samples_in_partition, config.max_batch_size)
  #results = []
  
  
  results = [False]*N_samples_in_partition
  
  
  collect_pass_results = [] # Records the pass@N
  
  print_example_prompt = True

  #if prompt_mode == "single":
  #  for pass_nr in range(pass_n):
  #    batch_size = min(config.max_batch_size, len(todo_indices))  # Ensure you use the maximum batch size, without prompting solved questions
  #    batch_ranges = make_batch_ranges(N_samples_in_partition, batch_size)
  #    for batch_range in batch_ranges:
  #      items = mbpp_partition.select(batch_indices)  # Partition the questions
  #      task_prompts = construct_task_prompts(items, batch_size)
      
  #else:
    # Go over the full dataset:
  start_time = time.time()
  for pass_nr in range(pass_n):
    incorrect_output_errors = 0
    compiling_errors = 0
    
    todo_indices = list(range(0, N_samples_in_partition)) # Keeps track of which questions are still unsolved
    n_iteration_tries_of_index = [0] * N_samples_in_partition # Keeps track of how many times a question has been attempted
    report_messages = [""] * N_samples_in_partition
    
    correct_indices = []    # If the code is correct, append the question index here
    incorrect_indices = []  # If the code is incorrect *max iter* times, append the index here
                            # In either case, pop the index from the todo list
                            

    
    while True:
      print_progress( start_time,
                      pass_nr,
                      correct_indices,
                      incorrect_indices,
                      N_samples_in_partition
                      )
      
      batch_size = min(config.max_batch_size, len(todo_indices))  # Ensure you use the maximum batch size, without prompting solved questions
      print(f"#Batch size: {batch_size}")
    
      batch_indices = todo_indices[0:batch_size]
      #print(f"{batch_indices}\n")
      
      items = mbpp_partition.select(batch_indices)  # Partition the questions
      task_prompts = construct_task_prompts(items)
      
      input_messages = [report_messages[batch_idx] for batch_idx in batch_indices]
      
      #print(f"#Input messages:#\n{input_messages}")

      if prompt_mode == "single":
        
        #input_messages = [message + 'Remember, only provide code: do NOT explain the code and do NOT give test cases.' for message in input_messages]
        just_code_string = ['\nRemember, only provide code: do NOT explain the code and do NOT give test cases.']*batch_size
        task_prompts = combine_batch_strings([task_prompts, just_code_string])        
          
        gen_codes = config.prompt(task_prompts, temp)
        if print_example_prompt:
          print(f"Example prompt:\n{task_prompts[0]}")
          print(f"Example output: \n{gen_codes[0]}")
          print_example_prompt = False
        #print(f"\nRaw code:\n{gen_codes[0]}")
        for i, gen_code in enumerate( gen_codes):
          #print(f"################")
          dataset_index = batch_indices[i]
          cleaned_code = evaluator.extract_code(gen_code)
          is_correct, eval_msg = evaluator.evaluate_code(   aug_mbpp_partition =  mbpp_partition, 
                                                            question_index     =  dataset_index, 
                                                            generated_code     =  cleaned_code)
          #print(task_prompts[i] + "\n##")
          
          #print(f"({is_correct}) code:\n{cleaned_code}")
          if is_correct:
            #print(f"Has passed: {dataset_index}")
            correct_indices.append(dataset_index)
          else:
            incorrect_indices.append(dataset_index)
          if eval_msg == "compiling":
            compiling_errors += 1
          if eval_msg == "output":
            incorrect_output_errors += 1
          todo_indices.remove(dataset_index)
      # Prompt coder and tester 1x
      # Get batch of: [code, reports]*bs
      elif prompt_mode =="iter":
        gen_codes, test_reports = iterative_generation.iter_response(config = config,
                                                       user_requirements = task_prompts,
                                                       do_analysis = False,
                                                       code_temp = temp,
                                                       report_messages=input_messages)
        
        n_todos_before = len(todo_indices)
        
        # Check and remove questions
        for i, code in enumerate(gen_codes):
          dataset_index = batch_indices[i]
          has_passed = check_passed(test_reports[i])
          cleaned_code = evaluator.extract_code(code)
          if has_passed or n_iteration_tries_of_index[dataset_index] == 5:
            is_correct, eval_msg = evaluator.evaluate_code( aug_mbpp_partition =  mbpp_partition, 
                                                            question_index     =  dataset_index, 
                                                            generated_code     =  cleaned_code)
            print(is_correct, eval_msg)
            todo_indices.remove(dataset_index)
            if is_correct:
              print(f"Has passed: {dataset_index}")
              correct_indices.append(dataset_index)
            else:
              incorrect_indices.append(dataset_index)
              if eval_msg == "compiling":
                compiling_errors += 1
              if eval_msg == "output":
                incorrect_output_errors += 1
              
          else:
            n_iteration_tries_of_index[dataset_index] += 1
            message = construct_report_message(cleaned_code, test_reports[i])
            report_messages[dataset_index] = message
      
        n_todos_after = len(todo_indices)
        print(f"#Number of questions passed: {n_todos_before-n_todos_after} ({N_samples_in_partition-len(todo_indices)}/{N_samples_in_partition} done)#")
      
      print(f"Correct: {len(correct_indices)}/{len(correct_indices)+len(incorrect_indices)}")
      total_errors = compiling_errors + incorrect_output_errors
      print(f"Erors: {compiling_errors} compiling ({((compiling_errors/total_errors)*100):.2f}) - {incorrect_output_errors} output ({((incorrect_output_errors/total_errors)*100):.2f})")
      
      if len(todo_indices) == 0:
        break # Finished all questions
        
    print(f"\n#! Correct !#\n{correct_indices}")
    print(f"\n#! Incorrect !#\n{incorrect_indices}")
    
    evaluations = [None]*N_samples_in_partition
    question_ids = [None]*N_samples_in_partition
    for dataset_index in list(range(0, N_samples_in_partition)):
      question_ids[dataset_index] = mbpp_partition[dataset_index]['task_id']
      if dataset_index in correct_indices:
        evaluations[dataset_index] = True
      elif dataset_index in incorrect_indices:
        evaluations[dataset_index] = False
    
    pass_results = {'pass':pass_nr, 'question_ids':question_ids, 'evaluations':evaluations, 'attempts':n_iteration_tries_of_index}
    pass_corr_perc = f"{(np.mean(np.array(evaluations))*100):.2f}"
    pass_avg_corr = f"{(np.mean(np.array(n_iteration_tries_of_index))*100):.2f}"
    collect_pass_results.append(pass_results)
    print(f"\n##########\nPass {pass_nr}:")
    print(f"Correct: {pass_corr_perc}%")
    print(f"Corrections: {pass_avg_corr}")
    
    print(np.array(collect_pass_results))
    
  return collect_pass_results
        
    

if __name__ == "__main__":
  # login() Log in to github here #TODO
  # Removed to prevent committing secret #TODO

  print("\nRunning -- TEST --\n")
  #print("Using device -", device_map)
  
  output_dir = "generated_solutions"
  os.makedirs(output_dir, exist_ok=True)
  #output_name = "base_model_full_mbpp"
  #output_name = "fourbit_no_finetune_full_mbpp"
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="qwen", help='must be "mistral", "deepseek", "qwen"')
  parser.add_argument('--quant_version', type=str, default="fourbit", help='must be "base", "fourbit", "eightbit"')
  parser.add_argument('--rank', type=int, default=8, help='must be 8, 16, 32, 64, 128')
  parser.add_argument('--prompt_mode', type=str, default="single", help='must be "single", "iter"')
  parser.add_argument('--pass_n', type=int, default=1, help='must be 1, 10')
  args = parser.parse_args()  
  
  dataset_name = "test_mbpp"
  prompt_mode = args.prompt_mode
  
  model_name = args.model_name
  quant_version = args.quant_version
  pass_n = args.pass_n
  #is_finetune = ()
  
  assert prompt_mode in ["single", "iter"]
  assert dataset_name in ["test_mbpp"]
    
  assert model_name in ["mistral", "deepseek", "qwen", "yi"]
  assert quant_version in ["base", "fourbit", "eightbit"]

  
  model_path = None
  # base no finetune
  if model_name == "mistral":
    if quant_version == "fourbit":
      model_path = "outputs_mistral8_fourbit/best_model"
    config = Mistral(quant_version, model_path=model_path)
  
  elif model_name == "qwen":
    if quant_version == "fourbit":# and args.finetune_type == "best":
      model_path = "outputs_qwen8_fourbit/best_model"
    config = Qwen(quant_version, model_path=model_path)
  
  elif model_name == "deepseek":
    if quant_version == "fourbit":# and args.finetune_type == "best":
      model_path = "outputs_deepseek8_fourbit/best_model"
    config = Deepseek(quant_version, model_path=model_path)
    
  elif model_name == "yi":
    #if quant_version == "fourbit" and args.finetune_type == "best":
    #  model_path = "outputs_mistral8_fourbit/best_model"
    config = Yi(quant_version, model_path=model_path)
 #config.lora_checkpoint= "/home2/s3978389/Thesis/outputs/CIA_18k/checkpoint-2000/"
    
  else:
    raise NotImplementedError
    
  #mem_before = mem_before = torch.cuda.memory_allocated() / 1e9
  config.init_model()
  print("Computing model size:")
  print(get_true_model_size_gb(config.model))
  #print(f"Model size (params only): {get_model_size_gb(config.model):.2f} GB")
  #print(f"True model size (including bnb): {get_true_model_size_gb(config.model):.2f} GB")
  #print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
  
  
  all_params, trainable_params = calc_trainable_parameters(config.model)
  
  print("\n`--------------------------------------`")
  print(f"`Model:               # {args.model_name}`")
  print(f"`Quantization:        # {args.quant_version}`")
  print(f"`Prompt mode:         # {args.prompt_mode}`")
  print(f"`Pass@_:              # {args.pass_n}`")
  print(f"`Batch size:          # {config.max_batch_size}`")
  print(f"`Trainable params:    # {(trainable_params/all_params)*100}`")
  #print(f"`VRAM used:           # {mem_after-mem_before} GB`")
  
  #print(f"'Attention:           # {config.model.model.layers[0].self_attn.__class__}")
  print("`--------------------------------------`\n")
  
  
  output_name = f"{model_name}_{quant_version}_{prompt_mode}_pass@{pass_n}"
  print(f"Outputting: {output_name}")
  
  out = test( output_name = output_name,
              config = config,
              #test_type = test_type,
              prompt_mode = prompt_mode,
              dataset_name = dataset_name,
              verbose = True,
              pass_n = pass_n)
  
  print(out)
  
  base_dir = os.path.dirname(os.path.abspath(__file__))
  
  save_path = os.path.join(base_dir, "Results", output_name)
  os.makedirs(save_path, exist_ok=True)
  
  # Create HF dataset
  hf_dict = {key: [d[key] for d in out] for key in out[0]}
  out_dataset = Dataset.from_dict(hf_dict)
  out_dataset.save_to_disk(save_path)
  
  all_evals = np.array(out_dataset['evaluations'])
  combined_evals = np.any(all_evals, axis=0)
  correct = sum(combined_evals)
  total = len(combined_evals)
  print(f"Correct: {correct}/{total} ({((correct/total)*100):.2f})")
  
  c = np.sum(all_evals, axis=0)
  print(f"c: {c}")
  n = args.pass_n
  k = 2

  c = np.sum(all_evals, axis=1)
  print(f"Pass@{k}: {pass_at_k(c, n, k)}")