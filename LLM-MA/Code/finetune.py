import os
import sys
import argparse
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
import torch

from config import Qwen, Deepseek, Mistral, Yi, print_trainable_parameters
#import torch.distributed as dist
#from torch.utils.data import DistributedSampler
#from torch.nn.parallel import DistributedDataParallel as DDP
from tools.tool_logger import LossLoggerCallback
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from test import construct_task_prompts
import shutil


from torch.utils.data import DataLoader ## Test

def map_tasks(batch_example):
  prompts = construct_task_prompts(batch_example)
  print(len(prompts))
  return {'text': prompts}

def test_mapping(whole_cia, config):
  print(f"`INSPECT INPUT MAP:`")
  # Take a small subset
  sample_dataset = whole_cia.select(range(4))
  
  # Create a DataLoader using your collator
  dataloader = DataLoader(sample_dataset, batch_size=2, collate_fn=config.collator)
  
  # Fetch one batch
  batch = next(iter(dataloader))
  
  # Decode and inspect
  input_ids = batch["input_ids"]
  labels = batch["labels"]
  
  for i in range(input_ids.shape[0]):
      print(f"\n#=== Sample {i} ===")
      print("Decoded input:")
      print(config.tokenizer.decode(input_ids[i], skip_special_tokens=False))
      
      print("\nDecoded labels:")
      label_tokens = [token if label != -100 else -100 for token, label in zip(input_ids[i], labels[i])]
      print(config.tokenizer.decode([t for t in label_tokens if t != -100]))
      
      print("\nNon-masked Tokens (used for loss):")
      visible_tokens = [id for id, l in zip(input_ids[i], labels[i]) if l != -100]
      print(config.tokenizer.decode(visible_tokens))
      
      print("? # of labeled tokens:", sum(label != -100 for label in labels[i].tolist()))
  return
  
def test_lora(config):
  model = config.model
  for name, param in model.named_parameters():
    if param.requires_grad and "lora" not in name:
        print("Unexpected trainable:", name)

# Initialize distributed training
#dist.init_process_group(backend="nccl")
#torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

#model_name = "mistralai/Mistral-7B-Instruct-v0.3"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.pad_token = tokenizer.eos_token

def freeze_model(model):
  for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
      param.data = param.data.to(param_dtype)
  return model
#model = AutoModelForCausalLM.from_pretrained(model_name)


  
def load_training_args(output_dir):
  batch_size = 8
  accumulate = 2
  return TrainingArguments(
          ddp_find_unused_parameters=False,
          gradient_checkpointing=False, # Set to true for reduced GPU memory usage

          fp16 = False,
          bf16 = True,

          dataloader_num_workers=4, # Seems to make no difference from 8
          dataloader_pin_memory=False,
          remove_unused_columns=False,
          
          per_device_train_batch_size = batch_size, # Seems to be the max for these GPUs
          gradient_accumulation_steps = accumulate, # Accumulates gradients over batches, effectively results in a xN batch size
  
          learning_rate = 2e-5,
          warmup_steps = 200,
          optim="adamw_torch",
          lr_scheduler_type="cosine",
          
          #max_steps = 200,
          num_train_epochs=20,
          save_steps = int(100/accumulate),
          save_total_limit=1,  
          #eval_steps=int(100/accumulate),
          logging_steps = 50,
          eval_strategy="steps", # epoch
          eval_on_start=True,
          
          load_best_model_at_end=True,
          metric_for_best_model="eval_loss",  # use evaluation loss to determine best model
          greater_is_better=False,
          output_dir = output_dir
      )
      
def save_best_model_as_best_model_folder(trainer, output_dir):
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        best_model_target = os.path.join(output_dir, "best_model")
        print(f"Saving best model from {best_ckpt} to {best_model_target}")
        shutil.copytree(best_ckpt, best_model_target, dirs_exist_ok=True)

def train(config, whole_dataset, mbpp_test, output_dir):
  model = config.model
  tokenizer = config.tokenizer
  
  training_args = load_training_args(output_dir)  
  
  model.config.use_cache = False # silence the warnings
  model.gradient_checkpointing_enable()
  
  for name, param in model.named_parameters():
      if "lora" in name and not param.requires_grad:
          print("Set lora param to grad")
          param.requires_grad = True
      
  print_trainable_parameters(model)
  
  print(model.forward.__annotations__)
  
  # Distribute
  accelerator = Accelerator(
      mixed_precision="bf16",  # Options: "no", "fp16", "bf16"
      cpu=False,               # Set to True if you want CPU-only
      device_placement=True,   # Automatically place tensors on the right device
      split_batches=True       # Split batches across devices
  )
  
  distribute_model = accelerator.prepare(model)
  
  split_ds = whole_dataset.train_test_split(train_size=0.9, shuffle=True)
  print(split_ds)
  
  loss_logger = LossLoggerCallback()
  
  # Train
  trainer = Trainer(
      model = distribute_model,
      train_dataset = split_ds['train'],
      eval_dataset  = mbpp_test,
      args = training_args,
      data_collator = config.collator,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=3), loss_logger]
  )
  
  
  last_checkpoint = get_last_checkpoint(output_dir)
  
  if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()
    
  save_best_model_as_best_model_folder(trainer, training_args.output_dir)
  loss_logger.to_csv(output_dir+"/loss_log.csv")
  print(f"Saved Loss log to {output_dir}/loss_log.csv")
  loss_logger.save_plot(output_dir+"/loss_log.png")
  print(f"Saved Loss plot to {output_dir}/loss_log.csv")
  
if __name__ == "__main__":
  # login() Log in to github here #TODO
  # Removed to prevent committing secret #TODO
  param_dtype = torch.bfloat16
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="deepseek", help='must be "mistral", "deepseek", "qwen", "yi"')
  parser.add_argument('--quant_version', type=str, default="fourbit", help='must be "base", "fourbit", "eightbit"')
  parser.add_argument('--lora_rank', type=int, default=64, help='must be an integer')
  args = parser.parse_args()
  
  print("`---FINETUNING---`")
  #print("Using device -", device_map)
  
  device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}
  
  model_name =  args.model_name#sys.argv[1]#"qwen"
  quant_version = args.quant_version#"fourbit"
  
  assert model_name in ["mistral", "deepseek", "qwen", "yi"], f"input must be a valid model: {model_name}"
  assert quant_version in ["base", "fourbit", "eightbit"], f"input must be valid quant version: {quant_version}"

  #os.makedirs(output_dir, exist_ok=True)
  
  print(f"Using `{model_name}`")
  print(f"Quant `{quant_version}`")

  match model_name:
    case "mistral":
      config = Mistral(quant_version, model_path=None, lora_rank = args.lora_rank)
  
    case "qwen":
      config = Qwen(quant_version, model_path=None, lora_rank = args.lora_rank)
  
    case "deepseek":
      config = Deepseek(quant_version, model_path=None, lora_rank = args.lora_rank)
    
    case "yi":
      config = Yi(quant_version, model_path=None, lora_rank = args.lora_rank)

  
  #whole_cia = load_dataset('arrow', data_files='./data/formatted_CIA_6k.arrow', split='train')
  #whole_cia = whole_cia.map(config.tokenize_function, batched=True, remove_columns=['text', 'code'], load_from_cache_file=False)
  
  mbpp_train = load_from_disk("./data/test/mbpp_io_augmented.hf")['train']
  mbpp_train = mbpp_train.map(map_tasks, batched=True)
  mbpp_train = mbpp_train.map(config.tokenize_function, batched=True, remove_columns=mbpp_train.column_names, load_from_cache_file=False)
  print(f"`Loaded dataset`")# - {torch.cuda.memory_allocated() / 1e9} GB used")
  mbpp_test = load_from_disk("./data/test/mbpp_io_augmented.hf")['test']
  mbpp_test = mbpp_test.map(map_tasks, batched=True, batch_size=50)
  mbpp_test = mbpp_test.map(config.tokenize_function, batched=True, remove_columns=mbpp_test.column_names, load_from_cache_file=False)


  
  #print(f"CIA: {whole_cia}")
  print(f"MBPP: {mbpp_test}")

  # For map debugging:
  #test_mapping(whole_cia, config)
  test_mapping(mbpp_test, config)
  
  print(f"`Mapped dataset`")# - {torch.cuda.memory_allocated() / 1e9} GB used")

  config.init_model() # LOAD MODEL AFTER DATASET - changing this order causes RAM crashes (process kills by habrok)  
  
  #print(get_peft_model_state_dict(config.model))      
  
  
  # For lora debugging:
  test_lora(config)

  
  output_dir = f"outputs_{model_name}{args.lora_rank}_{quant_version}"
  print(f"Outputting to: `{output_dir}`")
  train(config, mbpp_train, mbpp_test, output_dir)