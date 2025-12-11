# config.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM
from dataclasses import dataclass, field
from typing import Optional
import torch
import random
import numpy as np
import collections
import torch
import os

torch.manual_seed(333)
random.seed(333)
np.random.seed(333)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_trainable_parameters(model):
    all_param, trainable_params = calc_trainable_parameters(model)
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    """
    printing the number of trainable paramters in the model
    """

    return

def calc_trainable_parameters(model):
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  return all_param, trainable_params

def get_model_size_gb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_bytes = param_size + buffer_size
    size_in_gb = total_size_bytes / (1024 ** 3)
    return size_in_gb

def get_true_model_size_gb(model):
    total_size_bytes = 0
    quantized_size = 0
    lora_size = 0
    other_size = 0

    for name, module in model.named_modules():
        # Check for bitsandbytes quantized layer
        if 'bnb' in str(type(module)).lower():
            for buffer_name, buffer in module.named_buffers():
                size = buffer.numel() * buffer.element_size()
                total_size_bytes += size
                quantized_size += size

        # Count trainable LoRA params (adapter layers)
        if 'lora' in name.lower():
            for param in module.parameters():
                if param.requires_grad:
                    size = param.numel() * param.element_size()
                    total_size_bytes += size
                    lora_size += size

    # Add any remaining parameters and buffers
    for param in model.parameters():
        size = param.numel() * param.element_size()
        total_size_bytes += size
        other_size += size

    for buffer in model.buffers():
        size = buffer.numel() * buffer.element_size()
        total_size_bytes += size
        other_size += size

    size_gb = total_size_bytes / (1024 ** 3)
    return {
        "total_gb": size_gb,
        "quantized_gb": quantized_size / (1024 ** 3),
        "lora_gb": lora_size / (1024 ** 3),
        "other_gb": other_size / (1024 ** 3)
    }
    
def count_4bit_layers(model):
    from bitsandbytes.nn import Linear4bit
    return sum(isinstance(m, Linear4bit) for m in model.modules())

class DebugCollator(DataCollatorForCompletionOnlyLM):
    def __call__(self, features):
        #print("?? Collator was called")
        batch = super().__call__(features)
        #print(batch)
        return batch

def combine_batch_strings(lists):
  return ["".join(items) for items in zip(*lists)]

def get_device_map(quant_version="base"):
  if quant_version == "base":
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}
  else:
    return {"": torch.cuda.current_device()}

def get_quant_config( version):
        if version == "fourbit":
            print("Prepared 4-bit configuration")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif version == "eightbit":
          print("Prepared 8-bit configuration")
          return BitsAndBytesConfig(
            load_in_8bit=True,
            torch_dtype=torch.float16 # 8bit doesnt work with bf16
          )
        elif version == "base":
          print("Prepared base configuration")
          return None
        else:
            raise ValueError(f"Unknown quant_version: {version}")

class Config:
  def __init__(self, model_name, batch_map, quant_version="base", model_path=None):
    self.quant_version = quant_version
    self.model_path = model_path
    
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device_map: str = get_device_map(self.quant_version)
    self.data_type = torch.bfloat16 if self.quant_version != "eightbit" else torch.float16
    
    self.max_batch_size = batch_map[self.quant_version]
    
    self.lora_config = self.load_lora_config()
    
    self.bnb_config = get_quant_config(self.quant_version)
    
    self.tokenizer = self.load_tokenizer()
    

    self.attention_type = "flash_attention_2" if self.quant_version != "eightbit" else None
    
    #print(f"Tokenizer:           {self.tokenizer}")
    #print(f"Attention:           {self.attention_type}")
    #print(f"Maximum batch size:  {self.max_batch_size}")
    #print(f"Lora config:         {self.lora_config}")
    #print(f"Bnb config:          {self.bnb_config}")
    
    
  def init_model(self):
    self.model = self.load_model()
    print(f"Base model size: {get_model_size_gb(self.model):.2f} GB")
    
    if self.bnb_config is None:
      print("Loaded base model")
      return
    elif self.bnb_config.load_in_4bit:
      print("Loaded model in 4-bit")
      #(f"#4-bit layers after load_model(): {count_4bit_layers(self.model)}")
    elif self.bnb_config.load_in8bit:
      print("Loaded model in 8-bit")      
    
    # Load finetuned model for inference
    if self.model_path and os.path.exists(self.model_path):
      # Load finetuned adapter on top of base model
      self.model = PeftModel.from_pretrained(self.model, self.model_path)
      print(f"# Loaded fine-tuned model from: {self.model_path} #")
      print(f"Model size after checkpoint: {get_model_size_gb(self.model):.2f} GB")
      #print(f"#4-bit layers after loading finetuned model: {count_4bit_layers(self.model)}")
      
    else:
      print(f"No save loaded")
  
      # Otherwise prepare and load it for training
      if self.quant_version != "base":
        print(f"Size before preparing for kbit training: {get_model_size_gb(self.model):.2f} GB")
        self.model = prepare_model_for_kbit_training(self.model)
        print(f"Size after preparing for kbit training size: {get_model_size_gb(self.model):.2f} GB")
        self.model = get_peft_model(self.model, self.lora_config)
        print(f"Size after getting peft model: {get_model_size_gb(self.model):.2f} GB")
        
        print(f"# Prepared model for {self.quant_version.upper()} QLoRA training. #")
        #print(f"#4-bit layers after preparing for training: {count_4bit_layers(self.model)}")
    
    print("Computing model size")
    model_size_gb = get_model_size_gb(self.model)
    print(f"Model size: {model_size_gb:.2f} GB")
    print(f"#4-bit layers after init(): {count_4bit_layers(self.model)}")
    
  # Done
  def load_model(self):
    model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        torch_dtype=self.data_type,
        device_map= self.device_map,
        attn_implementation=self.attention_type,
        quantization_config = self.bnb_config
    )
    
    #print(f"#Attention: {model.model.layers[0].self_attn.__class__}")
    return model
  
##############################################################################

# QWEN
class Qwen(Config):
  def __init__(self, quant_version="base", model_path=None, lora_rank=8):
    self.model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    self.batch_map = {"base":40, "eightbit":80, "fourbit":100} # TODO: Adjust
    self.lora_rank = lora_rank
    super().__init__(self.model_name, self.batch_map, quant_version, model_path)
    self.collator = DataCollatorForCompletionOnlyLM('<|im_start|>assistant', tokenizer=self.tokenizer, pad_to_multiple_of=8)
  
  def format_prompt_batch(self, items):    
    if not isinstance(items, collections.abc.Mapping):
      items = {'text':items}
      add_generation = True
      #print("Adding generation")
      
    strings = items['text']
    output_format = []
    for i, message in enumerate(strings):
      chat_string = [
          {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
          {"role": "user", "content": message}
      ]
      if "code" in items:
        chat_string.append( {"role": "assistant", "content": items['code'][i]} )
        add_generation = False
      
      prompt = self.tokenizer.apply_chat_template(chat_string, tokenize=False, add_generation_prompt=add_generation)
      #print(prompt)
      output_format.append(prompt)
    return output_format
    
  

  def prompt(self, messages, temp, max_out_len=256):
    assert all(isinstance(m, str) for m in messages), "Each message must be a string!"
    prompts = self.format_prompt_batch(messages)
  
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
    
    if temp == 0.0:
      temp = None
      do_sample = False
      top_p = None
      top_k = None
      enable_thinking=False
    else:
      temp = 0.2
      do_sample = True
      top_p = 0.95
      top_k = 20
      enable_thinking=True
    
    outputs = self.model.generate(**inputs, 
                         max_new_tokens=max_out_len, 
                         temperature=temp,
                         do_sample=do_sample,
                         top_p=top_p,        # ? nucleus sampling for focus
                         top_k=top_k,
                         repetition_penalty =1.0,  # ? KEY to reduce loops
                         no_repeat_ngram_size=4, # ? helps prevent trivial loops
                         pad_token_id = self.tokenizer.pad_token_id,
                         eos_token_id = self.tokenizer.eos_token_id
                         )
    decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    #print("First normal decoded:")
    #print(decoded_output[0])
    for i, single_decoded_out in enumerate(decoded_output):
      in_str, decoded_output[i] = decoded_output[i].rsplit("assistant", 1)

    #print(f"`outputs:`{decoded_output[0]}`endof`")
    return decoded_output
    
  def tokenize_function(self, example_batch):
    return self.tokenizer(self.format_prompt_batch(example_batch),
                            padding=False,
                            truncation=False
                            )

  def load_lora_config(self):
    lora_config = LoraConfig(
      r=self.lora_rank,
      lora_alpha=int(0.5*self.lora_rank),
      target_modules=[
        "q_proj", "v_proj"#, "k_proj", "o_proj"
      ],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
      )
    return lora_config
  
  def load_tokenizer(self):
    return AutoTokenizer.from_pretrained(self.model_name, padding_side='left')

################################################################################################################

# DEEPSEEK
class Deepseek(Config):
  def __init__(self, quant_version="base", model_path=None, lora_rank=8):
    self.model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    self.batch_map = {"base":10, "eightbit":10, "fourbit":10}
    self.lora_rank = lora_rank
    super().__init__(self.model_name, self.batch_map, quant_version, model_path)
    self.collator = DataCollatorForCompletionOnlyLM('### Response:', tokenizer=self.tokenizer, pad_to_multiple_of=8)
    
    
  def format_prompt_batch(self, items):
    if not isinstance(items, collections.abc.Mapping):
      items = {'text':items}

    strings = items['text']
    bs = len(strings)
    output_format = []
    output_format.append( ['You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company.\n### Instruction:\n']*bs )
    output_format.append(strings)
    output_format.append( ['\n### Response:\n']*bs )
    if "code" in items:
      output_format.append(items['code'])
    
    combined = combine_batch_strings(output_format)
    
    return combined

  
  def prompt(self, messages, temp, max_out_len=256):
    assert all(isinstance(m, str) for m in messages), "Each message must be a string!"
    
    #print(f"`Message A`:\n{messages[0]}`END`")
    prompts = self.format_prompt_batch(messages)
    #print(f"`Formatted B`:\n{prompts[0]}`END`")
  
    # prompts = self.format_prompt_batch(messages)
    # Tokenizer (pr, return pt, padding True, trunc True).to(self.device)
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

    outputs = self.model.generate(
                          **inputs, 
                          max_new_tokens=max_out_len,
                          temperature=temp,
                          do_sample = True if temp > 0.0 else None,
                          num_return_sequences=1,
                          pad_token_id= self.tokenizer.pad_token_id,
                          eos_token_id=self.tokenizer.eos_token_id
                          )
                          
    decoded_out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    for i, single_decoded_out in enumerate(decoded_out):
      in_str, decoded_out[i] = decoded_out[i].split("### Response:", 1)
    return decoded_out
    
  
  def load_tokenizer(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    return tokenizer

  def load_lora_config(self):
    lora_config = LoraConfig(
      r=self.lora_rank,
      lora_alpha=int(0.5*self.lora_rank),
      target_modules=[
        "q_proj", "v_proj"#, "k_proj", "o_proj"
      ],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
      )
    return lora_config
  
  def tokenize_function(self, example_batch):
    return self.tokenizer(self.format_prompt_batch(example_batch), 
                            return_tensors=None,
                            padding=False,
                            truncation=True
                            )

#####################################################################################################

# Mistral
class Mistral(Config):
    def __init__(self, quant_version="base", model_path=None, lora_rank=8):
      self.model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
      self.batch_map = {"base":5, "eightbit":30, "fourbit":10} #EDIT
      self.tokenizer = self.load_tokenizer()
      self.lora_rank = lora_rank
      super().__init__(self.model_name, self.batch_map, quant_version, model_path)
      self.collator = DataCollatorForCompletionOnlyLM('[/INST]', tokenizer=self.tokenizer, pad_to_multiple_of=8)
      
    
      #######
  
    def prompt(self, messages, temp, max_out_len=256):
      assert all(isinstance(m, str) for m in messages), "Each message must be a string!"

      prompts = self.format_prompt_batch(messages)
      
      inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
      
      input_ids = inputs["input_ids"]
      attention_mask = inputs["attention_mask"]
  
      output = self.model.generate(
          input_ids,
          attention_mask= attention_mask,  # Explicitly set attention mask
          max_new_tokens= max_out_len,
          temperature=    temp,
          do_sample=      True if temp > 0.0 else None,
          pad_token_id=   self.tokenizer.pad_token_id,  # Avoids warning
          eos_token_id=   self.tokenizer.eos_token_id
      )
      
      decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=False)
      
      
      for i, single_decoded_out in enumerate(decoded_output):
        in_str, out_str = decoded_output[i].split('[/INST] ', 1)
        decoded_output[i] = out_str.split('</s>')[0]
      return decoded_output
    
    def load_tokenizer(self):
      tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      tokenizer.padding_side = 'left'
      tokenizer.pad_token = tokenizer.eos_token # TODO: recheck this
      return tokenizer
    
    def load_lora_config(self):
      lora_config = LoraConfig(
        r=self.lora_rank,
        lora_alpha=int(0.5*self.lora_rank),
        target_modules=["q_proj", "v_proj"],#, "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
        )
      return lora_config
    
    def tokenize_function(self, example_batch):
      return self.tokenizer(self.format_prompt_batch(example_batch), 
                            return_tensors=None,
                            padding=False,
                            truncation=True
                            )


    def format_prompt_batch(self, items):
      if not isinstance(items, collections.abc.Mapping):
        items = {'text':items}
        add_generation = True
#      print("items:")
#      print([key for key in items])
#      print(items['code'][:5])
#      
#      strings = items['text']
#      bs = len(strings)
#      output_format = []
#      output_format.append( ["[INST]"]* bs )
#      output_format.append( ["Below is an instruction that describes a task. Write a concise response that appropriately completes the request.\n"]*bs )
#      output_format.append(strings)
#      output_format.append( ["[/INST]"]*bs )
#      if "code" in items:
#        print(f"`control`: {len(items['code'])}")
      #  output_format.append(["\n"+code for code in items['code']])
      
        #output_format.append( ["`</s>`"]*bs )
      #combined = combine_batch_strings(output_format)
      
      output_format = []
      for i, message in enumerate(items['text']):
        chat_string = [
            {"role": "system", "content": "You are a helpful Python assistant."},
            {"role": "user", "content": message}
        ]
        if "code" in items:
          chat_string.append( {"role": "assistant", "content": items['code'][i]} )
          add_generation = False
      
        prompt = self.tokenizer.apply_chat_template(chat_string, tokenize=False, add_generation_prompt=add_generation)
        #print(prompt)
        output_format.append(prompt)
      
      return output_format


#################################################################################

# YI
class Yi(Config):
  def __init__(self, quant_version="base", model_path=None, lora_rank=8):
    self.model_name: str = "01-ai/Yi-Coder-9B-Chat"
    self.batch_map = {"base":14, "eightbit":10, "fourbit":20}
    self.lora_rank = lora_rank
    super().__init__(self.model_name, self.batch_map, quant_version, model_path)
    self.collator = DataCollatorForCompletionOnlyLM('<|im_start|>assistant\n', tokenizer=self.tokenizer, pad_to_multiple_of=8)
    
    
  def format_prompt_batch(self, items):
      if not isinstance(items, collections.abc.Mapping):
        items = {'text':items}
        add_generation = True
              
      output_format = []
      for i, message in enumerate(items['text']):
        chat_string = [
            {"role": "system", "content": "You are a helpful Python assistant."},
            {"role": "user", "content": message}
        ]
        if "code" in items:
          chat_string.append( {"role": "assistant", "content": items['code'][i]} )
          add_generation = False
      
        prompt = self.tokenizer.apply_chat_template(chat_string, tokenize=False, add_generation_prompt=add_generation)
        #print(prompt)
        output_format.append(prompt)
      
      return output_format

  
  def prompt(self, messages, temp, max_out_len=256):
    assert all(isinstance(m, str) for m in messages), "Each message must be a string!"
    
    #print(f"`Message A`:\n{messages[0]}`END`")
    prompts = self.format_prompt_batch(messages)
    #print(f"`Formatted B`:\n{prompts[0]}`END`")
  
    # prompts = self.format_prompt_batch(messages)
    # Tokenizer (pr, return pt, padding True, trunc True).to(self.device)
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

    outputs = self.model.generate(
                          **inputs, 
                          max_new_tokens=max_out_len,
                          temperature=temp,
                          do_sample = True if temp > 0.0 else None,
                          num_return_sequences=1,
                          pad_token_id= self.tokenizer.pad_token_id,
                          eos_token_id=self.tokenizer.eos_token_id
                          )
                          
    decoded_out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    for i, single_decoded_out in enumerate(decoded_out):
      print(decoded_out[i])
      in_str, decoded_out[i] = decoded_out[i].split("assistant", 1)
    return decoded_out
    
  
  def load_tokenizer(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left')
    return tokenizer

  def load_lora_config(self):
    lora_config = LoraConfig(
      r=self.lora_rank,
      lora_alpha=int(0.5*self.lora_rank),
      target_modules=[
        "q_proj", "v_proj"#, "k_proj", "o_proj"
      ],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
      )
    return lora_config
  
  def tokenize_function(self, example_batch):
    return self.tokenizer(self.format_prompt_batch(example_batch), 
                            return_tensors=None,
                            padding=False,
                            truncation=True
                            )