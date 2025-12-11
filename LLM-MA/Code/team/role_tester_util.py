from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, PeftModel

def generate_response(model, tokenizer, prompt_batch, temp=0.7):
    #print(f"temp: {temp}")
    inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True).to("cuda")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    do_sample = False
    if temp > 0.0:
      do_sample= True
    else:
      temp = None

    output = model.generate(
        input_ids,
        attention_mask= attention_mask,  # Explicitly set attention mask
        max_new_tokens= 512,
        temperature=    temp,
        do_sample=      do_sample,
        pad_token_id=   tokenizer.pad_token_id  # Avoids warning
    )
    
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    return decoded_output

class Standard_Config:    
    def __init__(self):
      self.model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
      
      self.use_lora: bool = False
      self.lora_checkpoint: Optional[str] = None
      self.lora_config: LoRA_config_four("default").load()
  
      self.attention_type: str = "flash_attention_2"
      
      self.device_map = "auto"
      self.max_batch_size = 200
  
    
    def load_model(self, device_map = None):
      if device_map:
        self.device_map = device_map
        base_model = AutoModelForCausalLM.from_pretrained(
          self.model_name,
          device_map=            self.device_map,
          torch_dtype =          torch.bfloat16,
          attn_implementation=   self.attention_type,
      )
      else:
        base_model =AutoModelForCausalLM.from_pretrained(
          self.model_name,
          torch_dtype =          torch.bfloat16,
          attn_implementation=   self.attention_type,
      )
  
      if self.use_lora:
          assert self.lora_checkpoint is not None, "LoRA checkpoint path must be set"
          model = PeftModel.from_pretrained(base_model, self.lora_checkpoint)
      else:
          model = base_model
  
      return model
    
    def load_tokenizer(self):
      tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      tokenizer.pad_token = tokenizer.eos_token
      return tokenizer