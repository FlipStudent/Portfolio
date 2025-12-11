import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from config import FourB_Config, Standard_Config

def record_speed(model, inputs):

  
  # Warm-up
  _ = model.generate(**inputs, max_new_tokens=512)
  
  # Benchmark
  start = time.time()
  output = model.generate(**inputs, max_new_tokens=512)
  end = time.time()

  # Count tokens
  num_generated_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
  print(f"Transformers - Tokens/sec: {num_generated_tokens / (end - start):.2f}")
  
def record_batched_speed(model, tokenizer, inputs, batch_size):
  
  # Generate
  with torch.no_grad():
      torch.cuda.synchronize()
      start = time.time()
      outputs = model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
          max_new_tokens=2048,
          do_sample=False,
          pad_token_id=tokenizer.pad_token_id
      )
      torch.cuda.synchronize()
      end = time.time()
      outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  
  len_lists = [len(list(output)) for output in outputs]
  total_length = sum(len_lists)
  duration = round(end-start, 2)
  
  return total_length, duration


if __name__ == "__main__":
  # login() Log in to github here #TODO
  # Removed to prevent committing secret #TODO
  
  #print("Using base model")
  #config = Standard_Config()
  #max_batch_size = 200
  
  print("Using quantized model")
  config = FourB_Config()
  config.lora_checkpoint= "/home2/s3978389/Thesis/outputs/CIA_18k/checkpoint-2000/"
  config.use_lora = True
  max_batch_size = 500
  
  model = config.load_model().to("cuda")
  tokenizer = config.load_tokenizer()
  
  prompt = "What is the capital of France? Explain in detail."
  
  start = time.time()
  
  N = 500
  S = max_batch_size
  batch_sizes = [S] * (N // S) + ([N % S] if N % S else [])

  for batch_size in batch_sizes:     
    batched_prompts = [prompt] * batch_size
    inputs = tokenizer(batched_prompts, return_tensors="pt").to("cuda")
    
    n_tokens, duration = record_batched_speed(model, tokenizer, inputs, batch_size=batch_size)
    print(f"Batched inference ({batch_size} prompts): {n_tokens / duration} tokens/sec ({duration} sec)")
  
  end = time.time()
  print(f"Total time: {(end-start):.2f} sec")
  
  
  
  
  
  #model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')