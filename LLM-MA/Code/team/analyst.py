from team import role_descriptions
from team import role_tester_util
import re

def extract_analysis(text):
    match = re.search(r'---\s*\n+(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Outdated
def analyze(model, tokenizer, user_requirement:str = None):
  prompt = TEAM + "\nThis is the user requirement: " + user_requirement + "\n" + ANALYST
  prompt += f"\n---\n"
  print(f"Analysis prompt: \n{prompt}")
  
  full_prompt_with_analysis = generate_code(model, tokenizer, prompt)
  return extract_analysis(full_prompt_with_analysis)

    
# Outdated
if __name__ == "__main__":
  config = util.FourB_Config()
  config.use_lora = True
  config.lora_checkpoint= "/home2/s3978389/Thesis/outputs/CIA_18k/checkpoint-2000/"
  
  model = config.load_model()
  tokenizer = config.load_tokenizer()

  generated_code = analyze(  model = model,
         tokenizer = tokenizer,
         user_requirement = exin.EX_QUESTION)
         
  print(generated_code)