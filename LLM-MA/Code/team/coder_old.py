from team import role_descriptions as rd
from team import role_tester_util
import re


# Formulae made with the help of chatGPT
def extract_code(text):
    match = re.search(r'-C-\s*\n+(.*)', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(r"```python\s+([\s\S]*?)```|```python\s+([\s\S]*)", text, re.DOTALL)
    if match:
        text = code = match.group(1) if match.group(1) is not None else match.group(2)
    return text

def code(config, user_requirements:str = None, messages:str = None, code_temp=0.7):
  prompts = []
  for idx, requirement in enumerate(user_requirements):
    prompt = rd.TEAM + "\n" + rd.DEVELOPER + "\n"
    prompt += "This is the user requirement: " + requirement + "\n"
    prompt += messages[idx]
    prompt += f'Remember, only provide code: do NOT explain the code and do NOT give test cases.'
    prompts.append(prompt)
    #print("Skipped the analysis")
    #if messages[idx] != "":
      #print(f"###! REPORT PROMPT !###\n{prompt}")
  #print(f"#Coder prompt 0:#\n {prompts[0]}")

  # Generate responses
  codes_with_prompt_repeat = config.prompt(prompts, temp=code_temp)
  
  
  # Extract code
  extracted_codes = []
  for single_repeat_code in codes_with_prompt_repeat:
    extracted_codes.append(extract_code(single_repeat_code))
  
  #print(f"\n##Codes without repeat:## \n{extracted_codes[0]}")
  print(f"\n#Coding prompt 0: \n{prompts[0]}")
  print(f"\n#Coding output 0: \n{extracted_codes[0]}") 
  return extracted_codes
  

    
if __name__ == "__main__":
  from team import example_inputs    # Just for testing purposes
  config = Standard_Config()
  
  model = config.load_model().to("cuda")
  tokenizer = config.load_tokenizer()
  
  prevrole = "analyst"
  assert prevrole in ["analyst", "tester"] 

  code = code(model = model,
              tokenizer = tokenizer,
              user_requirements = [EX_QUESTION]*5,
              prevroles = ["analyst"]*5,
              messages = [EX_ANALYSIS]*5,
              code_temp = 0.7)
  
  print(code)