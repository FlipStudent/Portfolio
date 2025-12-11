from team import role_descriptions as rd
from team import role_tester_util
import re

# Formulae made with the help of chatGPT
def extract_report(text):
    match = re.search(r'-T-\s*\n+(.*)', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(r"###\s*Response:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text



def test(config, user_requirements:str = None, codes:str = None):
  # Create input
  prompts = []
  for idx, requirement in enumerate(user_requirements):
    prompt = rd.TEAM + "\n" + rd.YOU_ARE_TESTER + "\nThis is the user requirement: " + requirement
    prompt += f"\nThe code provided by developer is as follows:\n{codes[idx]}\n\n"
    prompt += rd.TESTER_NEW
    prompts.append(prompt)
  
  # Generate
  test_reports_with_repeat = config.prompt(prompts, temp=0.0)
  
  # Clean output
  extracted_reports = []
  for repeat_report in test_reports_with_repeat: 
    extracted_reports.append( extract_report(repeat_report) )
  
  #print(f"\n#Testing prompt 0: \n{prompts[0]}\n")
  #print(f"\n#Tester output 0: \n{extracted_reports[0]}")
  
  return extracted_reports

    
if __name__ == "__main__":
  config = Standard_Config()
  
  model = config.load_model().to("cuda")
  tokenizer = config.load_tokenizer()
  
  example_question = "Write a function to sort a given matrix in ascending order according to the sum of its rows."
  example_code = "def solution(matrix):\n matrix.sort(key=sum)\n return matrix"

  reps = test(  model = model,
                tokenizer = tokenizer,
                user_requirements = [example_question]*5,
                codes = [example_code]*5)
  
  print(reps)