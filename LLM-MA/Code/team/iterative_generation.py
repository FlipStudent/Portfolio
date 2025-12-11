from team import role_descriptions# as desc
from team import role_tester_util# as util
from team.tester import test
from team.analyst import analyze
from team.coder import code
import re

def combine_batch_strings(lists):
  return ["".join(items) for items in zip(*lists)]


# Takes batches of input!
def iter_response(config, user_requirements, do_analysis=False, code_temp=None, report_messages = None):
  batch_size = len(user_requirements)
  #if do_analysis:
  #  analyses = analyze(model, tokenizer, user_requirements) # Works as expected
  #  mesages = analyses
    #print(f"\nAnalysis output:\n{analysis}\n\n")
  
  #prevrole = "analyst"
  
  #when_passed = [False] * batch_size
  #save_codes = [False] * batch_size
  #n_passed = 0    
    
  # 'batch_size' N codes
  generated_codes = code(config, user_requirements, report_messages, code_temp=code_temp)
  
  # 'batch_size' N reports
  test_reports = test(config, user_requirements, codes=generated_codes)
  
  #has_passed = [check_passed(report) for report in test_reports]
  
  return generated_codes, test_reports
    
  # Check the reports
  #for i, report in enumerate(test_reports):
    
  #print(f"passed: {check_passed(report)} - Not yet saved: {not save_codes[i]}")
  #  if check_passed(report) and not save_codes[i]:
  #      save_codes[i] = generated_codes[i]
  #      when_passed[i] = iteration
  #      n_passed += 1

  #    if evaluator:
        
  #      report_passed = check_passed(report)
        
  #      cleaned_code = evaluator.extract_code(generated_codes[i])
                                         
  #      real_passed = evaluator.evaluate_code(aug_mbpp_partition =  mbpp_partition, 
  #                                            question_index     =  batch_start_index+i, 
  #                                            generated_code     =  cleaned_code)
        # Check the tester performance
  #      if report_passed and real_passed:
  #        correct_pass += 1
  #      elif report_passed and not real_passed:
  #        incorrect_pass += 1
  #      elif not report_passed and real_passed:
  #        incorrect_fail += 1
  #      elif not report_passed and not real_passed:
  #        correct_fail += 1
  #    
  #  print(f"\n###Tester evaluation:### (debugging)")
  #  print(f"Correct pass: {correct_pass}")
  #  print(f"Correct fail: {correct_fail}")
  #  print(f"Incorrect pass: {incorrect_pass}")
  #  print(f"Incorrect fail: {incorrect_fail}\n\n")
  #  
  #  
  #  intro_tester = ["The report from the tester is as following:\n"]*batch_size
  #  # test_reports
  #  intro_codes = ["\nThis is the previous code which you need to improve:\n"]*batch_size
  #  clean_codes = [evaluator.extract_code(dirty_code) for dirty_code in generated_codes]
  #  
  #  messages = combine_batch_strings([intro_tester, test_reports, intro_codes, clean_codes])
  #
  # Cleanup: save the final output of whatever questions never got a code past the tester
  #for i, gen_code in enumerate(generated_codes):
  #  if not save_codes[i]:
  #    save_codes[i] = gen_code
  #
  #return save_codes, when_passed

    
if __name__ == "__main__":
  config = Standard_Config()
  
  model = config.load_model().to("cuda")
  tokenizer = config.load_tokenizer()
  
  ex_q1 = "Write a function to sort a given matrix in ascending order according to the sum of its rows."
  ex_q2 = "Write a function that returns 'hello world'"


  answers, when_passed = iter_response(  model = model,
                                         tokenizer = tokenizer,
                                         user_requirements = [ex_q1, ex_q2],
                                         max_iters = 10,
                                         code_temp = 0.7)
  
  #print(f"\n\nFinal answer: \n{answers}")
  
  print(f"first answer: {answers[when_passed[0]]} ({when_passed[0]})")
  print(f"second answer: {answers[when_passed[1]]} ({when_passed[1]})")