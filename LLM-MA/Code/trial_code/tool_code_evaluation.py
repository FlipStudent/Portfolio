import ast
import re
import multiprocessing
from datasets import load_dataset

# Contributions:
# Code compilation & function running: ChatGPT
# Logic & function correctness: me

class Evaluator():
  def __init__(self):
    pass
    
  def function_wrapper(self, queue, func, args, kwargs):
      try:
          result = func(*args, **kwargs)
          queue.put({"error": False, "output": result})
      except Exception as e:
          queue.put({"error": True, "output": str(e)})
  
  
  def run_with_timeout(self, func, args=(), kwargs={}, timeout=20):
      queue = multiprocessing.Queue()
      process = multiprocessing.Process(target=function_wrapper, args=(queue, func, args, kwargs))
      process.start()
      process.join(timeout)
  
      if process.is_alive():
          process.terminate()
          process.join()
          return {"error": True, "output": f"Execution timed out after {timeout} seconds."}
  
      return queue.get() if not queue.empty() else {"error": True, "output": "No result returned."}
  
  
  def extract_code(self, text):
      match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
      if match:
          return match.group(1).strip()
      else:
          match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
          if match:
              return match.group(1).strip()
      return text

  def full_execution(self, queue, code, test_input):
      local_vars = {
          "__builtins__": __builtins__,
      }
      try:
          compiled_code = compile(code, "<string>", "exec")
          exec(compiled_code, local_vars, local_vars)
  
          input_args = eval(f"[{test_input}]", local_vars, local_vars)
  
          func_defs = [node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)]
          func_name = func_defs[-1] if func_defs else "solution"
  
          if func_name not in local_vars:
              queue.put({"error": True, "output": "Function not found"})
              return
  
          result = local_vars[func_name](*input_args)
          queue.put({"error": False, "output": result})
      except Exception as e:
          queue.put({"error": True, "output": str(e)})
  
  
  def run_code_safely(self, code, test_input, timeout=20):
      queue = multiprocessing.Queue()
      process = multiprocessing.Process(target=self.full_execution, args=(queue, code, test_input)) # Maybe remove self
      process.start()
      process.join(timeout)
  
      if process.is_alive():
          process.terminate()
          process.join()
          return {"error": True, "output": f"Execution timed out after {timeout} seconds."}
  
      return queue.get() if not queue.empty() else {"error": True, "output": "No result returned."}
      
      
  def evaluate_code(self, aug_mbpp_partition, question_index, generated_code):
    in_out_pairs = aug_mbpp_partition[question_index]['in_out_pairs']
  
    for io_pair in in_out_pairs:
      input = io_pair['input']
      output_log = self.run_code_safely(generated_code, input)
      expected_output = eval(io_pair['output'])
      # print(f"Input: {input} ({type(input)} ({len(input)})), output: {output_log} ({expected_output})")
      if output_log['error'] == True:
        print(f"Compiling error: {output_log['output']}")
        return False
      if output_log['output'] != expected_output:         
        print(f"Incorrect output: {output_log['output']} - expected: {expected_output}")
        #print(generated_code)
        #print("Test")
        return False
  
    return True
