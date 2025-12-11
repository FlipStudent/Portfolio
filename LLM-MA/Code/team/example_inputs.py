EX_QUESTION = "Write a function to sort a given matrix in ascending order according to the sum of its rows."


EX_ANALYSIS = '''**User Requirement:**
Write a function that sorts a given matrix in ascending order according to the sum of its rows.

**Plan:**

1. **Decompose the requirement:**
   - Define a function to calculate the sum of a given row.
   - Define a function to sort a list of numbers in ascending order.
   - Define a function to iterate through a matrix and apply the sorting function to each row.

2. **High-Level Development Plan:**

   a. **Calculate Sum of a Row:**
     - For each row, initialize a variable to store the sum of its elements.
     - Iterate through each element in the row and add it to the sum variable.
     - Return the sum variable.

   b. **Sort a List of Numbers:**
     - Initialize an empty list to store the sorted numbers.
     - Iterate through the list, taking each number and append it to the sorted list in ascending order.
     - Return the sorted list.

   c. **Sort Matrix:**
     - Initialize an empty list to store the sorted rows.
     - Iterate through the matrix, taking each row and apply the row sorting function.
     - Save the sorted row to the sorted list.
     - Return the sorted list (which is the sorted matrix).

**Notes:**

- The
'''


EX_CODE = "def solution(matrix):\n matrix.sort(key=sum)\n return matrix"



EX_REPORT = '''{
"Code Review":
The code structure and syntax is correct. The code uses the built-in sort method with the key parameter to sort the matrix by the sum of its rows. The API used is correct. No syntax errors or logic holes were found.

"Code Description":
The code sorts a given matrix in ascending order according to the sum of its rows.

"Satisfying the requirements": True

"Edge cases":
The code should handle empty matrices, matrices with a single row, and matrices with multiple rows.

"Conclusion": Code Test Passed
}
'''