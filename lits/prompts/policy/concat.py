task_prompt_spec_math_qa = """Your task is to give the correct next step, given a science problem and an existing partial solution (not a complete answer). 
Assuming the input is n-steps, then the format of the input is:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ..."
where ... denotes omitted input information. 

Please follow the restricted output format:
* If no existing steps are given, generate Step 1.
* Otherwise, following the given step(s), output ONLY ONE step within 1000 tokens. SHOULD NOT output multiple steps.
* DO NOT repeat Problem or any Existing Steps.
* Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc. 
* When the final answer is/has been reached, begin the step with EXACTLY the phrase:\"Now we can answer the question: The answer is \", followed by EXACTLY one number. Do not include any other words, punctuation, or explanation after the number."""


task_prompt_spec_structured_math_qa = """Your task is to give the correct next step, given a science problem and an existing partial solution.

Return only one valid JSON object with exactly one non-empty string field. Use the field \"step\" when another reasoning operation is needed; its value must perform the actual next operation. Use the field \"answer\" only when the final answer has been established; its value must contain only that answer. Never return both fields. Do not use Markdown or explanatory text outside the JSON object."""


task_prompt_spec_joint = """Return the requested number of distinct candidates via submit_reasoning_steps once.
Each candidate independently continues the same partial solution; do not repeat existing steps or siblings.
Use exactly one nonempty field: step for an intermediate deduction when further reasoning is needed, or answer for the established final answer to the original question.
Once the question is solved, use answer immediately; do not repeat the derivation or put the final answer in step.
Perform the reasoning; do not merely describe what to do next.
Include reward in [0, 1] estimating absolute correctness and progress, not relative rank. Scores need not sum to one."""
