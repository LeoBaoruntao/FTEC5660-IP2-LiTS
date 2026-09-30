task_prompt_spec_continuation = "Please reason step by step, and put your final answer within \\boxed{}."


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

structured_reasoning_json_instruction = (
    'Use plain-text mathematical notation such as cos(theta), x^2, and pi. '
    'Avoid LaTeX commands and backslashes. If a backslash is necessary, JSON '
    'requires it to be doubled: ' + r'{"step": "Use \\cos(x)."}' + '. '
    'Escape quotes inside strings; never include raw control characters.'
)
task_prompt_spec_structured_math_qa += "\n" + structured_reasoning_json_instruction


reasoning_progress_instruction = """Each candidate independently continues the same partial solution; do not repeat existing steps or siblings.
Perform the reasoning; do not merely describe what to do next.
Once the question is solved, return the established final answer immediately; do not repeat the derivation or mark it as intermediate."""

joint_completion_instruction = """Use exactly one nonempty field: step for an intermediate deduction when further reasoning is needed, or answer for the established final answer to the original question."""

single_completion_instruction = """Use exactly content (a nonempty string) and is_final (a boolean). Put an intermediate deduction in content with is_final=false when further reasoning is needed, or the established final answer with is_final=true."""

reasoning_tool_retry_template = """\nThe previous response was invalid: {error}.
Submit {candidates} in one complete {tool_name} tool call. {completion_instruction}
Use plain-text math without backslashes or control characters.
Do not repeat text, emit multiple tool calls, or return ordinary prose."""

task_prompt_spec_joint = "\n".join((
    "Return the requested number of distinct candidates via submit_reasoning_steps once.",
    reasoning_progress_instruction,
    joint_completion_instruction,
    "Include reward in [0, 1] estimating absolute correctness and progress, not relative rank. Scores need not sum to one.",
))

task_prompt_spec_single = "\n".join((
    "Return one candidate via submit_reasoning_step exactly once.",
    reasoning_progress_instruction,
    single_completion_instruction,
    "Do not assign a score. Use plain-text math or correctly JSON-escaped backslashes.",
))
