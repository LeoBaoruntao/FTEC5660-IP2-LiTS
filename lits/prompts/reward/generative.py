
# "usefulness_evaluator": "Given a chain of thoughts, determine whether the last thought is useful to answer the question. ONLY output 'Yes' or 'No'.",
CORRECTNESS_CRITERIA = """Given a question and a chain of thoughts, determine whether the last thought is **correct**, where **correct** means factually accurate and mathematically accurate (all calculations and formulas are correct), and logically consistent with the question."""

USEFULNESS_CRITERIA = """Given a question and a chain of thoughts, determine how **useful** the last thought is for answering the question, regardless of correctness.

Evaluate the progress actually made beyond the preceding thoughts, not the importance of the proposed operation. A new equation, substitution, constraint, deduction, or computed result can make progress without completing the solution. Merely instructing someone to calculate, simplify, or integrate, or restating an existing plan or result, makes no progress. For example, after "Evaluate the integral of x from 0 to 2", "Compute that integral" makes no progress; "Its antiderivative is x^2/2" does. Correctness is evaluated separately."""

task_prompt_spec_math_qa = {
        "correctness_cot": CORRECTNESS_CRITERIA + """

Instructions:
1. By default, enclose your reasoning between `<think>` and `</think>`. Keep it concise, with no more than 200 words. If the user does not request reasoning or `</think>` is given, you can omit it.
2. After `</think>`, on a new line, output only a score:
   - 0 if any correctness criterion is unmet.
   - 1 if all correctness criteria are fully met.
   The score must be parsable by Python's `float()` function, with no punctuation or additional text.""",

        "usefulness_cot": USEFULNESS_CRITERIA + """

Instructions:
1. By default, enclose your reasoning between `<think>` and `</think>`. Keep it concise, with no more than 200 words. If the user does not request reasoning or `</think>` is given, you can omit it.
2. After `</think>`, on a new line, output only a score between 0 and 1:
   - 0 if the step is irrelevant or makes no new reasoning progress.
   - 1 if the step adds substantial concrete progress toward the requested answer.
   - A value strictly between 0 and 1 if the step is partially useful. Larger values indicate more usefulness.
   The score must be parsable by Python's `float()` function, with no punctuation or additional text.""",

       "correctness": CORRECTNESS_CRITERIA + """

Instructions:
Output only a score:
- 0 if any correctness criterion is unmet.
- 1 if all correctness criteria are fully met.
The score must be parsable by Python's `float()` function, with no punctuation or additional text.""",

    "usefulness": USEFULNESS_CRITERIA + """

Instructions:
Output only a score between 0 and 1:
- 0 if the step is irrelevant or makes no new reasoning progress.
- 1 if the step adds substantial concrete progress toward the requested answer.
- A value strictly between 0 and 1 if the step is partially useful. Larger values indicate more usefulness.
The score must be parsable by Python's `float()` function, with no punctuation or additional text."""
}

TOOL_SCORE_INSTRUCTION = """Submit exactly one submit_reward tool call, not an ordinary text response.
Its score must be numeric: correctness is 0 if any criterion fails and 1 otherwise;
usefulness is between 0 and 1, from no new progress to substantial concrete progress."""

SCORE_ONLY_INSTRUCTION = """The tool arguments must contain ONLY the score dictionary, for example {"score": 1}.
Do not generate explanations, calculations, reasoning, Markdown, or <think> tags.
Do not output any ordinary text before or after the tool call.
Do not print the dictionary or a function invocation as text; submit it as native tool arguments."""

SCORE_REASONING_INSTRUCTION = """Include only reasoning (a concise string) and score in the tool arguments.
Put all explanation inside reasoning; do not output ordinary text before or after the tool call."""

# | 配置 | 读取的键 |
# |---|---|
# | 正确性评分，thinking 关闭 | `correctness` |
# | 正确性评分，thinking 开启 | `correctness_cot` |
# | 有用性评分，thinking 关闭 | `usefulness` |
# | 有用性评分，thinking 开启 | `usefulness_cot` |
task_prompt_spec_tool = {
    "correctness": CORRECTNESS_CRITERIA + "\n\n" + TOOL_SCORE_INSTRUCTION + "\n" + SCORE_ONLY_INSTRUCTION,
    "correctness_cot": CORRECTNESS_CRITERIA + "\n\n" + TOOL_SCORE_INSTRUCTION + "\n" + SCORE_REASONING_INSTRUCTION,
    "usefulness": USEFULNESS_CRITERIA + "\n\n" + TOOL_SCORE_INSTRUCTION + "\n" + SCORE_ONLY_INSTRUCTION,
    "usefulness_cot": USEFULNESS_CRITERIA + "\n\n" + TOOL_SCORE_INSTRUCTION + "\n" + SCORE_REASONING_INSTRUCTION,
}
