# Reproduction protocol

Recorded before collecting final results on 1 November 2026.

## Target

Reproduce the archived LiTS mini-crossword demonstration at upstream commit
`25e94d862cb4514c22d9a80a635e392461769df7`.
The original `demos/demo_results/eval.log` reports 60% word accuracy and 0%
whole-puzzle accuracy on puzzle index 0, using Claude 3.5 Sonnet and LiTS 0.3.1.
The checked-out implementation identifies itself as 0.4.0. This is a narrow
repository-result reproduction, not a reproduction of a paper-wide benchmark.

## Common conditions

Use upstream MCTS, crossword transition, policy prompts, reward prompts and
evaluation. Use the bundled `demos/crosswords/data/mini0505.json`, puzzle 0.
Run three independent trials for each condition, with local seeds 41, 42, 43.
Local seeds do not guarantee deterministic hosted API responses.
Use `deepseek-v4-flash` at `https://api.deepseek.com`, thinking disabled.
Model and software version differ from the archived run, so original and new
scores are contextual comparisons, not evidence of a causal model advantage.

## Isolated modification

Baseline: 30 MCTS iterations, as in the archived config.
Modification: 10 MCTS iterations. Hold all other experiment settings fixed:
3 candidate actions, 10 maximum steps, 10 rollout steps, 1024 maximum output
tokens, no memory, no continuation. Retain upstream component decoding defaults.

## Measures

Primary: correctly filled words / 10 on the output selected by upstream scoring.
Secondary: exact solved puzzle, API calls, input/output tokens and wall time.
Report all three outcomes and sample mean, sample variance, and standard deviation.
Do not select a trial based on its ground-truth score. Keep pilot runs separate.
No accuracy improvement is required or assumed. These three trials on one puzzle
cannot establish general performance across crossword puzzles.

## Adaptations shared by both conditions

Remove an unused unconditional torch import for hosted-API execution.
Add DeepSeek's explicit thinking-disabled request parameter to the existing
OpenAI-compatible model wrapper. Do not replace the search algorithm.
Document additional compatibility fixes if discovered during the pilot.
