# Recorded presentation script

Suggested pace: 8–12 minutes, including a short results-file demonstration. Rehearse and adapt the wording to your own understanding.

## Slide 1: Reproducing LiTS on a mini crossword

I chose LiTS from the instructor list. My scope is deliberately narrow: one archived mini-crossword demonstration, followed by a comparison of thirty and ten MCTS iterations. I use the authors' code and evaluator. This is not a claim to reproduce every result in the paper.

## Slide 2: The target is a repository result

The original evidence is the configuration and evaluation log committed in demos/demo_results. It reports six of ten words correct and an unsolved full grid. I kept this target fixed rather than choosing a puzzle after seeing my model's performance. The current checkout is version 0.4.0, which matters for interpretation.

## Slide 3: How the agent works

The agent repeatedly generates actions and observes their consequences. It branches over possible placements and uses simulated continuations to decide where to search. That makes this a multi-step agentic system. The modification leaves the policy, transition, reward code and prompts unchanged.

## Slide 4: One experimental change

DeepSeek replaces the original Claude model because it is available in this environment. That replacement applies to both conditions. My treatment is only the iteration budget. Hosted sampling is not guaranteed deterministic by local seeds. I interleave the conditions and retain every final trial rather than choosing the best one.

## Slide 5: Measured results

The baseline mean word score is 86.7%, and the modified score is 66.7%. The full report shows every trial and the sample variance. I report token use rather than treating the upstream default dollar estimate as my actual bill. The observed difference is specific to one puzzle and three trials, so I do not claim statistical significance or broad generalization.

## Slide 6: A hidden evaluation assumption

This is the most important qualification I found in the source. The transition checks board progress against the reference answers, and that progress affects the search reward. I retained it to match the repository setup. Even if a run solves the puzzle, it does not demonstrate an independently deployed solver with no access to answer-based feedback.

## Slide 7: What failed and how it was repaired

I removed unused dependency imports and configured the provider explicitly. A pilot then revealed empty final answers until thinking was disabled. The first full batch also failed to write certain Unicode characters under Windows. Content matching subsequently revealed that archived query zero corresponds to current dataset index two. I archived the index-zero exploration and repeated the comparison on the matching puzzle. I preserved that batch, restarted all final trials under UTF-8, and required evaluation files before marking a run complete. These fixes apply to both conditions.

## Slide 8: Lessons and next steps

The project demonstrates an executable, measured reproduction attempt and a controlled budget comparison. It also shows why a repository result depends on versions, provider settings and the evaluator. A next study should use a larger fixed puzzle set, and separately test removal of privileged reward feedback. The repository contains raw logs, configuration, hashes and scripts so future users can inspect and repeat the work.
