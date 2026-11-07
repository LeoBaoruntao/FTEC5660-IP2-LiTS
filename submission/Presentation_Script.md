# English presentation script

Read only the spoken paragraphs. Square-bracket lines are stage directions and should not be read aloud. Rehearse once, aiming for 10–12 minutes including pauses. Stay below the 15-minute limit.

## Slide 1: Reproducing LiTS

[SLIDE 1 ON SCREEN. Approx. 30 seconds.]

Hello. In this project, I test LiTS, a system that uses a language model to search through possible answers. I focus on one small crossword from the authors' saved demo. I then make one change: I reduce the search budget from thirty iterations to ten. My main question is simple. Can the smaller budget save work without losing too much accuracy?

[CLICK ONCE to show Slide 2: Project scope. Then continue.]

## Slide 2: Project scope

[SLIDE 2 ON SCREEN. Approx. 40 seconds.]

I chose LiTS because its public code includes a crossword task, saved results, and an evaluation script. This gives me a clear starting point. I use the authors' code instead of building a new system. First, I run the same puzzle and measure my own results. Then I compare two search budgets under the same setup. I keep the scope small so I can check the data and explain the failures in detail. This study covers one puzzle, with three runs for each budget.

[CLICK ONCE to show Slide 3: The archived target. Then continue.]

## Slide 3: The archived target

[SLIDE 3 ON SCREEN. Approx. 45 seconds.]
[Point to the reference grid and the 60% score.]

The saved demo reports sixty percent word accuracy. That means six of the ten words were correct. The full puzzle was not solved. The grid on the left shows the reference answer used to identify the task. There is an important data detail here. The old result calls this query zero, but the matching puzzle is now at index two. I checked the actual board rather than relying on its number. The old demo uses Claude and an older LiTS version, so its score is only a reference for my new runs.

[CLICK ONCE to show Slide 4: How the agent searches. Then continue.]

## Slide 4: How the agent searches

[SLIDE 4 ON SCREEN. Approx. 45 seconds.]

The model first suggests a word and a place to put it. The environment then updates the board and checks the result. The search tries possible next moves and uses their scores to choose where to explore. This repeats across several steps. MCTS means Monte Carlo Tree Search. Here, it helps the system compare different paths through the puzzle. This is why LiTS is an agentic system: later actions depend on earlier actions and feedback. One prompt does not produce the whole result.

[CLICK ONCE to show Slide 5: The test setup. Then continue.]

## Slide 5: The test setup

[SLIDE 5 ON SCREEN. Approx. 45 seconds.]

I run the code on Windows through a hosted DeepSeek API. The model is deepseek-v4-flash, with thinking disabled. Both groups use this same model and the same puzzle. The word generator uses a temperature of zero point seven, with the original retry rule still active. I also keep the same prompts and evaluator. I save the settings, raw logs, and final output for each run. The local seed labels are forty-one, forty-two, and forty-three. These labels help track the runs, but they cannot make a hosted model fully repeatable.

[CLICK ONCE to show Slide 6: One change in the search budget. Then continue.]

## Slide 6: One change in the search budget

[SLIDE 6 ON SCREEN. Approx. 40 seconds.]

The baseline allows thirty search iterations. The modified version allows ten. The question is whether fewer iterations can keep most of the accuracy while using fewer model calls. I keep the other search settings the same, including the action count and maximum steps. The switch from Claude to DeepSeek applies to both groups. It is separate from the budget test. An iteration is not the same as one API call. The search may make several calls during one iteration, and different paths can stop at different points.

[CLICK ONCE to show Slide 7: How I measure the result. Then continue.]

## Slide 7: How I measure the result

[SLIDE 7 ON SCREEN. Approx. 40 seconds.]

I use two accuracy measures. Word accuracy is the number of correct words divided by ten. Exact accuracy asks whether the whole puzzle is correct. A run with eight correct words has eighty percent word accuracy, but its exact score is still zero. I also count model calls, tokens, and elapsed time. These help show the work needed to get the answer. I report all three final runs for each group, along with their mean and spread. I do not select only the best run.

[CLICK ONCE to show Slide 8: Accuracy falls with the smaller budget. Then continue.]

## Slide 8: Accuracy falls with the smaller budget

[SLIDE 8 ON SCREEN. Approx. 50 seconds.]
[Point to the blue bar, then the orange bar.]

The blue bars show thirty iterations. The orange bars show ten. The average word score falls from eighty-six point seven percent to sixty-six point seven percent. That is a drop of twenty percentage points. The larger budget solves the full puzzle in one of three runs. The smaller budget solves it in none of the three runs. The thirty-iteration scores are also closer together in this small sample. However, three runs on one puzzle are too few to claim that the same gap would hold across many puzzles.

[CLICK ONCE to show Slide 9: All six runs. Then continue.]

## Slide 9: All six runs

[SLIDE 9 ON SCREEN. Approx. 50 seconds.]
[Point to the seed 42 row.]

This table shows every final run. Please look at the row for seed forty-two. Both groups reach eighty percent word accuracy, but the smaller budget uses more model calls in that run. This is a useful reminder that a lower iteration limit does not always mean fewer calls. The search follows different paths, and model responses can vary. The seed labels do not make each pair identical. I keep these uneven results in the table because they are part of the evidence. The full raw records are also saved in the repository.

[CLICK ONCE to show Slide 10: The saving in calls is small. Then continue.]

## Slide 10: The saving in calls is small

[SLIDE 10 ON SCREEN. Approx. 45 seconds.]
[Point to calls first, then time.]

The average call count falls from about three hundred and thirty-six to three hundred and seventeen. This saves only about five point five percent of calls. Average time falls from about four hundred and sixteen seconds to three hundred and seventy-eight seconds. Timing includes network delays and up to three runs taking place at once, so it is a rough measure. I use recorded calls and token counts instead of treating the tool's default dollar estimate as a real bill. On this puzzle, the small saving comes with a much larger loss in accuracy.

[CLICK ONCE to show Slide 11: A limit in the reward design. Then continue.]

## Slide 11: A limit in the reward design

[SLIDE 11 ON SCREEN. Approx. 55 seconds.]

The most important limit comes from the reward code. The prompts do not give the model the answer key. However, the environment checks progress using the reference answers, and that feedback affects the search. In other words, the system gets help from information that a real solver would not normally have. I keep this behavior because it is part of the authors' setup, and I clearly report it. The result shows how this repository behaves under its own rules. It does not show that the system can solve new crosswords with no answer-based feedback.

[CLICK ONCE to show Slide 12: Problems found during the work. Then continue.]

## Slide 12: Problems found during the work

[SLIDE 12 ON SCREEN. Approx. 55 seconds.]

Several problems appeared before the final test. Unused imports made the small API setup depend on extra packages. I removed those startup blockers. Thinking mode also used the short answer budget, so I disabled it for both groups. Windows text encoding caused some output files to fail even when the process returned a normal exit code. I switched to UTF-8 and checked that evaluation files really existed. Finally, I found the puzzle index mismatch. I kept the failed and exploratory batches, excluded them from the final results, and ran the matching puzzle again under the corrected setup.

[CLICK ONCE to show Slide 13: What the evidence supports. Then continue.]

## Slide 13: What the evidence supports

[SLIDE 13 ON SCREEN. Approx. 45 seconds.]

The project gives me a working reproduction of the saved task and a measured budget comparison. In these runs, ten iterations offer a weak trade-off: lower accuracy with only a small saving in calls. The comparison with the old sixty percent score remains limited because the model and software version changed. A stronger next study would use more puzzles and more runs. I would also test the effect of removing answer-based reward feedback as a separate change. That would help show how much the current reward design contributes to the result.

[CLICK ONCE to show Slide 14: Conclusion and evidence. Then continue.]

## Slide 14: Conclusion and evidence

[SLIDE 14 ON SCREEN. Approx. 35 seconds.]

My main lesson is that reproducing a result takes more than getting the code to start. The exact task, model settings, saved outputs, and scoring rules all matter. For this puzzle, reducing the budget did not give the balance I hoped for. The repository includes the run scripts, six final results, and a debug diary. The report explains the limits as well as the numbers. Thank you for listening.

[END. Stop the recording after a short pause.]

