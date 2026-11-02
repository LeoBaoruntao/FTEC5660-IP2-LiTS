# Debug diary

## Environment preparation

The full upstream dependency list includes local-model GPU libraries, spatial
databases, and several memory providers. The selected hosted-API crossword path
does not use those features. A separate virtual environment and a minimal
dependency list were used; the original project metadata remains intact.

An initial install included aioboto3 alongside boto3. Pip spent time
backtracking over incompatible AWS SDK versions. That install was interrupted,
and aioboto3 was omitted because the synchronous OpenAI-compatible experiment
does not import the asynchronous Bedrock provider.

The command-line path also imported IPython unconditionally just to detect a
notebook. A fallback now returns no active shell when IPython is absent.
Requests was added to the minimal dependency list because tool utilities import
it. Pip's full-project dependency warnings are expected with the explicitly
documented API-only install; this is not a full local-model installation.

## Hosted model adaptation

`lits/config.py` imported torch but never referenced it. Removing that import
lets the existing lazy-loading approach work for hosted-only runs. The
OpenAI-compatible wrapper already supports a configurable base URL; a small
provider-specific branch sends DeepSeek's thinking-disabled option. These
adaptations apply equally to both experimental conditions.

## Scope and evaluation audit

The archived configuration says package version 0.3.1, while the pinned current
checkout says 0.4.0. Both this version difference and the Claude-to-DeepSeek
change must be disclosed, rather than treating the archived run as a perfectly
matched control.

The pilot initially returned empty final messages with 1,024 output tokens.
The upstream search defaults enabled thinking, so the provider-specific
non-thinking option was never applied. Explicitly disabling all three search
thinking flags fixed this. The corrected two-iteration pilot produced 30%
word accuracy; it is excluded from the prespecified final six trials.

The upstream inference report prints an estimated dollar cost using default
pricing. That is not a verified DeepSeek bill, so the report uses measured
calls and token counts rather than reporting those dollar estimates.

The policy and reward-model prompts omit the answer key, but the environment's
`goal_check` compares each board with reference answers. The resulting progress
contributes to the search reward in `EnvGroundedPRM.calculate_reward`. This
privileged environment feedback is retained for reproduction and limits any
claim about deployment on puzzles without an answer key.

## Archived configuration drift

The archived `terminate_on_terminal_node` argument is rejected by LiTS 0.4.0.
The runner omits this removed flag and explicitly keeps
`terminate_on_first_solution=false` in both conditions. This is another reason
the archived 0.3.1 result is a contextual reference, not an exact replication.

## Windows encoding and false success

The first full batch encountered a GBK UnicodeEncodeError while saving a
generated checkmark in a checkpoint. Upstream catches per-task failures and can
still return zero; its evaluator can likewise return zero without a result.
That entire batch is preserved as `failed_encoding_batch`. On 2 November all
six trials were restarted with Python UTF-8 mode. A run is now accepted only
when the evaluator creates `eval_results.json`; offline checks also inspect
the execution log. Infrastructure failures were not selected by accuracy.

## Dataset identity mismatch

Archived query 0 has the board AMIGO / MAMOU / INPUT / SIEGE / HALER, which is
current dataset row 2. Current row 0 begins AGEND. The index-0 exploratory
batch was stopped and preserved in `exploratory_index0`; its scores do not
count as same-puzzle reproduction. The final comparison was restarted on
content-matched row 2 with the same budgets and seeds. This correction was
based on data identity, not on which puzzle produced a better score.
