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

The policy and reward-model prompts omit the answer key, but the environment's
`goal_check` compares each board with reference answers. The resulting progress
contributes to the search reward in `EnvGroundedPRM.calculate_reward`. This
privileged environment feedback is retained for reproduction and limits any
claim about deployment on puzzles without an answer key.
