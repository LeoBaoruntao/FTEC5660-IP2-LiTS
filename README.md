# Individual Project 2 — LiTS reproduction

This repository reproduces one archived mini-crossword result from LiTS and compares 30 versus 10 MCTS iterations. The upstream project is [xinzhel/lits-llm](https://github.com/xinzhel/lits-llm), pinned at `25e94d862cb4514c22d9a80a635e392461769df7`. Its original README is preserved as [README_UPSTREAM.md](README_UPSTREAM.md). The original Apache license is retained.

## Deliverables

- [Report PDF](submission/Reproducibility_Report.pdf), fewer than 10 pages.
- [Presentation slides](submission/Presentation.pptx) and [English speaking script](submission/Presentation_Script.md).
- [Measured trial table](reproduction/analysis/trials.csv) and [full statistics](reproduction/analysis/summary.json).
- [Protocol](reproduction/PROTOCOL.md), [debug diary](reproduction/DEBUG_DIARY.md), and raw evidence in `reproduction/runs/`.
- Submission repository: [LeoBaoruntao/FTEC5660-IP2-LiTS](https://github.com/LeoBaoruntao/FTEC5660-IP2-LiTS).
- Submit the recorded Zoom presentation link separately through the course page. Slides are not a recording.

## Setup

Tested on Windows with Python 3.14. Use Python 3.12 or newer (the pinned upstream source includes syntax newer than its advertised minimum).

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r reproduction/environment-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Create an ignored `.env` file containing `DEEPSEEK_API_KEY=your_key`. Never commit it. No other account, local database, or GPU is needed for this narrow task. The full upstream package lists unrelated dependencies; the API-only installation deliberately does not install them. Do not interpret it as a supported installation for every LiTS feature.

## Run and evaluate

The model is `deepseek-v4-flash` at `https://api.deepseek.com`, with thinking disabled. The runner uses the original MCTS, crossword environment and prompts. Only `n_iters` differs between the two experimental conditions.

```powershell
# Inspect configuration without API calls
.\.venv\Scripts\python.exe reproduction/run_one.py --iterations 2 --seed 41 --output reproduction/new_dry_check --dry-run

# Single baseline run; choose a NEW output directory
.\.venv\Scripts\python.exe reproduction/run_one.py --iterations 30 --seed 41 --output reproduction/new_trial
$env:PYTHONPATH = (Join-Path $PWD 'demos')
.\.venv\Scripts\python.exe -X utf8 -m lits.cli.eval_search --result_dir reproduction/new_trial

# Verify and re-summarize the supplied evidence without API calls
.\.venv\Scripts\python.exe reproduction/check_results.py
.\.venv\Scripts\python.exe reproduction/summarize.py
```

For a completely new six-trial batch, use a separate checkout and move its existing `reproduction/runs` directory to an archive first. Then run `python reproduction/run_experiments.py`. The script refuses to overwrite results. It runs local seeds 41, 42 and 43, interleaves the two conditions, and permits three simultaneous processes. It makes paid hosted API calls. The included final data can be inspected offline instead.

## Interpretation

The final six trials produced the following results on the same puzzle. Standard deviations are sample standard deviations over three trials per condition.

| Condition | Mean word accuracy (SD) | Exact solutions | Mean model calls | Mean seconds |
| --- | ---: | ---: | ---: | ---: |
| Baseline: 30 iterations | 86.7% (11.5 percentage points) | 1/3 | 335.7 | 416.2 |
| Modified: 10 iterations | 66.7% (23.1 percentage points) | 0/3 | 317.3 | 378.0 |

Reducing the iteration budget lowered mean word accuracy by 20 percentage points while reducing mean calls by only 5.5%. The small sample and substantial variation do not support a general performance claim; timing also includes hosted-service latency and concurrent runs.

The original archived demonstration reports 60% word accuracy on archived query 0 (current dataset index 2), with an unsolved full puzzle, Claude 3.5 Sonnet and LiTS 0.3.1. This study uses DeepSeek and LiTS 0.4.0. Consequently, comparison with the archived score is contextual, not a causal comparison of models. The controlled comparison is between the two budgets within this study.

The environment uses answer-key-based progress in search rewards, even though the model prompts omit the key. This privileged feedback is retained and disclosed. Three repetitions on one puzzle do not establish full-dataset performance. Local seeds do not guarantee deterministic provider responses. Estimated dollar costs printed by upstream are not treated as verified DeepSeek bills; use the measured token counts.

## Files and provenance

Raw run folders contain `config.json`, `run_manifest.json`, `inferencelogger.log`, checkpoints, terminal nodes and `eval_results.json`. The summary contains SHA-256 hashes. `failed_encoding_batch` retains the interrupted Windows-encoding batch; pilots and failures are excluded from the final trial table. Code changes are separated into commits for setup, compatibility repairs, experiment tooling and final materials.

Do not upload `.env`, `.venv`, `__pycache__` or local build caches. Keep the original license and upstream attribution. Upload the report and two required links through the course page, and confirm the exact deadline shown there.
