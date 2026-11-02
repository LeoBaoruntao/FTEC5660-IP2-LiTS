"""Create report and recording material from measured summaries, never invented values."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission'; OUT.mkdir(exist_ok=True)
d=json.loads((ROOT/'reproduction/analysis/summary.json').read_text())
b=d['groups']['baseline']; m=d['groups']['reduced_budget']
def mean(g,k): return g[k]['mean']
def pct(x): return f'{x*100:.1f}%'
def stats(g,k): return f"{g[k]['mean']:.3f} (SD {g[k]['sample_sd']:.3f})"
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9))
styles['Title'].fontSize=21; styles['Title'].leading=26
story=[]
def p(t): story.append(Paragraph(t,styles['Body']))
def h(t): story.append(Paragraph(t,styles['Heading2']))
def table(rows,widths):
    t=Table([[Paragraph(str(c),styles['Body']) for c in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story.append(t)
def page(): story.append(PageBreak())
story.append(Paragraph('Reproducing LiTS on a Mini Crossword',styles['Title']))
p('Individual Project 2 | FTEC5660 | Reproducibility Work | 2 November 2026')
h('1  Target and scope')
p('This project reproduces one archived result from LiTS, a framework for multi-step language-model search. The target is the archived demonstration labelled query 0, matched by board content to index 2 in the current bundled dataset. The repository reports 60% word accuracy and 0% exact-puzzle accuracy for this demonstration, with 133 model calls. I use the authors\' search code and evaluation rather than implementing a substitute solver. The small modification reduces the MCTS iteration budget from 30 to 10.')
p(f"Across three new trials, the 30-iteration condition achieved {pct(mean(b,'word_accuracy'))} mean word accuracy, compared with {pct(mean(m,'word_accuracy'))} for 10 iterations. These are measurements on one puzzle, not estimates of accuracy across the full dataset. The model and LiTS version differ from the archived run, so the original number is a contextual reference rather than a matched control.")
h('Why this is an agentic system')
p('The policy proposes a word placement, the environment updates the board, and a reward model evaluates the proposal. MCTS selects and expands paths, simulates continuations, and backpropagates rewards before choosing subsequent actions. Later calls depend on earlier actions and observations. This is a multi-step search process, not a single prompt-response test.')
h('Source and provenance')
p('Repository: https://github.com/xinzhel/lits-llm<br/>Pinned upstream commit: 25e94d862cb4514c22d9a80a635e392461769df7<br/>Original evidence: demos/demo_results/config.json and eval.log.<br/>Data: demos/crosswords/data/mini0505.json, index 2 of 156 puzzles (the archived result calls it query 0).')
p('The archived configuration identifies LiTS 0.3.1 and Claude 3.5 Sonnet (20240620). The checked-out package is 0.4.0. The protocol was written before the final trials; failed setup runs, the two-iteration pilot and an initial index-0 exploration are kept separate from the six matched-puzzle observations.')
page(); h('2  Setup and experimental design')
p('Experiments ran on Windows using Python 3.14 in a dedicated virtual environment. Inference used the hosted DeepSeek API; no local GPU or model weights were required. Exact installed versions are saved in reproduction/environment-lock.txt. An API-only dependency list avoids unrelated local-model, geospatial and memory packages. This installation supports the selected task, not every LiTS feature. Policy temperature starts at 0.7 and increases by 0.2 on retries, capped at 1.0; policy top_p is 0.95. Reward sampling uses temperature 0.6 and the wrapper default top_p of 1.0. Policy output is capped at 1,024 tokens; binary reward outputs are capped at 10 tokens.')
table([['Setting','Value'],['Model and endpoint','deepseek-v4-flash; DeepSeek; https://api.deepseek.com'],['Generation','Thinking disabled; maximum 1,024 output tokens; upstream policy/reward sampling settings retained'],['Search','MCTS; 3 candidate actions; depth 10; rollout length 10; no memory or continuation'],['Controlled change','n_iters = 30 versus n_iters = 10'],['Repetition','Three trials per condition; local seeds 41, 42, 43'],['Data','Same puzzle and dataset hash in all six runs']], [150,345])
p('The seeds control local Python and NumPy randomness. They are not sent as a deterministic sampling guarantee to the hosted provider. Runs are interleaved across conditions with up to three active processes. Consequently wall time includes network latency and possible service contention; token counts provide a more stable resource measure. Maximum output length is a cap, not the number of tokens requested on every call.')
p('Prompts and the search algorithm are retained from the pinned checkout. Both conditions share the same compatibility adaptations: remove an unused torch import, tolerate missing IPython outside notebooks, use DeepSeek\'s thinking flag, omit a removed legacy search flag, and force UTF-8 file handling. These are not additional experimental treatment differences.')
h('3  Metrics and evaluation')
p('Word accuracy is the number of correct horizontal and vertical words divided by ten. Exact accuracy is one only when the complete board matches all ten answers. The original LiTS evaluation selects the output using its search returns, then applies the crossword goal checker. The study does not choose the best trial after seeing the answer key. Calls and tokens are summed from inferencelogger.log; elapsed time comes from run_manifest.json. Sample variance and standard deviation use n - 1 = 2 in the denominator.')
page(); h('4  Results')
rows=[['Condition','Seed','Words','Exact','Calls','Seconds']]
for r in d['trials']: rows.append([str(r['iterations'])+' iterations',r['seed'],pct(r['word_accuracy']),pct(r['exact_accuracy']),r['calls'],f"{r['elapsed_seconds']:.1f}"])
table(rows,[110,45,70,70,65,135])
story.append(Spacer(1,12))
table([['Measure','30 iterations','10 iterations'],['Mean word accuracy',pct(mean(b,'word_accuracy')),pct(mean(m,'word_accuracy'))],['Word accuracy SD',f"{b['word_accuracy']['sample_sd']:.4f}",f"{m['word_accuracy']['sample_sd']:.4f}"],['Word accuracy sample variance',f"{b['word_accuracy']['sample_variance']:.5f}",f"{m['word_accuracy']['sample_variance']:.5f}"],['Exact-puzzle rate',pct(mean(b,'exact_accuracy')),pct(mean(m,'exact_accuracy'))],['Mean API calls',f"{mean(b,'calls'):.1f}",f"{mean(m,'calls'):.1f}"],['Mean input tokens',f"{mean(b,'input_tokens'):.0f}",f"{mean(m,'input_tokens'):.0f}"],['Mean output tokens',f"{mean(b,'output_tokens'):.0f}",f"{mean(m,'output_tokens'):.0f}"],['Mean elapsed seconds',f"{mean(b,'elapsed_seconds'):.1f}",f"{mean(m,'elapsed_seconds'):.1f}"]],[215,140,140])
p(f"Relative to the archived 60% word score, the new baseline differs by {(mean(b,'word_accuracy')-.6)*100:+.1f} percentage points. Reducing the budget changes mean word accuracy by {(mean(m,'word_accuracy')-mean(b,'word_accuracy'))*100:+.1f} percentage points and mean API calls by {mean(m,'calls')-mean(b,'calls'):+.1f}. These differences describe this experiment; three trials on a single puzzle do not establish a general benefit or harm.")
p('The archived run used 55,630 input and 590 output tokens. Its displayed dollar estimate is not reused as a DeepSeek cost. This report makes no monetary claim without verified provider billing. Raw trial rows and sample statistics for every resource metric are supplied in trials.csv and summary.json.')
page(); h('5  What the results can and cannot establish')
p('The repository workflow can be executed with a hosted alternative model, and the same task can be scored automatically. Matching the archived number would not prove that the original experiment had been recreated exactly: the model changed, the package evolved, and the archived record contains only one run. A different result likewise cannot be attributed to a single cause. The within-study comparison is stronger because both conditions share the model, dataset and code, with only the iteration limit changed.')
p('A particularly important assumption appears in the environment reward. The policy and reward-model prompts do not include the answer key, but CrosswordsTransition.goal_check compares intermediate boards with the reference. EnvGroundedPRM then uses that progress in its reward. Search therefore benefits from privileged feedback. I retain it to reproduce the supplied setup, but the score should not be presented as performance on unseen crossword answers without such feedback.')
p('Repeated or invalid actions remain in the logs and contribute to the measured cost. More iterations do not necessarily create more useful exploration. The model can revisit the same proposal or produce explanatory text when a short action is required. This helps explain why iteration budget, call count and accuracy need to be measured separately.')
h('6  Debug diary')
p('A data identity check found that archived query 0 matches current dataset row 2 by its full answer board. I preserved the earlier row-0 exploration and restarted all six final trials on the matching row. The protocol amendment records this correction; no trial was chosen because of a favourable score.')
p('Installation first encountered AWS dependency backtracking. Removing the unused asynchronous provider from the API-only dependency list resolved that setup problem. Optional imports of torch and IPython also prevented lightweight execution and were adjusted without changing search behavior.')
p('The archived terminate_on_terminal_node option was rejected by the newer version. It was omitted, while terminate_on_first_solution remained explicitly false. A further pilot produced empty final answers because thinking consumed the output budget; explicitly disabling the policy, evaluator and terminal-model thinking flags allowed usable actions. The corrected pilot scored 30% and is not counted as a final trial.')
p('The first full batch exposed a Windows GBK encoding error when generated text contained a checkmark. Upstream swallowed the task exception and returned zero, and the evaluator could also return without a score file. The batch was archived and all six trials restarted under UTF-8. The runner now requires an evaluation artifact. Excluding these infrastructure failures was a documented batch-level decision, not selection based on accuracy.')
page(); h('7  Reproduction instructions and conclusion')
p('From the repository root, create a Python virtual environment and install reproduction/environment-lock.txt, then run pip install -e . --no-deps. Put DEEPSEEK_API_KEY in an ignored .env file. The run scripts never print or commit the key. See README.md for exact commands and output locations.')
p('Run reproduction/run_experiments.py for the six trials, reproduction/check_results.py for offline consistency checks, and reproduction/summarize.py to regenerate the numerical tables. Existing run directories are intentionally not overwritten: use a separate clone for a new batch. Each run retains its configuration, checkpoints, evaluation result, per-call usage and manifest. The summary includes SHA-256 hashes of its source files.')
p('The main lesson is that reproducibility depends on the execution and evaluation contract as much as on the model. A successful exit code was insufficient evidence of a completed run. Small changes in model configuration could prevent actions from being produced at all. For future users, I recommend a short pilot, explicit encoding and provider settings, artifact checks, and a clearly defined comparison before spending resources on a larger benchmark.')
p('A useful follow-up would repeat the budget comparison on a prespecified sample of additional puzzles. A separate experiment could remove answer-key-based reward shaping, but combining that change with the current budget change would make interpretation harder. Neither follow-up is claimed as completed here.')
h('References and evidence')
p('[1] LiTS source and README, pinned commit 25e94d862cb4514c22d9a80a635e392461769df7, https://github.com/xinzhel/lits-llm.<br/>[2] Archived demonstration: demos/demo_results/config.json and eval.log in that commit.<br/>[3] Crossword definitions and prompts: demos/lits_benchmark/crosswords.py; reward implementation: lits/components/reward/env_grounded.py.<br/>[4] This project: reproduction/PROTOCOL.md, DEBUG_DIARY.md, runs/, analysis/trials.csv and analysis/summary.json.')
def footer(c,doc):
    c.setFont('Helvetica',9); c.setFillColor(colors.grey); c.drawString(45,25,'LiTS reproduction | Individual Project 2'); c.drawRightString(550,25,str(doc.page))
SimpleDocTemplate(str(OUT/'Reproducibility_Report.pdf'),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)

slides=[
('Reproducing LiTS on a mini crossword','One archived result, one controlled budget change',
 'I chose LiTS from the instructor list. My scope is deliberately narrow: one archived mini-crossword demonstration, followed by a comparison of thirty and ten MCTS iterations. I use the authors\' code and evaluator. This is not a claim to reproduce every result in the paper.'),
('The target is a repository result','Archived query 0 = current data index 2 | 10 words\nArchived: 60% word accuracy, 0% exact accuracy\nClaude 3.5 Sonnet and LiTS 0.3.1',
 'The original evidence is the configuration and evaluation log committed in demos/demo_results. It reports six of ten words correct and an unsolved full grid. I kept this target fixed rather than choosing a puzzle after seeing my model\'s performance. The current checkout is version 0.4.0, which matters for interpretation.'),
('How the agent works','Propose a word → update the board → score the action\nMCTS selects, expands, simulates and backs up rewards\nThe next action depends on the current board',
 'The agent repeatedly generates actions and observes their consequences. It branches over possible placements and uses simulated continuations to decide where to search. That makes this a multi-step agentic system. The modification leaves the policy, transition, reward code and prompts unchanged.'),
('One experimental change','Baseline: 30 iterations    Modified: 10 iterations\nSame DeepSeek model, puzzle, prompts and evaluator\nThree trials per condition; local seeds 41, 42, 43',
 'DeepSeek replaces the original Claude model because it is available in this environment. That replacement applies to both conditions. My treatment is only the iteration budget. Hosted sampling is not guaranteed deterministic by local seeds. I interleave the conditions and retain every final trial rather than choosing the best one.'),
('Measured results',f"Mean word accuracy: {pct(mean(b,'word_accuracy'))} → {pct(mean(m,'word_accuracy'))}\nMean calls: {mean(b,'calls'):.1f} → {mean(m,'calls'):.1f}\nMean seconds: {mean(b,'elapsed_seconds'):.1f} → {mean(m,'elapsed_seconds'):.1f}\nOrder: 30 iterations → 10 iterations",
 f"The baseline mean word score is {pct(mean(b,'word_accuracy'))}, and the modified score is {pct(mean(m,'word_accuracy'))}. The full report shows every trial and the sample variance. I report token use rather than treating the upstream default dollar estimate as my actual bill. The observed difference is specific to one puzzle and three trials, so I do not claim statistical significance or broad generalization."),
('A hidden evaluation assumption','The prompts omit the answer key\nThe environment still uses reference answers for rewards\nThis is privileged feedback, not blind crossword solving',
 'This is the most important qualification I found in the source. The transition checks board progress against the reference answers, and that progress affects the search reward. I retained it to match the repository setup. Even if a run solves the puzzle, it does not demonstrate an independently deployed solver with no access to answer-based feedback.'),
('What failed and how it was repaired','Optional dependencies blocked API-only setup\nThinking mode exhausted the action output budget\nWindows encoding broke checkpoints despite exit code zero',
 'I removed unused dependency imports and configured the provider explicitly. A pilot then revealed empty final answers until thinking was disabled. The first full batch also failed to write certain Unicode characters under Windows. Content matching subsequently revealed that archived query zero corresponds to current dataset index two. I archived the index-zero exploration and repeated the comparison on the matching puzzle. I preserved that batch, restarted all final trials under UTF-8, and required evaluation files before marking a run complete. These fixes apply to both conditions.'),
('Lessons and next steps','Check artifacts, not only exit codes\nKeep the experimental change isolated\nExpand to more puzzles before making general claims',
 'The project demonstrates an executable, measured reproduction attempt and a controlled budget comparison. It also shows why a repository result depends on versions, provider settings and the evaluator. A next study should use a larger fixed puzzle set, and separately test removal of privileged reward feedback. The repository contains raw logs, configuration, hashes and scripts so future users can inspect and repeat the work.')]
(OUT/'slides_content.json').write_text(json.dumps(slides,indent=2),encoding='utf-8')
notes='# Recorded presentation script\n\nSuggested pace: 8–12 minutes, including a short results-file demonstration. Rehearse and adapt the wording to your own understanding.\n\n'
for i,(title,body,note) in enumerate(slides,1): notes+=f'## Slide {i}: {title}\n\n{note}\n\n'
(OUT/'Presentation_Script.md').write_text(notes.rstrip()+'\n',encoding='utf-8')
print('Report and slide content generated from measured results.')
