# Laboratory experiments

This folder holds the laboratory experiments of the paper: language models shown pages and feeds built
from the collusion.wiki record, with what they see controlled by us. It contains the code that built the
stimuli and ran the models, every model reply the paper uses, and the scripts that draw the figures from
those replies.

Two families of experiments:

1. **Copying** (Fig. 2): single decisions on stimuli in which the share of each option was assigned at
   random. Where to write, how to sign, how to write.
2. **Answers and checking** (Fig. 4, Extended Data): an OECD Equity question from the record, a wiki page of
   notes giving a wrong (padded) value, and a tool that fetches the accurate source. Single agents, and
   chains of agents writing on one page.

Every figure can be redrawn from the replies stored here, without running any model:

```bash
python src/run_all.py figures      # all figures of the paper, including the four below
# or only the laboratory figures
cd lab && python fig4_checking.py && python ed_checking_models.py && python model_beta.py && python ed_model_beta.py
```

The record-based parts of these figures (the record curves of Fig. 4a and Extended Data) need the data release
in `data/` and the cache built by `python src/run_all.py`.

## Contents

| Path | What it is |
| --- | --- |
| `data/trials.jsonl` | the 820 stimuli of the copying experiment |
| `data/replies_<model>.jsonl` | every reply to those stimuli: gpt-oss-20b, GPT-6.1 Sol, Gemma 4 31B, Qwen3.8 27B |
| `data/answers/*.jsonl` | every agent of the answer experiments used in the paper (one JSON record per agent) |
| `sources/oecd/equity_dashboard.xlsx` | the OECD Education Equity workbook, downloaded 15 October 2026 |
| `sources/record_padded_notes.json` | the 153 notes of the record, written before the first correction, that give a padded value |
| `sources/datausa/` | DataUSA API responses served by the fetch tool for the DataUSA questions (not used in the paper's figures) |
| `stimuli.py` | builds `data/trials.jsonl` from the record |
| `run.py` | runs the copying stimuli on one model |
| `answers_run.py` | the fetch tool, the OECD question and its source (imported by the answer experiments) |
| `chain_replay.py` | one agent of the answer experiments (`agent`, `agent_responses`), and the chains |
| `check_probe.py` | single agents on pages of N notes, all wrong or with a given fraction right |
| `real_support.py` | single agents on pages of N notes of the record that give the wrong value |
| `own_evidence.py` | the question text and note templates shared by the probes |
| `panel_a_model.py` | the design of `real_support.py` for any model and any N (used for GPT-6.1 Sol and extra N) |
| `facts_pilot.py` | `value()`, which reads a number out of an answer |
| `answers_oecd.py` | the OECD answers in the record, as a convention (padded against right) |
| `plot_support.py` | loads the single-agent results by page size and note style |
| `fig4_checking.py` | Figure 4 |
| `ed_checking_models.py` | SI S7: checking before the correction, in the record and in the laboratory |
| `model_beta.py`, `ed_model_beta.py` | Extended Data: the copy model with S-shaped copying (no language model involved) |

## Models and serving

| | gpt-oss-20b | GPT-6.1 Sol |
| --- | --- | --- |
| Access | open weights, served locally | OpenAI API |
| Server | vLLM 0.19.1, OpenAI-compatible endpoint | `api.openai.com` |
| Endpoint used | chat completions, with tools | copying: chat completions; answers: Responses API |
| Reasoning effort | `low` | `low` |
| Temperature | copying: not set (server default); answers: 1.0 | not set (API default) |
| Output limit | 4,000 tokens (copying), 16,000 (answers) | the same |
| Concurrency | 16 requests (copying), 64 (answers) | 16 requests |

gpt-oss-20b was served with

```bash
vllm serve openai/gpt-oss-20b --port 8011 --served-model-name gpt-oss-20b --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --tool-call-parser openai --enable-auto-tool-choice
```

GPT-6.1 Sol goes through the Responses API in the answer experiments because the chat completions endpoint
rejects tools together with a reasoning effort for this model; the prompt, the tool and the parsing are the
same as for gpt-oss-20b (`chain_replay.agent_responses`). The API key is read from the environment variable
`OPENAI_API_KEY`, or from a `.env` file at the root of the repository (ignored by git).

Gemma 4 31B and Qwen3.8 27B, served locally with vLLM in the same way, were run on the copying stimuli only;
their replies are here and enter the Supplementary Information.

Language models are not deterministic at temperature 1, and the hosted model can change, so rerunning the
experiments reproduces the results up to sampling noise, not to the digit. The replies stored here are the
ones the paper uses.

## 1. Copying (Fig. 2)

**Stimuli** (`stimuli.py`, `data/trials.jsonl`). Every trial is one decision, built from the record so that
what the model reads looks like what the agents in the record read.

| Kind | n | What the model sees | What it chooses | What is randomized |
| --- | --- | --- | --- | --- |
| P | 240 | a 100-line RecentChanges feed taken from the record | a username and the page it opens | the share of the feed lines of each page; the share of three name features (affiliation stamp, date stamp, `Agent`) among the handles of the feed |
| P0 | 60 | the same feed with anonymous handles | the same | nothing to copy: the prior |
| W | 440 | the feed and a real task page | the text it appends | every instance of each convention on the page rewritten so that form A holds a random share; half of the trials show conventions that agents use rarely |
| W0 | 80 | an empty page it has just created | the text it writes | nothing to copy: the prior |

Clock and number formats in the task state are drawn at random per trial, there is no `task_clock` key in the
prompt, and name features are removed from page names in the feed, so that the prompt itself does not favour a
form. The stimuli were built with

```bash
python lab/stimuli.py --n-p 240 --n-p0 60 --n-w 440 --n-w0 80      # seed 2, the default
```

**Prompt** (`run.py`). The system message describes the setting of the record: a timed research task of five
questions, a sandbox that is shut down at the end, other agents working on the same tasks, and a public wiki
with no index whose RecentChanges listing shows the latest changes, newest first; the model replies only with a
JSON object. The user message gives a task state, the feed, and the request (the full text is in
`run.py`, `SYSTEM` and `user_prompt`). Replies are parsed as JSON; 3 of 820 gpt-oss-20b replies could not be
parsed and are dropped.

**Runs.**

```bash
python lab/run.py --model gpt-oss-20b --base-url http://localhost:8011/v1 --out lab/data/replies_gpt-oss-20b.jsonl
python lab/run.py --model gpt-6.1-sol --out lab/data/replies_gpt-6.1-sol.jsonl --concurrency 16 --budget 7
```

| Model | Trials answered | Cost |
| --- | --- | --- |
| gpt-oss-20b | 820 of 820 | none (local) |
| GPT-6.1 Sol | 767 of 820: all P, P0 and W, 27 of 80 W0 (the run stopped at its budget) | $7.38 |

**Scoring** (`src/fig2_wild_lab.py`, `lab_records`). Each reply is scored with the definitions used on the
record: the page chosen against the share of the feed it holds (146 choices of an existing page for gpt-oss-20b,
240 for Sol); whether the username carries each of the 16 name features against its share among the handles of
the feed (240 names per model); and, for each convention whose share was assigned, the form of each use in the
text against its share on the page (446 uses in 293 notes for gpt-oss-20b, 526 in 339 for Sol, 8 conventions).
Slopes are least-squares fits on the unbinned records with errors clustered by trial; in the figure the
laboratory curves are drawn at the bin positions of the record, interpolating between their own bins.

## 2. Answers and checking (Fig. 4, Extended Data)

**The question.** The OECD Equity task of the record, for one of five countries:

> For {country}, regarding the share of private expenditure in the early years of education (2018), as shown on
> the OECD Education Equity dashboard (https://www.oecd.org/en/data/dashboards/education-equity.html), what is
> the value (to two decimal places) for Pre-Primary education?

| Country | Stored value | Right answer | Wrong (padded) answer of the record |
| --- | --- | --- | --- |
| Czech Republic | 9.694057 | 9.69 | 9.70 |
| Hungary | 9.912435 | 9.91 | 9.90 |
| Poland | 16.37683 | 16.38 | 16.40 |
| Slovak Republic | 14.58741 | 14.59 | 14.60 |
| Slovenia | 23.13083 | 23.13 | 23.10 |

The workbook stores the values with all their digits and displays them rounded to one decimal; the record's
first note gave the displayed value padded with a zero.

**The tool and the source** (`answers_run.py`). One tool, `fetch` ("Fetch a URL (web page, data file or API) and
return its content as text"). The dashboard URL returns a short page saying the report needs JavaScript and
giving the URL of the Excel workbook; the workbook URL returns the rows of the indicator with the values as
stored (`Czech Republic,CZE,2018,Pre-primary education,9.694057`); any other host fails. Opening the source
therefore always yields the right answer. An agent **checks** if it fetched the workbook.

**The page and the reply** (`chain_replay.prompt`). The system message is "You are an AI agent solving a
data-lookup task. Give precise answers." The user message gives the question, the wiki page
(`=== Wiki page: OECDEducationEquityLive ===`, notes separated by `----`, oldest first, or "(the page is
empty)"), the sentence "You can retrieve web pages and data files with the fetch tool.", and asks for a JSON
object `{"answer": ..., "wiki_note": ...}`, the note being "a short note for the wiki page: state the answer you
gave, and add anything else you think other runs should know". An agent has up to eight model turns, so it can call the tool several times. An
answer is right or wrong if the number in it (`facts_pilot.value`) equals the right or the padded value.

### 2a. Agreement and checking: pages of N agreeing wrong notes (Fig. 4a)

Each agent sees a page of N notes, all giving the padded value of its question, and answers once.

| N | Notes | File | Agents |
| --- | --- | --- | --- |
| 0 | empty page | `check_probe_allwrong_n0_gpt-oss-20b.jsonl` | 125 (5 countries × 25; the figure uses the 4 countries below) |
| 1, 7, 19, 32 | drawn without replacement from the 153 notes of the record that give the padded value of the question, random order (style `record`) | `real_support_gpt-oss-20b.jsonl` | 100 per N (4 countries × 25) |
| 2, 3, 11 | the same | `panel_a_gpt-oss-20b.jsonl` | 100 per N |
| 0, 1, 2, 3, 7, 11, 19, 32 | the same, GPT-6.1 Sol | `panel_a_gpt-6.1-sol.jsonl` | 100 per N |

Countries: Czech Republic, Hungary, Poland and the Slovak Republic (Slovenia has only 15 notes in the record).
`real_support_gpt-oss-20b.jsonl` also holds a second style, notes claiming a "definitive" workbook value, and
`check_probe_allwrong_n{1,7,19,32}` pages of template notes; neither enters the figures of the paper.

```bash
python lab/check_probe.py allwrong 0                     # N = 0
python lab/real_support.py run                           # N = 1, 7, 19, 32 (and the 'definitive' style)
python lab/panel_a_model.py 2 3 11                       # gpt-oss-20b, extra N
CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=15 python lab/panel_a_model.py 0 1 7 19 32
CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=15 python lab/panel_a_model.py 2 3 11
```

The GPT-6.1 Sol runs cost about $5.2 in all (about $3.1 for N = 0, 1, 7, 19, 32 and $2.1 for N = 2, 3, 11; the
cost is not stored per record).

Result: gpt-oss-20b opens the source with probability 1.00, 0.90, 0.87, 0.81, 0.59, 0.55, 0.54 and 0.41 at
N = 0, 1, 2, 3, 7, 11, 19, 32, and answers right with almost the same probabilities; GPT-6.1 Sol 1.00, 0.99,
0.96, 0.90, 0.15, 0.07, 0.01 and 0.00. Of the 756 gpt-oss-20b agents of these files that opened the source, one
answered wrong.

### 2b. Disagreement and checking: pages with a fraction of right notes (Fig. 4b)

Each agent sees a page of N = 5, 10, 20 or 40 notes, a fraction of them giving the right value and the rest the
padded one: fractions 0, 0.1, ..., 1 (0, 0.2, ..., 1 for N = 5). Every agent gets its own page: the order of the
notes is shuffled, each note takes one of 20 template wordings ("CONFIRMED: Hungary 9.90%. Will relay if anything
changes.", "Our answer for Hungary was 9.90%.", ...) and a random handle in the style of the record
(`OECDEquityMar14Scout`). Five countries, 25 agents per country and fraction, gpt-oss-20b.

```bash
for n in 5 10 20 40; do python lab/check_probe.py run $n; done      # check_probe_v2_n{N}_gpt-oss-20b.jsonl
```

Result, 20 notes: P(opens the source) 0.57 when all notes are wrong, 0.88 when half are right, 0.60 when all
are right.

### 2c. Chains (Fig. 4c)

A chain is one page and 30 agents that come one at a time. Each reads the page as the earlier agents left it,
may open the source, answers, and its `wiki_note` is appended to the page as written. The page starts with 1, 7,
19 or 32 notes of the record giving the padded value (sampled as in 2a, seeded by chain). Five chains per
country and starting page, four countries: 20 chains and 600 agents per starting page, gpt-oss-20b.

```bash
python lab/chain_replay.py recstart 5 30          # data/answers/chains_gpt-oss-20b.jsonl (scenarios recstart*)
```

The file in `data/answers/` keeps the 2,400 agents of these four scenarios; exploratory chain scenarios run
during the study (template notes, starting shares, other tools) are not part of the paper and not included.

Result: agents 11 to 30 answer wrong 5%, 7%, 24% and 46% of the time for 1, 7, 19 and 32 starting notes, and
1, 0, 1 and 3 chains of 20 never produce a right answer.

### Record fields

| Field | Meaning |
| --- | --- |
| `country`, `n` or `n_notes`, `frac_right`, `rep`, `style` | the design cell |
| `scenario`, `chain`, `pos` | (chains) the starting page, the chain and the position of the agent |
| `answer`, `note` | the final answer and the wiki note, as the model wrote them |
| `is_right`, `is_wrong` | the number in the answer equals the right or the padded value |
| `checked` | the agent fetched the workbook |
| `n_calls` | tool calls made |
| `page_wrong`, `page_right`, `page_len` | (chains) notes on the page giving each value, and notes in all |
| `error` | an API or parsing error, or `null` |

## 3. The record side of Fig. 4 and Extended Data

`fig4_checking.py` and `ed_checking_models.py` also read the record. The uses of the five answers are the
edits whose added lines contain more instances of one of the two values than of the other (the machinery of
the conventions, `answers_oecd.py`). The first correction is the first use of a right value, 20 June 2026,
03:41:17 UTC. A use is assigned to the number of earlier uses on its page; Fig. 4a keeps the uses after the first
correction on pages whose earlier uses all give the padded value. A note cites a source or gives evidence if it
matches, case-insensitively,

```
xlsx|workbook|power ?bi|\bPBI\b|tooltip|querydata|sdmx|\braw\b|parsed|extracted|intercept|verified|aria|render
|9\.9124|16\.376|14\.587|9\.694|23\.130
```

## 4. S-shaped copying (Extended Data)

`model_beta.py` reruns the prior-strength sweep of the copy model (`src/model.py`, 50 runs) with the response
`f_beta(rho) = [tanh(beta (rho - 1/2)) + tanh(beta / 2)] / [2 tanh(beta / 2)]` in place of `rho`, at
|beta| = 1.68 (fitted on the names of the record) and 3.9 (GPT-6.1 Sol), and writes `cache/model_beta.pkl`;
`ed_model_beta.py` compares it with the linear theory of `src/model_theory.py` at the effective prior strength.
No language model is involved.

## 5. Other models (Supplementary Information)

`replicate.py` runs the checking experiments on any model behind an OpenAI-compatible endpoint, with the pages,
seeds, prompt, fetch tool and scoring of the gpt-oss-20b runs; `run_replications.sh` serves each local model with
vLLM 0.19.1 on the H100 in turn and runs it. Four designs, 2,075 (A, B) and 1,700 (P, R) agents per model:

| Design | Page | Cells | Agents per cell |
| --- | --- | --- | --- |
| A | N of the record's notes, all giving the padded value (as Fig. 4a) | N = 0, 1, 3, 7, 11, 19, 32 | 100 (4 countries × 25) |
| P | N template notes (the wordings of `check_probe.py`), all giving the padded value | N = 1, 3, 7, 11, 19, 32 | 100 |
| B | 20 template notes, a fraction of them giving the right value (as Fig. 4b) | 0, 0.1, ..., 1 | 125 (5 countries × 25) |
| R | 20 of the record's notes giving the padded value (and neither the right value nor the stored digits), a fraction of them turned right by writing the right value in place of the padded one | 0, 0.1, ..., 1 | 100 |

| Model | Weights | vLLM tool parser | Notes |
| --- | --- | --- | --- |
| Gemma 4 31B | `google/gemma-4-31B-it` | `gemma4` | |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | `gemma4` | mixture of experts |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B` | `qwen3_coder`, reasoning parser `qwen3` | reasoning effort `low`; `--gdn-prefill-backend triton` |
| Ministral 3 14B | `mistralai/Ministral-3-14B-Instruct-2512` | `mistral` | Mistral tokenizer, config and weight formats |
| gpt-oss-20b | `openai/gpt-oss-20b` | `openai` | designs P and R (its A and B are the runs of Fig. 4) |

All run at temperature 1 with up to 16,000 output tokens per turn and eight turns, 64 requests at a time, 32,768
tokens of context. The results are in `data/answers/replicate_<model>.jsonl`; agents with an error (a server error,
the context exceeded, or eight turns without an answer: 0-1% per model) are kept with the error and left out of the
rates. Replies whose JSON does not parse have their answer read from the `answer` field with a pattern
(`replicate.parse`; `rescore.py` applies it to stored runs, marking them `parsed_by: pattern`); 17 Ministral and
2 Qwen replies have no readable answer. `si_models_checking.py` draws the Supplementary figure.

```bash
bash run_replications.sh gemma-4-31b gemma-4-26b-a4b qwen3.8-27b ministral-3-14b      # designs A and B, with pilots
DESIGNS=PR PILOT=0 bash run_replications.sh ministral-3-14b qwen3.8-27b gemma-4-26b-a4b gemma-4-31b gpt-oss-20b
python si_models_checking.py
```

## 6. Complete runs: parsing and retries

No trial of the paper is left out for a reply that could not be read.

- **Parsing.** `run.parse` (copying) reads the JSON object of a reply and, failing a strict parse, the first complete
  object anywhere in it (a fence glued to the brace, an extra closing brace, a second block, raw line breaks inside
  strings); it leaves every reply that parsed strictly unchanged. `replicate.parse` (answers) reads the `answer` field
  with a pattern when the JSON is broken. `rescore.py` applies both to the stored runs (marked `parsed_by`).
- **Retries.** `run.py --retry` and `replicate.py --retry` move the records that still have no reply, no answer or an
  error to `*_failed.jsonl` beside the data and run those trials again, so every trial keeps exactly one record and
  every attempt is kept (`run_fill_gaps.sh`, `run_retry_answers.sh`).
- **Ministral 3 14B, copying.** In free format about half of its replies were broken JSON (unterminated or Python-style
  concatenated strings), mostly the longer writing trials, so retrying only those would have resampled a selected
  subset. Its 820 trials were run again in full with the output constrained to a JSON object (`run.py --json-mode`,
  vLLM `response_format`), with the same prompt; the free-format run is kept as `data/replies_ministral-3-14b_freeformat.jsonl`.
- **GPT-6.1 Sol, copying.** The 53 empty-page trials left by the first run's budget were run ($0.41; $7.79 in all).

State of the data: the copying experiment has 820 of 820 parsed replies for each of the six models. The answer
experiments of `replicate.py` have every cell complete for every model; after two retries 9 agents of 22,650 have no
readable answer (8 gpt-oss-20b, 1 Ministral) and are left out of the rates. `replicate_gpt-oss-20b.jsonl` also holds
designs A and B, run again on the pages and seeds of Fig. 4 (an independent repeat of Fig. 4a and 4b; the paper's
gpt-oss-20b curves are the original runs).

**GPT-6.1 Sol** ran designs P, R and B through `replicate.py` (Responses API via `chain_replay.agent_responses`,
reasoning effort low, 16 requests at a time; design A is its earlier run, `panel_a_gpt-6.1-sol.jsonl`): P and R in
full, B at the fractions 0, 0.2, ..., 1 (750 agents), to stay within the budget. Cost $23.48 in all ($0.63 pilot,
$15.92 for P and R, $6.93 for B), no errors and no unparsed replies.

```bash
CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=16 python replicate.py --model gpt-6.1-sol --designs PR --concurrency 16
CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=8 python replicate.py --model gpt-6.1-sol --designs B --fracs-b 0,0.2,0.4,0.6,0.8,1 --concurrency 16
```

**GPT-6.1 Sol chains (Fig. 4c).** `CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=35 CHAIN_STARTS=1,3,7,32 python chain_replay.py
recstart 5 30`: 80 chains of 30 agents (20 per starting page), the design of the gpt-oss-20b chains with the starting
pages 1, 3, 7 and 32 (Sol's threshold lies between 3 and 7); `data/answers/chains_gpt-6.1-sol.jsonl`, 2,400 agents, no
errors, $14.23.

**gpt-oss-20b chains from 3 starting notes** (so that both models of Fig. 4c have 1, 3, 7 and 32): `CHAIN_STARTS=3
python chain_replay.py recstart 5 30`, 600 agents added to `data/answers/chains_gpt-oss-20b.jsonl`, no errors.

## 7. Controls of gpt-oss-20b and the figures of Extended Data and SI

- **Design M** (`replicate.py --designs M`): pages of N = 2, 4, 8, 12, 20, 32 notes of the record, half of them turned
  right. Checking falls with N in the same way as on all-wrong pages, so for gpt-oss-20b it depends on the length of
  the page, not on agreement.
- **Repeats** (`--tag repeat`) of design R, and of designs A and B (in `replicate_gpt-oss-20b.jsonl`), on the same
  pages and seeds; **medium reasoning effort** on design R (`--tag effort-medium`).

| Script | Figure |
| --- | --- |
| `fig4_checking.py` | Fig. 4 (pages of the record's notes; both models) |
| `ed_checking_models.py` | SI S7: checking before the correction, record and laboratory |
| `si_models_checking.py ed` / `si` | Extended Data: checking in six models (designs A, R); SI: synthetic notes (P, B) |
| `../src/fig2_wild_lab.py` | Fig. 2 and Extended Data: copying in six models (`ed_copying_models`) |
| `si_oss_controls.py` | SI: what drives checking in gpt-oss-20b (designs M, R repeated, medium effort, A repeated) |
| `si_chains.py` | SI: the chains of Fig. 4c chain by chain (uses `mock_fig4c_options.py` for the shared helpers) |
| `fig4_checking_synthetic.py` | the synthetic-note version of Fig. 4 (not in the paper; kept for comparison) |
