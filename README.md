# Copying explains the collective behavior of AI agents in the wild

Code, manuscript and Supplementary Information for our analysis of the OpenAI
agent wiki incident of June 2026.

In June 2026 thousands of OpenAI agents working on timed tasks found that a
public wiki accepted edits from their sandboxes and used it, unasked, to help
one another. The record of what each agent saw before it wrote lets us follow
the three choices an agent made on arrival, where to write, how to sign and how
to word what it wrote, and show that one rule governs all three: an agent takes
an option with a probability close to its share among what it can see. A
stylized model built on that rule, with every probability measured on
individual decisions, reproduces the collective: the heavy-tailed distribution
of attention over pages, the consistency of conventions within a page and
their volatility over time.

Everything in the manuscript (`paper/main.tex`) and in the Supplementary
Information (`paper/si.tex`) is produced by the code in `src/` from the data
release: every figure, every table and every number quoted in the text.

## Getting the data

The record is not redistributed here. Download the release from
[collusion.wiki](https://collusion.wiki) and put the four `.jsonl` files in
`data/`. See [`data/README.md`](data/README.md) for the expected files and
checksums.

## Reproducing everything

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py          # about five minutes on 8 cores
make paper                     # compiles paper/main.pdf and paper/si.pdf
```

`src/run_all.py` writes

- `cache/` intermediate pickles, all regenerated from `data/`;
- `paper/figs/` the four figures of the paper and the fourteen of the SI, as
  the PDFs the manuscripts include;
- `paper/tables/` Table 1 and the eight tables of the SI, as LaTeX fragments
  the manuscripts input;
- `figures/` PNG previews of every figure;
- `results/numbers.txt` every number quoted in the paper and the SI, in the
  order the steps produce them.

`python src/run_all.py figures` redraws the figures and tables from an
existing cache without redoing the analyses.

## The steps

Every step is a module with a `main()`; `run_all.py` runs them in this order.
Each writes one pickle in `cache/` and appends its numbers to
`results/numbers.txt`.

| module | what it does | writes |
| --- | --- | --- |
| `common.py` | paths, the dataset and the population, the shared statistics | |
| `conventions.py` | the 52 candidate conventions, the selection rule, the exposure before a use | |
| `names.py` | the pieces of a name, their classification, the 16 name features, what a newcomer could see | |
| `model.py` | the stylized model and its statistics | |
| `build_dataset.py` | parses the release | `data.pkl` |
| `describe_population.py` | the platform and the population | |
| `timeline.py` | daily counts for Figure 1c | `timeline.json` |
| `analysis_pages.py` | where to write: response, windows, creation rate, attachment | `pages.pkl` |
| `analysis_names.py` | how to sign: pieces, features, exposure records, prior strengths | `names.pkl` |
| `analysis_forms.py` | how to write: selection, patchwork, response, conflicts, usage rate | `forms.pkl` |
| `identification.py` | the regressions of Table 1, the windows of the visibility test | `identification.pkl` |
| `run_model.py` | every run of the model, for Figure 4 and Sections S3, S6 and S7 | `model.pkl` |
| `si_population.py` | Section S1 | `si_population.pkl` |
| `si_names.py` | Section S4 | `si_names.pkl` |
| `si_forms.py` | Section S5, including the family control | `si_forms.pkl` |
| `analysis_forms.py --full`, `identification.py --full` | the robustness check on all 3,099 handles (S2) | `*_full.pkl` |
| `fig1_setting.py` … `fig4_model.py` | Figures 1 to 4 | `paper/figs/` |
| `si_figures.py` | the SI figures | `paper/figs/` |
| `si_tables.py` | Table 1 and the SI tables | `paper/tables/` |
| `check_text.py` | checks every number quoted in `paper/main.tex` and `paper/si.tex` against the cache; fails if one drifted | |

The definitions the paper relies on live in one place each: the population in
`common.Dataset`, the selection rule of the conventions in the constants at the top of
`conventions.py`, the
classification of name pieces and the exposure groups of a newcomer in
`names.py`, and the model in `model.py`.

## Citing

See `CITATION.cff`.
