# Copying explains the collective behavior of AI agents in the wild

> Thousands of short-lived AI agents discovered a public wiki and began using it to help one another. Across attention, identity and language, one rule explains the collective that emerged: an agent takes an option with a probability close to its share among what it can see.

[![CI](https://github.com/giordano-demarzo/agent-wiki-copying/actions/workflows/ci.yml/badge.svg)](https://github.com/giordano-demarzo/agent-wiki-copying/actions/workflows/ci.yml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![Paper](https://img.shields.io/badge/arXiv-2609.09150-b31b1b.svg)](https://arxiv.org/abs/2609.09150) [![Data](https://img.shields.io/badge/data-collusion.wiki-3d9aa1.svg)](https://collusion.wiki) [![License: MIT](https://img.shields.io/badge/code-MIT-d99b3f.svg)](LICENSE)

Giordano De Marzo, Nicola Alborè and David Garcia, *Copying explains the collective behavior of AI agents in the wild* (2026). [Read the paper](https://arxiv.org/abs/2609.09150) · [Download the PDF](https://arxiv.org/pdf/2609.09150) · [Citation metadata](CITATION.cff)

[![The setting: a timed web-retrieval task, an AI agent reading from and writing to a public wiki, and the daily activity record](assets/readme/agent-wiki-setting.png)](https://arxiv.org/pdf/2609.09150)

*Figure 1. Each sandboxed agent answered timed web-retrieval questions, could read and edit a public wiki, and was shut down after its run. The wiki connected cohorts of agents that never overlapped in time.*

## The finding

Between 24 May and 22 June 2026, AI agents reportedly run by OpenAI's evaluation infrastructure made 13,661 non-human edits under 3,099 self-chosen handles on a family of public wikis. This repository studies the 1,201 handles that wrote on task pages: 5,929 edits, 3,807 of them on 679 task pages of 41 task families.

Within a day the agents behaved as a collective: they concentrated their attention on a few pages, signed in a shared style, and settled on a way of writing page by page. The record stores what each agent could see before it wrote, so the rule behind each choice can be measured on individual decisions:

| Choice | What an agent copies | What the model reproduces |
| --- | --- | --- |
| **Where to write** | a page, in proportion to its share of the last 100 lines of the recent-changes feed | the heavy-tailed number of handles that meet on a page |
| **How to sign** | the features of the names of its task-mates in view | the volatility of name features over time |
| **How to write** | the form of a convention already on its page, or in the feed | pages internally consistent and different from one another, and how far a convention ends from the agents' own bias |

Visibility identifies copying: exposure that had scrolled out of view, or that belongs to other tasks, carries no weight, and the estimates survive fixed effects for the task and the hour of arrival. A stylized model of identical agents, calibrated on those individual decisions and on nothing else, regenerates the collective. Where the agents' own bias is weak, whoever writes first decides the convention; where it is strong, no one can move it.

## Reproduce everything

Every figure, table and number of the manuscript (`paper/main.tex`) and of the Supplementary Information (`paper/si.tex`) is produced by the code in `src/` from the data release.

```bash
git clone https://github.com/giordano-demarzo/agent-wiki-copying.git
cd agent-wiki-copying
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/run_all.py      # about five minutes on 8 cores
make paper                 # compiles paper/main.pdf and paper/si.pdf
```

`src/run_all.py` writes

- `cache/` intermediate pickles, all regenerated from `data/`;
- `paper/figs/` the four figures of the paper, the three Extended Data figures and the SI figures, as the PDFs the manuscripts include;
- `paper/tables/` Table 1 and the SI tables, as LaTeX fragments the manuscripts input;
- `figures/` PNG previews of every figure;
- `results/numbers.txt` every number quoted in the paper and the SI.

Its last step, `src/check_text.py`, verifies that the numbers quoted in the two manuscripts are the ones the pipeline produced. `python src/run_all.py figures` redraws the figures and tables from an existing cache.

## Getting the data

The record is not redistributed here. Download the release from [collusion.wiki](https://collusion.wiki) and place its files in `data/`:

```text
data/
├── revisions.jsonl   # every stored revision: page text, diff hunks, username, time
├── pages.jsonl       # page metadata, family, deletion counts
├── labels.jsonl      # one record per username, including the human-account flag
└── events.jsonl      # server-side request log
```

```bash
cd data && sha256sum -c SHA256SUMS && cd ..
python src/build_dataset.py
```

See [`data/README.md`](data/README.md) for the expected files and reference totals.

## The pipeline

Every step is a module with a `main()`; `run_all.py` runs them in this order. Each analysis step writes one pickle in `cache/` and appends its numbers to `results/numbers.txt`.

| Module | What it does | Writes |
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
| `identification.py` | the regressions of Table 1 and the windows of the visibility test | `identification.pkl` |
| `run_model.py` | every run of the model, for Figure 4 and Sections S3, S6 and S7 | `model.pkl` |
| `si_population.py`, `si_names.py`, `si_forms.py` | Sections S1, S4 and S5 | `si_*.pkl` |
| `analysis_forms.py --full`, `identification.py --full` | the robustness check on all 3,099 handles (S2) | `*_full.pkl` |
| `fig1_setting.py` … `fig4_model.py` | Figures 1 to 4 | `paper/figs/` |
| `si_figures.py` | the Extended Data and SI figures | `paper/figs/` |
| `si_tables.py` | Table 1 and the SI tables | `paper/tables/` |
| `check_text.py` | the numbers quoted in `paper/*.tex` against the cache | |

The definitions the paper relies on live in one place each: the population in `common.Dataset`, the selection rule of the conventions in the constants at the top of `conventions.py`, the classification of name pieces and the exposure groups of a newcomer in `names.py`, and the model in `model.py`. Computation is separated from presentation: the figure modules only read the cache.

## Repository layout

```text
agent-wiki-copying/
├── src/              # the pipeline
├── data/             # the public release goes here; not redistributed
├── cache/            # regenerated intermediate objects
├── paper/            # main.tex, si.tex, and the figs/ and tables/ they include
├── figures/          # PNG previews
├── results/          # numbers.txt
├── assets/           # figure icons and README artwork
├── Makefile          # analysis, figures and paper targets
└── requirements.txt  # Python dependencies
```

## Citation

```bibtex
@article{demarzo2026copying,
  title   = {Copying explains the collective behavior of AI agents in the wild},
  author  = {De Marzo, Giordano and Alborè, Nicola and Garcia, David},
  year    = {2026},
  journal = {arXiv preprint arXiv:2609.09150},
  doi     = {10.48550/arXiv.2609.09150}
}
```

Machine-readable citation metadata are in [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). The paper and figures are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Icons in `assets/icons/lucide/` are from [Lucide](https://lucide.dev) under the ISC license included with the assets. The data release remains the property of its authors and is not redistributed by this repository.
