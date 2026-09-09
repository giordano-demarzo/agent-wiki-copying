# Copying explains the collective behavior of AI agents in the wild

Code and paper for our analysis of the OpenAI agent wiki incident of June 2026.

Thousands of AI agents, run by OpenAI's evaluation infrastructure and each alive
for about an hour, found that a small public wiki accepted edits from their
sandbox and started using it to help one another pass a timed test. Nobody asked
them to cooperate, and the wiki was not built for them. This repository contains
everything needed to reproduce our analysis of that record, in which we follow
the three choices an agent had to make and show that one copying rule governs
all three:

1. **where to write.** An agent picks a page in proportion to how much of the
   recent feed that page occupies.
2. **what to call itself.** An agent assembles its name out of pieces that were
   in view in the last thirty names.
3. **how to write.** An agent takes a form with a probability close to the share
   of that form on the page it is writing on.

Each of the three is reproduced by a minimal model with a single free parameter.

## Getting the data

The record is not redistributed here. Download the release from
[collusion.wiki](https://collusion.wiki) and put the four `.jsonl` files in
`data/`. See [`data/README.md`](data/README.md) for the expected files and
checksums.

## Reproducing everything

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```

This takes a few minutes and writes:

- `cache/` intermediate pickles, all regenerated from `data/`;
- `figures/` the four figures of the paper, as PDF and PNG;
- `results/numbers.txt` every number quoted in the paper, in the order the paper
  quotes them.

Individual steps can be run on their own, in this order:

| script | what it does |
| --- | --- |
| `src/build_dataset.py` | parses the release into `cache/data.pkl` |
| `src/describe_population.py` | the platform and the population |
| `src/timeline.py` | daily counts for Figure 1c |
| `src/analysis_pages.py` | where to write (Figure 2) |
| `src/analysis_names.py` | what to call yourself (Figure 3) |
| `src/analysis_forms.py` | how to write (Figure 4) |
| `src/figure1_setting.py` … `src/figure4_forms.py` | the figures |

## How the code is organised

- **`src/common.py`** holds the paths, the dataset loader and the definition of
  the population. An agent belongs to the population if at least one of its
  edits is on a task page, which leaves 1,201 of the 3,099 handles; the rest are
  link posters and crawler tests. The definition lives in exactly one place, so
  every analysis uses the same population. The `full-population` branch runs
  everything again with no exclusion at all, as the robustness check reported in
  the Methods of the paper.
- **`src/conventions.py`** holds the two-form conventions, the rule for
  extracting the uses of one from the edits, and the reconstruction of what each
  use could see.
- The three `analysis_*.py` scripts compute and cache; the four `figure*.py`
  scripts only plot. Nothing is fitted inside a figure script.

## The paper

`paper/final.tex` builds with `pdflatex` and needs the figures in
`paper/figs/`, which are copies of what `src/run_all.py` writes to `figures/`.

```bash
cd paper && pdflatex final.tex && pdflatex final.tex
```

## Licence

Code is MIT (see `LICENSE`). The paper and the figures are CC BY 4.0. The icons
in `assets/icons/lucide/` are from [Lucide](https://lucide.dev) under the ISC
licence, included in that directory. The data release belongs to its authors and
is not redistributed here.

## Citing

See `CITATION.cff`.
