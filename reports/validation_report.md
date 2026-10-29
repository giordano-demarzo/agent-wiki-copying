# Validation report

Branch `revise-extended-data-si`, 28 October 2026. Nothing is committed.

## Build

| Document | Pages | Undefined references | Overfull boxes |
|---|---:|---:|---:|
| `paper/main.pdf` | 21 | 0 | 0 |
| `paper/extended_data.pdf` (new) | 5 | 0 | 0 |
| `paper/si.pdf` | 22 | 0 | 0 |

The three documents reference one another through `xr-hyper` (`\externaldocument`), so each needs the `.aux`
of the others: compile `extended_data`, `si`, `main` twice (bibtex for `main`). No `??` in any PDF text.

`python src/run_all.py` ran every analysis step to the end (including the new `ed_patchwork` and `si_lab`) and
**stopped at `fig1_episode`**: `assets/icons/lucide/` is deleted in the working tree (`git status` shows it as
`D`). It was not restored, because the deletion may be intentional (`git checkout -- assets/icons` restores
it). The remaining figure steps and `lab/fig4_checking.py`, `lab/ed_checking_models.py` were run by hand
afterwards. `results/numbers.txt` therefore holds the full run plus a few sections appended by those manual
reruns; a clean full run needs the icons back.

## Numbers in the text

`python src/check_text.py`: **128 of 128 checks pass**, on `main.tex`, `si.tex` and `extended_data.tex`.
The SI checks were rewritten for the reorganized text; new checks cover the Extended Data captions, SI S5
(choice set), S7 (laboratory), and the revised main-text sentences. The 16 main-text checks that no longer
found their sentence (from the local rewrite of the main text) were replaced by checks of the current sentences.
Not machine-checked: the response slopes of Fig. 2 quoted in the Results (they are printed in `numbers.txt` by
`fig2_wild_lab`, not cached), the ED5 caption counts (printed by `lab/ed_checking_models.py`), and the
per-bin counts of the ED4 caption (printed by `fig4_model`).

## Checkpoints of the plan (`reports/checkpoints.md`)

All match the previous draft except:

| Quantity | Draft | Now | Cause |
|---|---:|---:|---|
| Table 1, page candidates | 89,855 | **87,349** | 39 appends whose chosen page is not in the choice set are left out (S5) |
| Table 1, page decisions | 1,933 | **1,894** | same |
| conflicts page vs rest of the task (forms) | 212 | **204** | the old SI number was stale; the code gives 204 (68%, unchanged) |

## Discrepancies found in the main text and corrected

| Where | Was | Now | Source |
|---|---|---|---|
| Results, names slope of the record | 0.65 | **0.57** | `fig2_wild_lab`: `(b) record: slope 0.57 ± 0.06`, the same estimator as the laboratory's 0.62 and 0.65; 0.65 predates the threshold change of 20 Oct |
| Methods, page regression | 89,855 from 1,933 | 87,349 from 1,894 | choice-set fix |
| Results, switch to the wrong answer | SI S5 | SI S7 | S5 was "How to write" |
| Results, window | "(ADD ROBUSTNESS TO SI)" | SI S2 | |
| Methods, Fig. 4b | country count not stated | five countries (Slovenia included) | `lab/data/answers/check_probe_v2_*`: 5 countries × 25 |
| Data availability | "No new data were collected" | laboratory data generated and archived | the lab experiments are new data |
| Results, ED3 sentence | "close to those observed" | "of the observed size, volatility more closely than consistency" | ED3: 11/16 inside the band for consistency, 13/15 for volatility |

## Discrepancies found in the SI and corrected

Family control (old ED3 text): 212 → 204 conflicts; 877 → 822 family-vs-feed conflicts; 491 → 434 strata; joint
fit 0.10/0.15 → 0.09/0.11, now with standard errors clustered by handle (0.03 → 0.05). Slopes on all uses
"0.89 to 0.95" → 0.85, 0.90, 0.95. Clogit log-odds 2.7 (factor 15) → 2.9 (factor 18) after the choice-set fix.
Own-task visible page chosen 8.6% (n 15,800) → 8.7% (n 15,571); out of view 0.4% (n 40,930) → 0.5% (n 39,431).
"slope about five times larger" for own task → about ten times (Table S(reg_pages): 2.10 against 0.21).

## Figures and captions

Every ED caption states the unit, what is measured, n, the definition of the bars and the meaning of the
comparison. Bars: Wilson intervals (ED2a, ED5a), Wald intervals from the clustered covariance (ED2b,c),
bootstrap over pages (ED1b) or over resamples of the ratio (ED5b), 10th–90th percentile of model runs (ED3,
stated as the spread of realizations, not an uncertainty), delta-method standard errors of binned RMS (ED4).
Figures were inspected as PNG previews in `figures/` (`ed_*.png`, `figS_*.png`).

## Open issues

1. **Fig. 1** cannot be rebuilt until the icons are restored (see above).
2. The names contrast "page authors − task's other pages" (0.12, p = 0.15) is not significant; the SI says so.
   The main text claims the page-vs-task comparison only for conventions (68%), which holds.
3. The task × time null of ED1b is weaker than the time null (12/16 vs 15/16; median excess 0.02 vs 0.08).
   The main text does not quote either; the caption gives both.
4. No GPT-6.1 Sol trials exist for the composition experiment (Fig. 4b) or the chains; nothing was run.
5. The record's "cites a source" proxy was not audited by hand.
6. `extended_data.tex` is a separate document. If the journal wants the ED at the end of the main PDF, the
   five `figure` blocks can be moved back into `main.tex` unchanged (labels and `\ref`s are the same).
