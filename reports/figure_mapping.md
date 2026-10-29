# Figure mapping: old → new

Branch `revise-extended-data-si`, 28 October 2026. Old numbering is that of `main(8).pdf` / `si(2).pdf`.
Every removed figure keeps its script and its output in the repository (`src/`, `figures/*.png`,
`paper/figs/*.pdf`); only the `\includegraphics` and the references were removed from the manuscript.

## Extended Data

| Old | Content | New | Script → output |
|---|---|---|---|
| ED1a | attention concentrates, 6-h blocks | **ED1a** | `src/ed_figures.py:ed1` → `ed_attention_patchwork.pdf` |
| ED1b | how names are dressed over time | **removed** (no claim of the main text rests on it) | `src/fig2_collective.py` (kept, not in the build of the paper) |
| ED1c | the patchwork, time-shuffled null (median of 50) | **ED1b**, with 1,000 permutations, a second null within task family × 3 h, a bootstrap over pages | `src/ed_patchwork.py` → `cache/ed_patchwork.pkl`; `src/ed_figures.py:ed1` |
| ED2a,b | names at four task-mate windows, coefficients | **ED2c**, as the contrasts feed − out of view and page − out of view; coefficients in Table S(reg_names) | `src/ed_figures.py:ed2` → `ed_exposure_task.pdf` |
| ED3a | page against the rest of the task, conflicts | **ED2a** | same |
| ED3b | joint fit page / same task / feed | **ED2b**, as Wald contrasts with the full clustered covariance, main and stricter fixed effects | same |
| ED3c | response to the page share split by the task's lean | **removed** (repeats Fig. 2c) | `src/si_figures.py:fig_family` (kept) |
| ED4a–c | each convention against the model | **ED3** (unchanged content; misses labelled) | `src/ed_figures.py:ed3` → `ed_perconv.pdf` |
| ED5 | S-shaped copying | **removed** from ED and SI | `lab/model_beta.py`, `lab/ed_model_beta.py` (kept, not in the build) |
| ED6a,b | checking in GPT-6.1 Sol, record before the correction | **ED5**; both models at every page size (Sol no longer pooled at 2–3 notes); the record proxy labelled as such | `lab/ed_checking_models.py` → `ed_checking_models.pdf` |
| SI Fig. S10a–c | prior fit, selection rules, recovery | **ED4** (moved, not duplicated) | `src/fig4_model.py:prior_checks` → `ed_prior_fit.pdf` |

## Supplementary figures

| Old | Content | New |
|---|---|---|
| S1 | spans, activity, task families | **removed**; numbers in the text of S1 |
| S2 | feed windows, model windows, creation rate | **Fig. S1** (panels a, b); the creation rate in the text |
| S3 | attachment against visibility | **Fig. S2** |
| S4 | name fashions, record and replay | **removed** |
| S5 | each convention's response | **Fig. S3** |
| S6 | conventions over time (heat map) | **removed**; one sentence in S4 |
| S7 | dominant form against prior strength | **removed** (Fig. 3b and ED4 cover it) |
| S8a | same-page agreement against the time shuffle | **removed** (ED1b) |
| S8b | response with the page tally 6 h earlier | **removed** as a figure; slopes in Table S(slopes) |
| S9 | consistency within a page along s | **removed**; numbers in S6 |
| S10 | prior fit and 1/s | **ED4** |
| S11 | winner from run to run | **removed**; numbers in S6 |
| S12 | strength and direction of the prior | **removed**; numbers in S6 |
| S13 | regression coefficients | **removed** (Tables 1 and S9–S11) |

New tables: S(lab_design), S(lab_agreement), S(lab_country), S(lab_chains) in S7.

## References updated in the main text

| Main text, old | New |
|---|---|
| Extended Data Fig. `ed:collective`c (Fig. 2c caption) | Extended Data Fig. `ed:patchwork`b |
| Extended Data Fig. `ed:visibility` (Results, windows) | Extended Data Fig. `ed:task`c |
| — | new sentence citing Extended Data Fig. `ed:task`a,b |
| Extended Data Fig. `ed:perconv` | unchanged label, now ED3 |
| Extended Data Fig. `ed:checking` | unchanged label, now ED5 |
| — | Fig. 3b paragraph cites Extended Data Fig. `ed:priorfit` (ED4) |
| "Supplementary Information S5" (switch to the wrong answer) | S7 |
| "(ADD ROBUSTNESS TO SI)" | Supplementary Information S2 |
| "Supplementary Information S8" (regressions) | S5 |
| "Supplementary Information S2/S3/S4/S5" (population, window, pieces, candidates) | S1, S2, S3, S4 |
