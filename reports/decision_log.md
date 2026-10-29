# Decision log: Extended Data and Supplementary Information

Branch `revise-extended-data-si`, 28 October 2026. Plan: `AI_Swarm_wiki/agent_extended_data_SI_workplan.md`.
No language model was called; no code, data or output was deleted (removed figures keep their scripts and files).
**Results that differ from the previous draft are marked ⚠.**

## Analyses added

| Analysis | Decision | Why |
|---|---|---|
| ED1b: patchwork, 1,000 permutations within 3-h blocks (`src/ed_patchwork.py`) | added, replaces the median of 50 shuffles | the plan asks for 500–1,000 permutations and an interval; the old median of 50 gave no resolution |
| ED1b: second null, permutation within task family × 3-h block | added, shown next to the time null, not in place of it | controls for what agents of one task write at one time. ⚠ Weaker result: 12 of 16 conventions above the 97.5th percentile (15 of 16 for the time null), median excess 0.024 against 0.084 |
| ED1b: bootstrap over pages of the same-page agreement | added | pages are the unit within which uses depend on one another |
| ED2b: Wald contrasts with the full clustered covariance, main and stricter fixed effects | added (`common.contrast`, covariances now stored by `identification.py`) | the main text compares coefficients; the plan asks not to compare intervals of single coefficients |
| ED2b: page − task's other pages, clustered by handle | added; the joint fit of `si_forms.py` was not clustered before | ⚠ standard errors of that fit grow from 0.03 to 0.05; the names contrast (0.12, p = 0.15) is not significant, and the text says so |
| Page choice set: Σ_j y_dj = 1 per append | **checked; it failed for 39 of 1,933 appends** (chosen page of another task outside the feed, so not a candidate). These appends are now left out rather than given an invented y = 1 | ⚠ Table 1 page row now 87,349 candidates of 1,894 appends (was 89,855 / 1,933). LPM coefficients change in the third decimal (2.09 → 2.10, 0.20 → 0.21); the conditional logit own/other feed-share coefficients 9.93/8.29 → 9.17/9.13; own-task visible chosen 8.7% (n 15,571) vs 0.5% (n 39,431); clogit visible own-task log-odds advantage 2.7 → 2.9 (factor 15 → 18) |
| SI S7 tables from the stored lab replies (`src/si_lab.py`) | added | the plan's P0-D: prepared, missing and used by experiment, model, condition and country; chains with the chain as unit |
| Points per bin in Fig. 3b and ED4 (`fig4_model.bin_counts`) | added to the numbers file and the ED4 caption | the plan asks for n per bin |
| `src/audit_checkpoints.py` | added | recomputes the checkpoints of the plan and compares with the draft |

## Analyses kept

| Analysis | Where now | Note |
|---|---|---|
| attention concentration (old ED1a) | ED1a | kept: it is a temporal indicator that Fig. 3a (a size distribution) does not show; caption now states the 18 blocks, the 30-edit threshold, the edits and pages per block, and that the even split is a reference, not a null |
| conflicts page vs rest of the task (old ED3a) | ED2a | ⚠ **204** conflicts, not 212 as in the old SI text, and still 68% [61, 74]; the old text was stale (family vs feed 822, not 877; strata 434, not 491; joint-fit family 0.09 not 0.10, feed 0.11 not 0.15) |
| names at four windows (old ED2) | ED2c, as contrasts | the coefficients stay in Table S(reg_names) |
| each convention vs model (old ED4) | ED3 | unchanged numbers; misses labelled; main text no longer says "close to those observed" but "of the observed size, volatility more closely than consistency" |
| prior fit, selection rules, recovery (old SI Fig. S10) | ED4 | moved, not duplicated; caption states the first-use bins lie above 1/s |
| checking in GPT-6.1 Sol (old ED6) | ED5 | Sol now at all 8 page sizes (was 2–3 pooled and 7 omitted), gpt-oss-20b at all 8 in panel a too; record measure labelled "cites a source (proxy)" |
| all SI tables (families, robustness, pieces, conventions, thresholds, slopes, habit, samples, regressions, choices, Wald, two-way, clogit) | S1–S6 | all kept |

## Analyses removed from the manuscript (kept in the repository)

| Item | Why |
|---|---|
| old ED1b, names over time | no claim of the main text rests on it |
| old ED3c, response split by the task's lean | repeats Fig. 2c; the conflict test and the contrasts carry the claim |
| old ED5, S-shaped copying (`lab/model_beta.py`, `lab/ed_model_beta.py`) | lateral to the linear model of the paper; no main-text sentence cited it; removed from `run_all.py` LAB_FIGURES, scripts kept |
| SI Figs. S1, S4, S6, S7, S8, S9, S11, S12, S13 | per the plan; the numbers that the text needs are kept in the text (S1 spans, S6 strength/direction/run, S4 slopes table) |
| SI paragraph on name fashions and the replay | no longer needed without its figure; the Scout adoption is kept in one sentence |
| SI Fig. S2c (creation rate by active pages) | the rate does not depend on the active pages, so it does not threaten a constant p_c; numbers in the text |

## Main text (only references, numbers and the sentences the plan asks for)

- `(ADD ROBUSTNESS TO SI)` → "other windows in Supplementary Information S2".
- Switch to the wrong answer: S5 → **S7**; all other SI section numbers realigned (S1–S6).
- New sentence after the window robustness: the page against the task's other pages, 68%, citing ED2a,b.
- Fig. 3b paragraph: one sentence separating the trend from the amplitude, citing ED4, with the exact 0.68–1.26 range.
- ED3 sentence softened (see above).
- Methods, page regression: the 39 appends left out and the new counts.
- Methods, Fig. 4b: five countries stated (Slovenia added); the S7 tables cited.
- Data availability: "No new data were collected" removed; the laboratory data are described as generated for this study and archived.
- Extended Data moved out of `main.tex` into `paper/extended_data.tex`; references across the three documents with `xr-hyper`.

## Not done, and why

- **Composition test for GPT-6.1 Sol** (plan, ED5 priority): no Sol trials with mixed pages exist; no API call was made. Stated in S7.6.
- **Manual audit of the record notes classified as citing a source**: not done; S7.2 says so.
- **Finite-size test with variable N** (ED4 optional): not run; the existing figure does not leave an ambiguity that it would resolve.
- **Log-loss / uniform-choice benchmark for the page R²** (optional): not run; the SI no longer presents R² as fit.
- **A separate `supplementary_information.tex`**: the SI stays in `paper/si.tex` to keep the existing build and the user's workflow; `extended_data.tex` is new.

## Follow-up (28 Oct): SI Fig. S3

The per-convention panels plotted first uses, which for 7 of 16 conventions fill a single bin, against a line
fitted on all uses: the line could not be judged from the points. Fig. S3 now plots the records the priors are
fitted on (`si_forms.per_convention` stores them as `fit_records`, from `conventions.prior_exposure`), with
the first uses faint. The line lies inside the 95% Wilson interval of 67 of 74 bins (misses: Rn/#n and
task clock/scaffold at mixed shares, update/edit and LIVE/live at one or two bins); the intervals ignore the
repetition of a handle's own form and are too narrow. The SI paragraph and caption were rewritten; the old
claim that "only the two ends of the line are populated" held for three panels only.
