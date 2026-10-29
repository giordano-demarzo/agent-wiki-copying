"""Reproduce every number, figure and table of the paper and of the
Supplementary Information from the data release.

    python src/run_all.py            everything (about five minutes on 8 cores)
    python src/run_all.py figures    only the figures and tables, from an existing cache

Expects the release in data/ (see data/README.md). Writes cache/ (intermediate
pickles), paper/figs/ (the PDFs the manuscript includes), paper/tables/ (the
LaTeX tables it inputs), figures/ (PNG previews) and results/numbers.txt (every
number quoted in the paper and the SI, in the order the steps produce them).
"""
import importlib
import os
import subprocess
import sys
import time

from common import RESULTS, ROOT

ANALYSIS = [
    ('build_dataset', 'parse the release into cache/data.pkl'),
    ('describe_population', 'the platform and the population'),
    ('timeline', 'daily counts of edits and handles'),
    ('network', 'the network of handles of Figure 1c'),
    ('analysis_pages', 'where to write'),
    ('analysis_names', 'how to sign'),
    ('analysis_forms', 'how to write'),
    ('answers_record', 'the OECD answers in the record: the wrong consensus and its correction'),
    ('identification', 'the regressions of Table 1'),
    ('run_model', 'the model'),
    ('si_recovery', 'SI: the prior fit applied to the model record'),
    ('si_population', 'SI: the record and the population'),
    ('si_names', 'SI: how to sign'),
    ('si_forms', 'SI: how to write'),
    ('si_thresholds', 'SI: the convention selection rule'),
    ('ed_patchwork', 'Extended Data Fig. 1b: the patchwork against two permutation nulls'),
    ('si_lab', 'SI S7: the laboratory experiments, counted from the stored replies'),
]
FULL = [('analysis_forms', 'how to write, all handles'), ('identification', 'the regressions, all handles')]
FIGURES = [
    ('fig1_episode', 'Figure 1'),
    ('fig3_rule', 'the response curves of the record (numbers for the text)'),
    ('fig2_wild_lab', 'Figure 2, record and laboratory'),
    ('fig4_model', 'Figure 3 and Extended Data Fig. 4'),
    ('ed_figures', 'Extended Data Figs. 1-3'),
    ('si_figures', 'the SI figures'),
    ('si_tables', 'Table 1 and the SI tables'),
    ('check_text', 'the numbers quoted in the manuscripts against the cache'),
]


def step(module, what, **kw):
    t0 = time.time()
    print(f'\n----- {module}: {what} -----', flush=True)
    importlib.import_module(module).main(**kw)
    print(f'      ({time.time() - t0:.0f} s)', flush=True)


# the figures built on the laboratory experiments (lab/; their data are in lab/data/)
LAB_FIGURES = [
    ('fig4_checking.py', 'Figure 4'),
    ('ed_checking_models.py', 'Extended Data Fig. 5: checking in GPT-6.1 Sol and in the record before the correction'),
]
# model_beta.py and ed_model_beta.py (S-shaped copying) are kept in lab/ but no longer enter the paper


def lab_step(script, what):
    t0 = time.time()
    print(f'\n----- lab/{script}: {what} -----', flush=True)
    subprocess.run([sys.executable, script], cwd=os.path.join(ROOT, 'lab'), check=True)
    print(f'      ({time.time() - t0:.0f} s)', flush=True)


def main(only_figures=False):
    numbers = os.path.join(RESULTS, 'numbers.txt')
    if not only_figures:
        if os.path.exists(numbers):
            os.remove(numbers)
        for module, what in ANALYSIS:
            step(module, what)
        for module, what in FULL:
            step(module, what, full=True)
    for module, what in FIGURES:
        step(module, what)
    for script, what in LAB_FIGURES:
        lab_step(script, what)
    print(f'\nDone. Numbers in results/numbers.txt, figures in paper/figs/, tables in paper/tables/.')


if __name__ == '__main__':
    main(only_figures='figures' in sys.argv[1:])
