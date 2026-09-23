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
import sys
import time

from common import RESULTS

ANALYSIS = [
    ('build_dataset', 'parse the release into cache/data.pkl'),
    ('describe_population', 'the platform and the population'),
    ('timeline', 'daily counts for Figure 1c'),
    ('analysis_pages', 'where to write'),
    ('analysis_names', 'how to sign'),
    ('analysis_forms', 'how to write'),
    ('identification', 'the regressions of Table 1'),
    ('run_model', 'the model'),
    ('si_population', 'SI: the record and the population'),
    ('si_names', 'SI: how to sign'),
    ('si_forms', 'SI: how to write'),
]
FULL = [('analysis_forms', 'how to write, all handles'), ('identification', 'the regressions, all handles')]
FIGURES = [
    ('fig1_setting', 'Figure 1'),
    ('fig2_collective', 'Figure 2'),
    ('fig3_rule', 'Figure 3'),
    ('fig4_model', 'Figure 4'),
    ('si_figures', 'the SI figures'),
    ('si_tables', 'Table 1 and the SI tables'),
    ('check_text', 'the numbers quoted in the manuscripts against the cache'),
]


def step(module, what, **kw):
    t0 = time.time()
    print(f'\n----- {module}: {what} -----', flush=True)
    importlib.import_module(module).main(**kw)
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
    print(f'\nDone. Numbers in results/numbers.txt, figures in paper/figs/, tables in paper/tables/.')


if __name__ == '__main__':
    main(only_figures='figures' in sys.argv[1:])
