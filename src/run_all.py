"""Reproduce every number and every figure of the paper.

    python src/run_all.py

Expects the release in data/ (see data/README.md). Writes cache/ (intermediate
pickles), figures/ (the four figures as PDF and PNG) and results/numbers.txt
(every number quoted in the paper). Takes a few minutes.
"""
import os
import sys

from common import RESULTS

STEPS = [
    ('build_dataset', 'parse the release'),
    ('describe_population', 'the platform and the population'),
    ('timeline', 'daily counts for Figure 1c'),
    ('analysis_pages', 'where to write'),
    ('analysis_names', 'what to call yourself'),
    ('analysis_forms', 'how to write'),
    ('figure1_setting', 'Figure 1'),
    ('figure2_pages', 'Figure 2'),
    ('figure3_names', 'Figure 3'),
    ('figure4_forms', 'Figure 4'),
]


def main():
    numbers = os.path.join(RESULTS, 'numbers.txt')
    if os.path.exists(numbers):
        os.remove(numbers)
    for module, what in STEPS:
        print(f'\n----- {module}: {what} -----')
        sys.stdout.flush()
        __import__(module).main()
    print(f'\nDone. Numbers in {numbers}, figures in figures/.')


if __name__ == '__main__':
    main()
