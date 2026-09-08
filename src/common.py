"""Paths, dataset loading and the definition of the population.

Every other module imports from here, so that the population is defined in
exactly one place: an agent belongs to the population if at least one of its
edits is on a task page.
"""
import collections
import datetime
import os
import pickle
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'cache')
FIGURES = os.path.join(ROOT, 'figures')
RESULTS = os.path.join(ROOT, 'results')
ICONS = os.path.join(ROOT, 'assets', 'icons', 'lucide')
for _d in (CACHE, FIGURES, RESULTS):
    os.makedirs(_d, exist_ok=True)

URL = re.compile(r'https?://[^\s\]\)|<>"\'\}]+')

# Page families that are not task families: coordination pages, link caches,
# crawler chains, and test or unclassified pages.
NONTASK = re.compile(
    r'relay-coordination|source-cache|source-or-unclassified|loop-chain'
    r'|off_store|probe-test|vermont-rent|unclassified|test|infrastructure|^unknown$')


def T(s):
    """Timestamp of a revision, as a datetime."""
    return datetime.datetime.fromisoformat(s[:19])


def cache_path(name):
    return os.path.join(CACHE, name)


def load_raw():
    """The parsed data release, as written by build_dataset.py.

    Human accounts and the blank username are dropped here, so that no
    downstream module has to remember to do it.
    """
    with open(cache_path('data.pkl'), 'rb') as fh:
        d = pickle.load(fh)
    human = {k for k, v in d['labs'].items() if v['is_human_handle']}
    d['revs'] = [r for r in d['revs'] if r['label'] != '' and r['label'] not in human]
    d['revs'].sort(key=lambda r: r['time'])
    return d


class Dataset:
    """The record, with the population already separated out.

    Attributes
    ----------
    revs_all : every non-human edit, in time order.
    revs     : the edits of the population only, in time order.
    pages    : page metadata, keyed by page_id.
    task_families : the set of page families named after a question source.
    swarm    : handles that never wrote on a task page (excluded).
    handles  : the population's handles, in order of first edit.
    birth    : handle -> time of its first edit.
    """

    def __init__(self, keep_swarm=False):
        d = load_raw()
        self.pages = d['pages']
        self.events = d['events']
        self.revs_all = d['revs']
        by_label = collections.defaultdict(list)
        for r in self.revs_all:
            by_label[r['label']].append(r)
        self.task_families = {f for f in {p['page_family'] for p in self.pages.values()}
                              if not NONTASK.search(f)}
        self.swarm = set() if keep_swarm else {
            h for h, rs in by_label.items()
            if not any(self.pages[r['page_id']]['page_family'] in self.task_families
                       for r in rs)}
        self.revs = [r for r in self.revs_all if r['label'] not in self.swarm]
        self.by_label = {h: rs for h, rs in by_label.items() if h not in self.swarm}
        self.birth = {h: T(rs[0]['time']) for h, rs in by_label.items()}
        self.handles = sorted(self.by_label, key=lambda h: self.birth[h])

    def family(self, rev):
        return self.pages[rev['page_id']]['page_family']

    def is_task(self, rev):
        return self.family(rev) in self.task_families

    def by_page(self, task_only=False):
        out = collections.defaultdict(list)
        for r in self.revs:
            if task_only and not self.is_task(r):
                continue
            out[r['page_id']].append(r)
        return out


def wilson(k, n, z=1.96):
    """Point estimate and 95% Wilson interval for a proportion."""
    import numpy as np
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def ccdf(x):
    import numpy as np
    x = np.sort(np.asarray(x, float))
    u = np.unique(x)
    return u, np.array([np.mean(x >= v) for v in u])


class Report:
    """Collects the numbers quoted in the paper into results/numbers.txt."""

    def __init__(self, section):
        self.section = section
        self.lines = []

    def __call__(self, line):
        print(line)
        self.lines.append(line)

    def write(self):
        path = os.path.join(RESULTS, 'numbers.txt')
        mode = 'a' if os.path.exists(path) else 'w'
        with open(path, mode) as fh:
            fh.write(f'\n=== {self.section} ===\n')
            fh.write('\n'.join(self.lines) + '\n')
