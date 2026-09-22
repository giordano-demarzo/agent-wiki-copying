"""Paths, the dataset, the population, and the statistics shared by every step.

Every other module imports from here, so that the population is defined in
exactly one place: a handle belongs to the population if at least one of its
edits is on a task page.
"""
import collections
import datetime
import os
import pickle
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'cache')
FIGS = os.path.join(ROOT, 'paper', 'figs')        # the PDFs the manuscript includes
TABLES = os.path.join(ROOT, 'paper', 'tables')    # the LaTeX tables the manuscript inputs
PREVIEW = os.path.join(ROOT, 'figures')           # PNG previews, not tracked
RESULTS = os.path.join(ROOT, 'results')           # results/numbers.txt
ICONS = os.path.join(ROOT, 'assets', 'icons', 'lucide')
for _d in (CACHE, FIGS, TABLES, PREVIEW, RESULTS):
    os.makedirs(_d, exist_ok=True)

URL = re.compile(r'https?://[^\s\]\)|<>"\'\}]+')

# Page families that are not task families: coordination pages, link caches,
# crawler chains, and test or unclassified pages.
NONTASK = re.compile(
    r'relay-coordination|source-cache|source-or-unclassified|loop-chain'
    r'|off_store|probe-test|vermont-rent|unclassified|test|infrastructure|^unknown$')

FEED_LINES = 100      # the part of the feed an agent is assumed to read
BLOCK_H = 3           # hours in a block of the task x block fixed effects


def T(s):
    """Timestamp of a revision, as a datetime."""
    return datetime.datetime.fromisoformat(s[:19])


def cache_path(name):
    return os.path.join(CACHE, name)


def save_cache(name, obj):
    with open(cache_path(name), 'wb') as fh:
        pickle.dump(obj, fh)


def load_cache(name):
    with open(cache_path(name), 'rb') as fh:
        return pickle.load(fh)


def load_raw():
    """The parsed data release, as written by build_dataset.py.

    Human accounts and the blank username are dropped here, so that no
    downstream module has to remember to do it.
    """
    d = load_cache('data.pkl')
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
    first_family : handle -> family of its first task edit ('none' if it has none).

    With full=True nobody is excluded: the robustness check of the SI.
    """

    def __init__(self, full=False):
        d = load_raw()
        self.full = full
        self.pages = d['pages']
        self.events = d['events']
        self.revs_all = d['revs']
        by_label = collections.defaultdict(list)
        for r in self.revs_all:
            by_label[r['label']].append(r)
        self.task_families = {f for f in {p['page_family'] for p in self.pages.values()}
                              if not NONTASK.search(f)}
        self.swarm = set() if full else {
            h for h, rs in by_label.items()
            if not any(self.pages[r['page_id']]['page_family'] in self.task_families
                       for r in rs)}
        self.revs = [r for r in self.revs_all if r['label'] not in self.swarm]
        self.by_label = {h: rs for h, rs in by_label.items() if h not in self.swarm}
        self.birth = {h: T(rs[0]['time']) for h, rs in by_label.items()}
        self.handles = sorted(self.by_label, key=lambda h: self.birth[h])
        self.index = {h: i for i, h in enumerate(self.handles)}
        self.first_family = {}
        for h in self.handles:
            fams = [self.family(r) for r in self.by_label[h] if self.is_task(r)]
            self.first_family[h] = fams[0] if fams else 'none'

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


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """Point estimate and 95% Wilson interval for a proportion."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def ccdf(x):
    x = np.sort(np.asarray(x, float))
    u = np.unique(x)
    return u, np.array([np.mean(x >= v) for v in u])


def share(values, members, kmin):
    """Mean of a 0/1 indicator over a group, or NaN if the group has fewer
    than kmin members."""
    return values[members].mean() if len(members) >= kmin else np.nan


def fit(columns, y, groups=(), cluster=None):
    """Least squares with an intercept and fixed effects for every level of
    each grouping variable. Returns the coefficients of `columns`, their
    standard errors (clustered by `cluster` if given) and the R^2."""
    X = [np.ones(len(y))] + list(columns)
    for g in groups:
        for level in np.unique(g)[1:]:
            X.append((g == level).astype(float))
    X = np.column_stack(X)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    if cluster is None:
        cov = u @ u / (len(y) - X.shape[1]) * XtXi
    else:
        M = np.zeros((X.shape[1],) * 2)
        for level in np.unique(cluster):
            m = cluster == level
            sc = X[m].T @ u[m]
            M += np.outer(sc, sc)
        cov = XtXi @ M @ XtXi
    r2 = 1 - u @ u / ((y - y.mean()) @ (y - y.mean()))
    k = len(columns)
    return b[1:1 + k], np.sqrt(np.diag(cov))[1:1 + k], r2


def demean(X, groups):
    """Subtract the mean of each group from every column of X."""
    out = np.array(X, float).copy()
    for level in np.unique(groups):
        m = groups == level
        out[m] -= out[m].mean(0)
    return out


def block_of(t):
    """Index of the BLOCK_H-hour block of a timestamp in seconds."""
    return int(t // (BLOCK_H * 3600))


def co_occurrence(groups, minority):
    """P(two uses both take the minority form | same group) divided by the
    same probability for two uses in different groups. `minority` is a 0/1
    indicator of the rarer form; groups are pages (forms) or page ids."""
    groups, x = np.asarray(groups), np.asarray(minority, float)
    if x.mean() > 0.5:
        x = 1 - x
    if x.sum() < 5:
        return np.nan
    same = groups[:, None] == groups[None, :]
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)
    both = x[:, None] * x[None, :]
    return both[same].mean() / both[diff].mean() if both[diff].mean() > 0 else np.nan


def co_occurrence_coauthors(x, coauthors):
    """The same ratio for a 0/1 feature of handles: pairs that co-authored a
    task page against pairs that did not."""
    x = np.asarray(x, float)
    if x.mean() > 0.5:
        x = 1 - x
    other = ~coauthors
    np.fill_diagonal(other, False)
    both = x[:, None] * x[None, :]
    return both[coauthors].mean() / both[other].mean() if both[other].mean() > 0 else np.nan


def coauthor_matrix(page_sets, n):
    """n x n boolean matrix: True if two handles wrote on a common page."""
    A = np.zeros((n, n), bool)
    for hs in page_sets:
        hs = list(hs)
        if len(hs) > 1:
            A[np.ix_(hs, hs)] = True
    np.fill_diagonal(A, False)
    return A


def volatility(x, block):
    """Standard deviation of the share over consecutive blocks, divided by the
    binomial standard deviation at the overall share."""
    x = np.asarray(x, float)
    n = len(x) // block * block
    f = x.mean() if len(x) else np.nan
    if n < 2 * block or not 0 < f < 1:
        return np.nan
    return np.std(x[:n].reshape(-1, block).mean(1)) / np.sqrt(f * (1 - f) / block)


# ---------------------------------------------------------------- reporting
class Report:
    """Collects the numbers quoted in the paper into results/numbers.txt."""

    def __init__(self, section):
        self.section = section
        self.lines = []
        print(f'\n=== {section} ===')

    def __call__(self, line):
        print(line)
        self.lines.append(line)

    def write(self):
        path = os.path.join(RESULTS, 'numbers.txt')
        with open(path, 'a') as fh:
            fh.write(f'\n=== {self.section} ===\n')
            fh.write('\n'.join(self.lines) + '\n')


def write_table(name, lines):
    """A LaTeX tabular for the manuscript, in paper/tables/."""
    with open(os.path.join(TABLES, name), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote paper/tables/{name}')


def tex(s):
    """Escape a label for a LaTeX table."""
    return (s.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('#', '\\#')
            .replace('&', '\\&').replace('%', '\\%').replace('$', '\\$'))
