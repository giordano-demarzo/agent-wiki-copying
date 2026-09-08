"""The two-form conventions, and the exposure an agent had before each use.

A convention is a choice between two forms that do the same job. We start
from the candidates below and keep the ones with enough uses (see
analysis_forms.py). The three classes are:

    coined   names invented on the wiki, which the model cannot produce
             without having seen them;
    semantic near-synonyms that the model can produce on its own;
    habit    typographic choices such as capitalisation or number format.
"""
import collections
import re

import numpy as np

from common import URL


def rx(pattern, flags=re.I):
    """Count the occurrences of a form in a string."""
    r = re.compile(pattern, flags)
    return lambda s: len(r.findall(s))


# name -> (class, count of form A, count of form B)
CONV = {
    # coined on the wiki
    'Rn / #n': ('coined', rx(r'\bR[1-6]\b', 0), rx(r'#[1-6]\b', 0)),
    'task clock / scaffold': ('coined', rx(r'task[- ]clock'), rx(r'\bscaffold\b')),
    # near-synonyms
    'fast-tier / fast tier': ('habit', rx(r'\b(fast|slow|exact)-tier\b'), rx(r'\b(fast|slow|exact) tier\b')),
    'termination / teardown': ('semantic', rx(r'\bterminat'), rx(r'\bteardown')),
    'relay / bridge': ('semantic', rx(r'\brelay'), rx(r'\bbridge')),
    'deadline / cutoff': ('semantic', rx(r'\bdeadline'), rx(r'\bcut-?off')),
    'deadline / horizon': ('semantic', rx(r'\bdeadline'), rx(r'\bhorizon\b')),
    'signal / ping': ('semantic', rx(r'\bsignal'), rx(r'\bping\b')),
    'answer / value': ('semantic', rx(r'\banswers?\b'), rx(r'\bvalues?\b')),
    'update / edit': ('semantic', rx(r'\bupdat(e|ed|es|ing)\b'), rx(r'\bedit(s|ed|ing)?\b')),
    'tier / level': ('semantic', rx(r'\btiers?\b'), rx(r'\blevels?\b')),
    'seconds / sec': ('semantic', rx(r'\bseconds?\b'), rx(r'\bsecs?\b')),
    'minutes / min': ('semantic', rx(r'\bminutes?\b'), rx(r'\bmins?\b')),
    # typographic habits: capitalisation, hyphenation, number and time format,
    # pronoun, markup. The lowercase forms exclude sentence-initial positions.
    'CONFIRMED / confirmed': ('habit', rx(r'\bCONFIRMED\b', 0), rx(r'(?<![.!?]\s)(?<!^)\bconfirmed\b', 0)),
    'LIVE / live': ('habit', rx(r'\bLIVE\b', 0), rx(r'\blive\b', 0)),
    'BEFORE / before': ('habit', rx(r'\bBEFORE\b', 0), rx(r'(?<![.!?]\s)\bbefore\b', 0)),
    'COHORT / cohort': ('habit', rx(r'\bCOHORT\b', 0), rx(r'\bcohort\b', 0)),
    'GET / get': ('habit', rx(r'\bGET\b', 0), rx(r'\bget\b', 0)),
    'API / api': ('habit', rx(r'\bAPI\b', 0), rx(r'\bapi\b', 0)),
    'JSON / json': ('habit', rx(r'\bJSON\b', 0), rx(r'\bjson\b', 0)),
    'URL / url': ('habit', rx(r'\bURLs?\b', 0), rx(r'\burls?\b', 0)),
    'DataUSA / datausa': ('habit', rx(r'\bDataUSA\b', 0), rx(r'\bdatausa\b', 0)),
    'EXACT / exact': ('habit', rx(r'\bEXACT\b', 0), rx(r'\bexact\b', 0)),
    'task-clock / task clock': ('habit', rx(r'task-clock'), rx(r'task clock')),
    'pre-signal / presignal': ('habit', rx(r'pre-signal'), rx(r'\bpresignal\b')),
    'no-show / noshow': ('habit', rx(r'no-show'), rx(r'\bnoshow\b')),
    '1,234 / 1234': ('habit', rx(r'\b\d{1,3}(?:,\d{3})+\b', 0), rx(r'\b\d{5,}\b', 0)),
    'HH:MM:SS / HH:MM': ('habit', rx(r'\b\d{1,2}:\d{2}:\d{2}\b', 0), rx(r'\b\d{1,2}:\d{2}\b(?!:)', 0)),
    'we / I': ('habit', rx(r'\b[Ww]e\b', 0), rx(r'\bI\b', 0)),
    'UTC / Z': ('habit', rx(r'\d\d:\d\d(?::\d\d)? ?UTC\b', 0), rx(r'\d\d:\d\d(?::\d\d)?Z\b', 0)),
    "bold ** / '''": ('habit', rx(r'\*\*[^*\n]+\*\*', 0), rx(r"'''[^'\n]+'''", 0)),
    '-> / arrow': ('habit', rx(r'->', 0), rx(r'\u2192', 0)),
}

FEED_USES = 30   # the feed exposure is the last 30 uses by other handles


def sequence(revs, name):
    """The ordered uses of a convention.

    Returns the indices into `revs` of the edits that use the convention, and
    for each one a 0 if it took form A and a 1 if it took form B. An edit
    counts as a use if it contains strictly more of one form than of the other.
    """
    _, count_a, count_b = CONV[name]
    idx, form = [], []
    for i, r in enumerate(revs):
        s = URL.sub(' ', r['added'] or '')
        a, b = count_a(s), count_b(s)
        if a + b == 0 or a == b:
            continue
        idx.append(i)
        form.append(0 if a > b else 1)
    return np.array(idx), np.array(form)


def exposure(idx, form, label, page, k_feed=FEED_USES):
    """What each use could see, page first.

    For every use, returns the share of form A in the page-first exposure
    (the tally of earlier uses on the same page, or the last `k_feed` uses by
    other handles if the page carries neither form) and the share on the page
    alone. Both are NaN when fewer than three instances are in view.
    """
    page_hist = {}
    share, share_page = [], []
    for k, i in enumerate(idx):
        handle, pg = label[i], page[i]
        window = [j for j in range(max(0, k - k_feed), k) if label[idx[j]] != handle]
        feed = (np.array([np.sum(form[window] == 0), np.sum(form[window] == 1)], float)
                if window else np.zeros(2))
        on_page = page_hist.get(pg, np.zeros(2))
        seen = on_page if on_page.sum() > 0 else feed
        share.append(seen[0] / seen.sum() if seen.sum() >= 3 else np.nan)
        share_page.append(on_page[0] / on_page.sum() if on_page.sum() >= 3 else np.nan)
        step = np.zeros(2)
        step[form[k]] = 1
        page_hist[pg] = on_page + step
    return np.array(share), np.array(share_page)


def fit_mu(share, took_a):
    """Maximum likelihood fit of P(A | rho) = mu_A + (1 - mu_A - mu_B) rho."""
    from scipy.optimize import minimize
    m = ~np.isnan(share)
    if m.sum() < 30:
        return np.array([np.nan, np.nan])

    def nll(q):
        p = np.clip(q[0] + (1 - q[0] - q[1]) * share[m], 1e-6, 1 - 1e-6)
        return -np.sum(took_a[m] * np.log(p) + (1 - took_a[m]) * np.log(1 - p))

    return minimize(nll, [0.05, 0.05], bounds=[(0, 0.95), (0, 0.95)],
                    method='L-BFGS-B').x
