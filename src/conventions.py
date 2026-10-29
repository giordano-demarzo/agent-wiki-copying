"""The two-form conventions, their selection, and the exposure an agent had
before each use.

A convention is a choice between two forms that do the same job. We start from
the 52 candidates below and keep those whose two base probabilities the record
can measure (select()). The three classes are:

    coined   names invented on the wiki, which the model cannot produce
             without having seen them;
    semantic near-synonyms that the model can produce on its own;
    habit    habits of style and typography, such as capitalisation or number
             format.
"""
import collections
import re

import numpy as np

from common import FEED_LINES, URL, cache_path

# the selection rule: a convention is kept if the record can measure its base
# probabilities, which needs enough uses, both forms in real use, and pages
MIN_USES = 100        # uses in total
MIN_MINORITY = 0.05   # share of the uses taking the rarer form
MIN_PAGES = 10        # pages carrying five or more uses
MIN_IN_VIEW = 3       # instances in view for a use to enter a fit or a curve
CLASSES = ('coined', 'semantic', 'habit')


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
    # HH:MM must not be the tail of an HH:MM:SS: without the lookbehind every HH:MM:SS also counted as an HH:MM
    'HH:MM:SS / HH:MM': ('habit', rx(r'\b\d{1,2}:\d{2}:\d{2}\b', 0), rx(r'(?<![\d:])\b\d{1,2}:\d{2}\b(?!:)', 0)),
    'we / I': ('habit', rx(r'\b[Ww]e\b', 0), rx(r'\bI\b', 0)),
    'UTC / Z': ('habit', rx(r'\d\d:\d\d(?::\d\d)? ?UTC\b', 0), rx(r'\d\d:\d\d(?::\d\d)?Z\b', 0)),
    "bold ** / '''": ('habit', rx(r'\*\*[^*\n]+\*\*', 0), rx(r"'''[^'\n]+'''", 0)),
    '-> / arrow': ('habit', rx(r'->', 0), rx(r'\u2192', 0)),
    # further pairs, added to cover a wider range of prior strengths; they go
    # through the same selection rule as the ones above
    'em dash / hyphen': ('habit', rx(r'—|–', 0), rx(r'(?<=\w) - (?=\w)', 0)),
    'curly / straight quotes': ('habit', rx(r'[“”‘’]', 0), rx(r'"', 0)),
    'e.g. / eg': ('habit', rx(r'\be\.g\.', 0), rx(r'\beg\b', 0)),
    '% / percent': ('habit', rx(r'\d\s?%', 0), rx(r'\d\s?percent\b')),
    'USD / $': ('habit', rx(r'\bUSD\b', 0), rx(r'\$\s?\d', 0)),
    'OK / okay': ('habit', rx(r'\bOK\b', 0), rx(r'\bokay\b')),
    'ISO date / written date': ('habit', rx(r'\b20\d\d-\d\d-\d\d\b', 0), rx(r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* 20\d\d\b', 0)),
    '24h / am-pm': ('habit', rx(r'\b([01]\d|2[0-3]):[0-5]\d\b(?!\s?[ap]\.?m)', 0), rx(r'\b1?\d(:\d\d)?\s?[ap]\.?m\.?\b', 0)),
    'true/false / yes/no': ('habit', rx(r'\b(true|false)\b', 0), rx(r'\b(yes|no)\b', 0)),
    'TODO / todo': ('habit', rx(r'\bTODO\b', 0), rx(r'\btodo\b', 0)),
    'N/A / NA': ('habit', rx(r'\bN/A\b', 0), rx(r'\bNA\b', 0)),
    "'''bold''' / == heading": ('habit', rx(r"'''", 0), rx(r'^==+', re.M)),
    '* bullet / - bullet': ('habit', rx(r'^\* ', re.M), rx(r'^- ', re.M)),
    'Note: / NB': ('habit', rx(r'\bNote:', 0), rx(r'\bNB\b', 0)),
    'vs. / vs': ('habit', rx(r'\bvs\.', 0), rx(r'\bvs\b(?!\.)', 0)),
    '& / and': ('habit', rx(r' & ', 0), rx(r' and ', 0)),
    '1st / first': ('habit', rx(r'\b\d+(st|nd|rd|th)\b', 0), rx(r'\b(first|second|third|fourth|fifth)\b')),
    'therefore / so': ('semantic', rx(r'\btherefore\b'), rx(r'(?<=[,;]) so\b')),
    'Wiki / wiki': ('habit', rx(r'\bWiki\b', 0), rx(r'\bwiki\b', 0)),
    'ID / id': ('habit', rx(r'\bIDs?\b', 0), rx(r'\bids?\b', 0)),
    # 'cache / cached copy' was removed: 'cached' is a participle ("table cached") and 'cache' a noun or verb, not two forms of one meaning
}

# pairs whose uses are not a choice between two forms: almost every '&' of the record (177 of 181 in the
# uses that take it) is inside a category name quoted from a statistics site, such as 'Visual & Performing Arts'
NOT_A_CHOICE = {'& / and': 'category names'}

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


def prior_exposure(idx, form, label, page, lines=FEED_LINES, first_only=False):
    """What each use could see, as the model reads it, for the fit of the priors.

    Returns, for every use, the share of form A in view: the tally of earlier
    uses on the same page (the writer's own included), or, if the page carries
    neither form, the uses by other handles in the last `lines` revisions of
    the feed. NaN when fewer than MIN_IN_VIEW instances are in view, and, with
    `first_only`, for every use but a handle's first (the SI check: later uses
    repeat the handle's own earlier form, which the fit on all uses counts as
    copying).
    """
    page_hist, met, share = {}, set(), []
    for k, i in enumerate(idx):
        handle, pg = label[i], page[i]
        on_page = page_hist.get(pg, np.zeros(2))
        if first_only and handle in met:
            share.append(np.nan)
        else:
            if on_page.sum() > 0:
                seen = on_page
            else:
                window = [j for j in range(k) if idx[j] >= i - lines and label[idx[j]] != handle]
                seen = np.array([np.sum(form[window] == 0), np.sum(form[window] == 1)], float)
            share.append(seen[0] / seen.sum() if seen.sum() >= MIN_IN_VIEW else np.nan)
        met.add(handle)
        step = np.zeros(2)
        step[form[k]] = 1
        page_hist[pg] = on_page + step
    return np.array(share)


# the directions tried before refining: dense near 0 and 1, where a rare form puts pi
PI_GRID = np.unique(np.concatenate([np.geomspace(1e-4, 0.05, 25), np.linspace(0.05, 0.95, 37),
                                    1 - np.geomspace(1e-4, 0.05, 25), [0.0, 1.0]]))


def _nll_function(share, took_a):
    m = ~np.isnan(share)
    x, y = share[m], took_a[m]

    def nll(s, pi):
        p = np.clip(np.multiply.outer(np.atleast_1d(s * pi), np.ones(len(x))) + (1 - s) * x, 1e-6, 1 - 1e-6) \
            if np.ndim(pi) else np.clip(s * pi + (1 - s) * x, 1e-6, 1 - 1e-6)
        return -(np.log(p) * y + np.log(1 - p) * (1 - y)).sum(-1)
    return nll, m.sum()


def _best_pi(nll, s):
    """The direction that maximizes the likelihood at prior strength s: the best
    point of PI_GRID, refined between its neighbours."""
    from scipy.optimize import minimize_scalar
    v = nll(s, PI_GRID)
    k = int(np.argmin(v))
    lo, hi = PI_GRID[max(k - 1, 0)], PI_GRID[min(k + 1, len(PI_GRID) - 1)]
    r = minimize_scalar(lambda p: nll(s, p), bounds=(lo, hi), method='bounded')
    return (r.x, r.fun) if r.fun < v[k] else (PI_GRID[k], v[k])


def fit_mu(share, took_a):
    """Maximum likelihood fit of P(A | rho) = mu_A + (1 - mu_A - mu_B) rho,
    parametrized as mu_A = s pi, mu_B = s (1 - pi) with s and pi in [0, 1]:
    the profile over a grid of s, refined around its best point."""
    from scipy.optimize import minimize_scalar
    nll, n = _nll_function(share, took_a)
    if n < 30:
        return np.array([np.nan, np.nan])
    grid = np.linspace(0, 1, 101)
    prof = [_best_pi(nll, s)[1] for s in grid]
    k = int(np.argmin(prof))
    r = minimize_scalar(lambda s: _best_pi(nll, s)[1], bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 100)]),
                        method='bounded')
    s = r.x if r.fun < prof[k] else grid[k]
    pi = _best_pi(nll, s)[0]
    return np.array([s * pi, s * (1 - pi)])


def profile_interval(share, took_a, grid=np.linspace(0, 1, 101)):
    """Profile-likelihood estimate and 95% interval of the prior strength
    s = mu_A + mu_B: for every s on the grid the direction is optimised, and the
    interval is where the log-likelihood is within 1.92 of its maximum."""
    nll, _ = _nll_function(share, took_a)
    prof = np.array([_best_pi(nll, s)[1] for s in grid])
    ok = grid[prof <= prof.min() + 1.92]
    return grid[prof.argmin()], ok.min(), ok.max()


FLOOR = 1e-3   # a fitted prior below this is at the boundary: the record shows no innovation to that form


def profile_upper(share, took_a, which, grid=np.concatenate([np.linspace(0, 0.02, 201), np.linspace(0.02, 0.95, 931)])):
    """The 95% profile-likelihood upper bound of mu_A (which = 0) or mu_B (which = 1):
    the largest value whose profile log-likelihood, with the other prior
    optimised, is within 1.92 of the maximum."""
    from scipy.optimize import minimize_scalar
    m = ~np.isnan(share)
    x, y = share[m], took_a[m]

    def nll(mu_a, mu_b):
        p = np.clip(mu_a + (1 - mu_a - mu_b) * x, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    prof = lambda v: minimize_scalar(lambda o: nll(o, v) if which == 1 else nll(v, o), bounds=(0, 1 - v), method='bounded').fun
    best = min(prof(v) for v in (0.0, 0.01, 0.05, 0.1))
    ok = [v for v in grid if prof(v) <= best + 1.92]
    return max(ok) if ok else np.nan


def floor_mu(share, took_a, mu):
    """The fitted priors with a prior at zero replaced by the largest value the
    record cannot exclude (its 95% upper bound), so that pi = mu_A / s is
    defined and the distance from the prior is a lower bound. Returns the pair
    and whether a floor was applied."""
    mu = np.array(mu, float)
    floored = False
    for which in (0, 1):
        if np.isfinite(mu[which]) and mu[which] < FLOOR:
            mu[which] = profile_upper(share, took_a, which)
            floored = True
    return mu, floored


def fit_prior(idx, form, label, page, first_only=False):
    """The fitted base probabilities of one convention: the pair, its profile
    interval of s, the pair with a prior at zero floored (floor_mu), and
    whether a floor was applied."""
    share = prior_exposure(idx, form, label, page, first_only=first_only)
    took_a = (form == 0).astype(float)
    mu = fit_mu(share, took_a)
    mu_floor, floored = floor_mu(share, took_a, mu)
    return mu, profile_interval(share, took_a), mu_floor, floored


def candidates(ds):
    """Every candidate convention with the quantities the selection uses, its
    fitted base probabilities, and whether it is kept."""
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    rows = {}
    for name, (cls, _, _) in CONV.items():
        idx, form = sequence(ds.revs, name)
        per_page = collections.Counter(page[i] for i in idx)
        n_pages = sum(1 for v in per_page.values() if v >= 5)
        minority = min(np.mean(form == 0), np.mean(form == 1)) if len(idx) else np.nan
        mu, interval = np.array([np.nan, np.nan]), None
        mu_floor, floored = mu.copy(), False
        if len(idx) >= MIN_USES and minority > 0:
            mu, interval, mu_floor, floored = fit_prior(idx, form, label, page)
        fails = [why for why, bad in (('uses', len(idx) < MIN_USES), ('pages', n_pages < MIN_PAGES),
                                      ('minority', not minority >= MIN_MINORITY)) if bad]
        if name in NOT_A_CHOICE:
            fails.append(NOT_A_CHOICE[name])
        rows[name] = dict(cls=cls, uses=len(idx), n_pages=n_pages, minority=minority, mu=mu,
                          mu_floor=mu_floor, floored=floored,
                          s_interval=interval, fails=fails, kept=not fails,
                          n_handles=len({label[i] for i in idx}))
    return rows
