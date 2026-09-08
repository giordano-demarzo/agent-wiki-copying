"""What to call yourself: how agents build their handle. Produces cache/names.pkl.

Two things are computed here.

1. The copying response. For eight name tokens, whether a newcomer's handle
   carries the token, against the share of names carrying it in what the
   newcomer could see: the last 30 distinct handles that wrote before it, the
   30 before those, and the authors of the page it first writes on.
2. The neutral model. Names are strings of capitalised pieces; the model has
   each newcomer copy pieces from the last 30 names, with a rate of
   innovation as its only parameter.
"""
import collections
import pickle
import re

import numpy as np

from common import Dataset, Report, T, cache_path

FEED_NAMES = 30      # how many distinct earlier handles a newcomer is assumed to see
MIN_PAGE_AUTHORS = 3
PIECES_PER_NAME = 3  # round(4128 / 1201), fixed from the data, not fitted

TOKENS = {
    'Scout': r'Scout',
    'Watcher': r'Watch',
    'Helper': r'Helper',
    'Research': r'Research',
    'Agent': r'Agent',
    'Coord': r'Coord',
    'OAI/OpenAI': r'OAI|OpenAI',
    'date': r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d',
}

PIECE = re.compile(r'[A-Z][a-z]+|[A-Z]{2,}(?![a-z])')


def pieces(handle):
    """The capitalised pieces of a name: OpenAIResearchHelperMay08 gives
    Open, AI, Research, Helper, May."""
    return [p for p in PIECE.findall(handle) if not re.fullmatch(r'\d+', p)]


def exposure_windows(ds, k=FEED_NAMES):
    """For each handle: the last k distinct handles seen, the k before those,
    and the handles that had already edited the page of its first edit."""
    ts = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    labels = [r['label'] for r in ds.revs_all]
    page_ids = [r['page_id'] for r in ds.revs_all]
    out = {}
    for handle in ds.handles:
        j = np.searchsorted(ts, ds.birth[handle].timestamp())
        recent = []
        for i in range(j - 1, -1, -1):
            if labels[i] != handle and labels[i] not in recent:
                recent.append(labels[i])
            if len(recent) >= 2 * k:
                break
        first_page = ds.by_label[handle][0]['page_id']
        authors = list({labels[i] for i in range(j)
                        if page_ids[i] == first_page and labels[i] != handle})
        out[handle] = (recent[:k], recent[k:2 * k], authors)
    return out


def token_records(ds, windows):
    """One record per (token, handle): the three exposures and the outcome."""
    feed, older, page, carries = [], [], [], []
    for pattern in TOKENS.values():
        r = re.compile(pattern)
        for handle in ds.handles:
            recent, before, authors = windows[handle]
            if len(before) < 10:
                continue
            share = lambda names: np.mean([bool(r.search(h)) for h in names])
            feed.append(share(recent))
            older.append(share(before))
            page.append(share(authors) if len(authors) >= MIN_PAGE_AUTHORS else np.nan)
            carries.append(int(bool(r.search(handle))))
    return (np.array(feed), np.array(older), np.array(page), np.array(carries))


def ols(columns, y):
    """Ordinary least squares with an intercept; returns coefficients and
    standard errors."""
    x = np.column_stack([np.ones(len(y))] + columns)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    s2 = resid @ resid / (len(y) - x.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(x.T @ x)))
    return beta, se


def neutral_model(n_handles, m_pieces, k, eps, seed=0):
    """Neutral copying with innovation.

    Each newcomer draws m_pieces pieces. Each is copied uniformly from the
    pieces of the last k names with probability 1 - eps, and is a piece never
    used before with probability eps.
    """
    rng = np.random.default_rng(seed)
    names, next_id = [], 0
    for i in range(n_handles):
        pool = [p for ps in names[max(0, i - k):i] for p in ps]
        drawn = []
        for _ in range(m_pieces):
            if pool and rng.random() > eps:
                drawn.append(pool[rng.integers(len(pool))])
            else:
                next_id += 1
                drawn.append(f'new{next_id}')
        names.append(drawn)
    return names


def main(eps=0.10):
    say = Report('What to call yourself (Figure 3)')
    ds = Dataset()

    per_name = [pieces(h) for h in ds.handles]
    counts = collections.Counter(p for ps in per_name for p in ps)
    n_pieces = sum(len(ps) for ps in per_name)
    say(f'{len(ds.handles)} handles, {n_pieces} name pieces '
        f'({n_pieces / len(ds.handles):.2f} per name), {len(counts)} distinct')
    say(f'  most used: {counts.most_common(8)}')
    fresh = np.mean([p not in {q for ps in per_name[:i] for q in ps}
                     for i, ps in enumerate(per_name) for p in ps])
    say(f'  share of pieces that had never appeared in an earlier name: {fresh:.3f}')

    # daily turnover of the five most used pieces
    days = sorted({ds.birth[h].strftime('%d %b') for h in ds.handles},
                  key=lambda s: (s[3:], s[:2]))
    tops = []
    for day in days:
        born = [h for h in ds.handles if ds.birth[h].strftime('%d %b') == day]
        if len(born) < 50:
            continue
        c = collections.Counter(p for h in born for p in pieces(h))
        tops.append((day, [p for p, _ in c.most_common(5)]))
    turnover = [len(set(tops[i][1]) - set(tops[i - 1][1])) for i in range(1, len(tops))]
    say(f'  top-5 pieces per day: {tops}')
    say(f'  new entries in the daily top five: {turnover}, mean {np.mean(turnover):.2f}')

    windows = exposure_windows(ds)
    feed, older, page, carries = token_records(ds, windows)
    b_time, se_time = ols([feed, older], carries)
    m = ~np.isnan(page)
    b_src, se_src = ols([feed[m], page[m]], carries[m])
    say(f'joint fit, how far back: last {FEED_NAMES} names {b_time[1]:.2f}+-{se_time[1]:.2f}, '
        f'the {FEED_NAMES} before those {b_time[2]:.2f}+-{se_time[2]:.2f} (n={len(carries)})')
    say(f'joint fit, which source: feed {b_src[1]:.2f}+-{se_src[1]:.2f}, '
        f"page's authors {b_src[2]:.2f}+-{se_src[2]:.2f} (n={m.sum()})")

    observed = np.array(sorted(counts.values(), reverse=True))
    runs = []
    for seed in range(20):
        names = neutral_model(len(ds.handles), PIECES_PER_NAME, FEED_NAMES, eps, seed)
        c = collections.Counter(p for ps in names for p in ps)
        runs.append(np.array(sorted(c.values(), reverse=True)))
    say(f'neutral model, eps={eps}: distinct pieces {np.mean([len(f) for f in runs]):.0f} '
        f'(observed {len(counts)}), most used piece {np.mean([f[0] for f in runs]):.0f} '
        f'(observed {observed[0]})')
    ks = (2, 5, 20, 100)
    say(f'  P(a piece is used by >= {ks} handles): '
        f'observed {[round(float(np.mean(observed >= k)), 3) for k in ks]}, '
        f'model {[round(float(np.median([np.mean(f >= k) for f in runs])), 3) for k in ks]}')

    with open(cache_path('names.pkl'), 'wb') as fh:
        pickle.dump(dict(feed=feed, older=older, page=page, carries=carries,
                         b_time=b_time, se_time=se_time, b_src=b_src, se_src=se_src,
                         counts=counts, runs=runs, eps=eps, tops=tops), fh)
    say.write()


if __name__ == '__main__':
    main()
