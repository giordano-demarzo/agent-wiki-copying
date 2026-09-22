"""The stylized model of the paper.

Identical agents arrive at uniformly random times over the central 90% of the
recorded arrivals, as many as there are handles in the population. Each makes
EDITS edits on as many distinct pages: the first on arrival, the others at
uniformly random times within the median activity span of a handle. All edits
of all agents are processed in time order, and every edit follows one rule.

    Where to write. The agent reads the last 100 lines of the feed. With
    probability c it creates a page; otherwise it picks uniformly one of the
    lines that belong to pages it has not written on, and writes on its page.
    Every edit puts its page at the top of the feed.

    How to write. Each convention is used with the usage rate, and takes form A
    with probability mu_A + (1 - mu_A - mu_B) rho, with rho the share of A among
    the uses already on the page or, if the page carries none, among the uses
    carried by the last 100 lines of the feed, and 1/2 if there are none.

    How to sign. On its first edit the agent decides each name feature by the
    same rule, with rho the share of the feature among the agents it can see:
    the authors of its page and the agents of the last 100 lines.

Everything the model takes from the record is measured on individual decisions
(analysis_pages.py, analysis_forms.py): the number of agents and their arrival
window, the number of pages an agent writes on, the activity span, the creation
probability c, the usage rate, and the base probabilities of each convention.
"""
import collections

import numpy as np

from common import FEED_LINES, co_occurrence, co_occurrence_coauthors, coauthor_matrix, volatility

S_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DIRECTIONS = (0.15, 0.3, 0.5, 0.7, 0.85)   # the range of pi = mu_A / s of the conventions
RUNS = 50
ARRIVAL_SEED = 12345


def parameters(pages, usage):
    """The model's inputs, from cache/pages.pkl and the measured usage rate."""
    inp = pages['inputs']
    return dict(n_agents=inp['n_agents'], c=inp['c'], edits=int(round(inp['pages_per'].mean())),
                span_h=inp['span_h'], window=inp['arrival_window'], usage=usage)


def run(par, seed, conventions=None, lines=FEED_LINES, choice='lines', fallback='feed', empty_page='feed',
        usage=None):
    """One run. `conventions` maps a name to (mu_A, mu_B).

    Options used by the robustness checks of the SI:
      lines       the number of feed lines an agent reads;
      choice      'lines' (the model), 'pages' (uniform among the visible pages)
                  or 'all' (uniform among all existing pages);
      fallback    'feed' (the model) or 'last30' (the last 30 uses anywhere);
      empty_page  'feed' (the model) or 'half' (rho = 1/2 on a page with no uses);
      usage       overrides the usage rate.
    """
    rng = np.random.default_rng(seed)
    conventions = conventions or {}
    usage = par['usage'] if usage is None else usage
    n, span = par['n_agents'], par['span_h'] * 3600
    births = np.sort(np.random.default_rng(ARRIVAL_SEED).uniform(*par['window'], n))
    events = sorted((t, h) for h in range(n)
                    for t in [births[h]] + list(births[h] + rng.random(par['edits'] - 1) * span))
    feed = []                                    # (page, agent), newest first
    authors = collections.defaultdict(list)      # page -> agents in order
    mine = [set() for _ in range(n)]
    seen = [None] * n                            # agents in view at the first edit
    uses = {k: collections.defaultdict(list) for k in conventions}   # page -> forms (1 = A)
    record = {k: [] for k in conventions}        # (page, agent, form) in order
    recent = {k: collections.deque() for k in conventions}           # (edit number, form), newest last
    next_page = 0
    for e, (_, h) in enumerate(events):
        window = feed[:lines]
        if choice == 'lines':
            cands = [p for p, _ in window if p not in mine[h]]
        elif choice == 'pages':
            cands = list(dict.fromkeys(p for p, _ in window if p not in mine[h]))
        else:
            cands = [p for p in authors if p not in mine[h]]
        if not cands or rng.random() < par['c']:
            next_page += 1
            pg = next_page
        else:
            pg = cands[rng.integers(len(cands))]
        if seen[h] is None:
            seen[h] = list(dict.fromkeys(authors[pg] + [a for _, a in window]))
        mine[h].add(pg)
        for k, (mu_a, mu_b) in conventions.items():
            if rng.random() >= usage:
                continue
            if fallback == 'feed':
                while recent[k] and recent[k][0][0] < e - lines:
                    recent[k].popleft()
                in_feed = [f for _, f in recent[k]]
            else:
                in_feed = [f for _, f in list(recent[k])[-30:]]
            src = uses[k][pg] if (uses[k][pg] or empty_page == 'half') else in_feed
            rho = np.mean(src) if len(src) else 0.5
            form = int(rng.random() < mu_a + (1 - mu_a - mu_b) * rho)
            uses[k][pg].append(form)
            record[k].append((pg, h, form))
            recent[k].append((e, form))
        authors[pg].append(h)
        feed.insert(0, (pg, h))
        del feed[1000:]
    return dict(n=n, seen=seen, record=record,
                page_agents=[set(v) for v in authors.values()],
                handles_per_page=np.array([len(set(v)) for v in authors.values()]))


def names(result, mu, rng):
    """Replay one name feature through a run: each agent, in order of arrival,
    decides it from the agents it could see at its first edit."""
    v = np.zeros(result['n'])
    for h in range(result['n']):
        s = result['seen'][h]
        rho = v[s].mean() if len(s) else 0.5
        v[h] = rng.random() < mu[0] + (1 - mu[0] - mu[1]) * rho
    return v


def form_statistics(rec):
    """Co-occurrence of the minority form, volatility on each agent's first use
    and on all uses, and final share of A, for the uses of one convention."""
    if not rec:
        return dict(gap=np.nan, vol=np.nan, vol_all=np.nan, final=np.nan)
    pages, agents, a = (np.array(x) for x in zip(*rec))
    seen, first = set(), []
    for k, h in enumerate(agents):
        if h not in seen:
            seen.add(h)
            first.append(k)
    return dict(gap=co_occurrence(pages, a), vol=volatility(a[first], 50), vol_all=volatility(a, 50),
                final=a.mean())


def name_statistics(v, A):
    return dict(gap=co_occurrence_coauthors(v, A), vol=volatility(v, 100), final=v.mean())


def page_statistics(result):
    hpp = result['handles_per_page']
    return dict(p5=np.mean(hpp >= 5), p10=np.mean(hpp >= 10), p20=np.mean(hpp >= 20), p40=np.mean(hpp >= 40),
                max=hpp.max(), n_pages=len(hpp), ccdf=np.array([np.mean(hpp >= k) for k in range(1, 300)]))


def _grid_run(args):
    par, seed, options = args
    grid = {(s, p): (s * p, s * (1 - p)) for s in S_GRID for p in DIRECTIONS}
    r = run(par, seed, conventions=grid, **options)
    A = coauthor_matrix(r['page_agents'], r['n'])
    rng = np.random.default_rng(500 + seed)
    out = {('pages',): page_statistics(r)}
    for key, mu in grid.items():
        out[('form',) + key] = form_statistics(r['record'][key])
        out[('name',) + key] = name_statistics(names(r, mu, rng), A)
    return out


def _pages_run(args):
    par, seed, options = args
    return page_statistics(run(par, seed, **options))


def _fitted_run(args):
    par, seed, conventions = args
    r = run(par, seed, conventions=conventions)
    return {k: form_statistics(r['record'][k]) for k in conventions}


def collect(results):
    """List of per-run dicts -> key -> statistic -> array over runs."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for res in results:
        for key, stats in res.items():
            for s, v in stats.items():
                out[key][s].append(v)
    return {k: {s: np.array(v) for s, v in d.items()} for k, d in out.items()}


def sweep(par, pool, runs=RUNS, **options):
    """The prior-strength axis: every (s, pi) of the grid, for forms and names."""
    return collect(pool.map(_grid_run, [(par, seed, options) for seed in range(runs)]))


def pages_only(par, pool, runs, **options):
    return collect([{'pages': r} for r in pool.map(_pages_run, [(par, seed, options) for seed in range(runs)])])['pages']


def fitted(par, pool, conventions, runs=RUNS):
    """Each convention at its own fitted base probabilities."""
    return collect(pool.map(_fitted_run, [(par, seed, conventions) for seed in range(runs)]))


def curve(sw, kind, stat, s=S_GRID, directions=DIRECTIONS, q=(10, 90)):
    """Median and percentiles over runs and directions, along s."""
    med, lo, hi = [], [], []
    for s_ in s:
        v = np.concatenate([sw[(kind, s_, p)][stat] for p in directions]).astype(float)
        v = v[np.isfinite(v)]
        med.append(np.median(v) if len(v) else np.nan)
        lo.append(np.percentile(v, q[0]) if len(v) else np.nan)
        hi.append(np.percentile(v, q[1]) if len(v) else np.nan)
    return np.array(med), np.array(lo), np.array(hi)


def distance_curve(sw, kind, q=(10, 90)):
    """|final share - pi| over runs and directions, along s."""
    med, lo, hi = [], [], []
    for s_ in S_GRID:
        d = np.concatenate([np.abs(sw[(kind, s_, p)]['final'] - p) for p in DIRECTIONS])
        med.append(np.median(d))
        lo.append(np.percentile(d, q[0]))
        hi.append(np.percentile(d, q[1]))
    return np.array(med), np.array(lo), np.array(hi)
