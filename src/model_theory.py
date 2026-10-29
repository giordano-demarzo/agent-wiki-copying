"""Exact moments of the copy model.

The rule is linear in rho, and rho is an average of earlier uses (on the page,
in the feed, or of the agents in view). Write v_i for the form of use i (1 = A),
pi = mu_A / s and a = 1 - s. Then

    E[v_i | past] = pi + a (sum_j w_ij v_j - pi)       (rows of w sum to 1)
                  = pi + a (1/2 - pi)                    (no use to read: rho = 1/2)

so the means m_i = E[v_i] - pi and the covariances C_ij = Cov(v_i, v_j) close:

    m_i  = a sum_j w_ij m_j            (or a (1/2 - pi))
    C_ij = a sum_k w_ik C_kj           (j < i)
    C_ii = (pi + m_i)(1 - pi - m_i)    (v_i is 0 or 1)

Who reads whom (w) does not depend on the forms, so one simulated copy graph
gives the first two moments of every statistic of the forms exactly, for every
(s, pi), without simulating a single form. The final share, its distance from
the prior and the volatility over blocks follow from m and C (the distance
assumes a Gaussian final share, which fails only near s = 0).

Closed forms (mean field on the same graph). A use read by a share k of later
readers passes on a(k) of its deviation; with every later use carrying the
average influence 1/s, the final share of N uses has

    Var(f) = pi (1 - pi) (1 + (1 - s)^2 V) / (N s^2),     V = Var(k) / E[k]^2,

and, with an exponential copying lag of mean tau (in uses), the spectrum of the
forms is pi (1 - pi) (1 + a^2 V |K|^2) / |1 - a K|^2 with K = 1 / (1 + i w tau),
which gives the volatility over K blocks of B uses. The pages: every feed line
is picked by later edits Poisson(1 - c) times, so a page is the total progeny
of a Galton-Watson process, Borel distributed with mean 1/c:
P(k) = exp(-mu k) (mu k)^(k-1) / k!, mu = 1 - c.

    python model_theory.py      -> cache/model_theory.pkl (the exact moments on 20 copy graphs, used by fig4_model.py)
"""
import collections
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import model as md
from common import FEED_LINES, load_cache, save_cache

GRAPHS = 20
# the prior-strength axis of the theory: finer than the simulation grid, same ends
S_FINE = tuple(np.round(np.concatenate([np.arange(0.02, 0.1, 0.02), np.arange(0.1, 1.0001, 0.05)]), 3))


def copy_graph(par, seed, lines=FEED_LINES):
    """The model's page dynamics (model.run) for one generic convention, keeping,
    for every use, the earlier uses it reads, and for every agent the agents it
    sees at its first edit."""
    rng = np.random.default_rng(seed)
    n, span = par['n_agents'], par['span_h'] * 3600
    births = np.sort(np.random.default_rng(md.ARRIVAL_SEED).uniform(*par['window'], n))
    events = sorted((t, h) for h in range(n) for t in [births[h]] + list(births[h] + rng.random(par['edits'] - 1) * span))
    feed, mine, seen = [], [set() for _ in range(n)], [None] * n
    page_uses = collections.defaultdict(list)
    recent = collections.deque()
    src, agent = [], []
    pages = 0
    for e, (_, h) in enumerate(events):
        window = feed[:lines]
        cands = [p for p, _ in window if p not in mine[h]]
        if not cands or rng.random() < par['c']:
            pages += 1
            pg = pages
        else:
            pg = cands[rng.integers(len(cands))]
        if seen[h] is None:
            seen[h] = list(dict.fromkeys(a for _, a in window))
        mine[h].add(pg)
        if rng.random() < par['usage']:
            while recent and recent[0][0] < e - lines:
                recent.popleft()
            src.append(list(page_uses[pg]) or [u for _, u in recent])
            agent.append(h)
            page_uses[pg].append(len(src) - 1)
            recent.append((e, len(src) - 1))
        feed.insert(0, (pg, h))
        del feed[1000:]
    first = sorted({h: i for i, h in reversed(list(enumerate(agent)))}.values())
    return dict(uses=src, first=np.array(first), names=seen)


def weights(sources):
    """Sparse rows of w: (indices, weight) per item, None when there is nothing to read."""
    return [(np.array(s), 1.0 / len(s)) if len(s) else None for s in sources]


def moments(W, s, pi):
    """Means m (relative to pi) and covariance C of the forms."""
    a, N = 1 - s, len(W)
    m, C = np.zeros(N), np.zeros((N, N))
    for i, w in enumerate(W):
        if w is None:
            m[i] = a * (0.5 - pi)
        else:
            idx, wt = w
            m[i] = a * wt * m[idx].sum()
            C[i, :i] = a * wt * C[idx, :i].sum(0)
            C[:i, i] = C[i, :i]
        q = pi + m[i]
        C[i, i] = q * (1 - q)
    return m, C


def statistics(m, C, pi, sel, block):
    """Distance from the prior (mean and sd of the final share) and the expected
    volatility over consecutive blocks of the uses in `sel`."""
    msel, Csel = m[sel], C[np.ix_(sel, sel)]
    n = len(sel)
    mean_f, var_f = m.mean(), C.mean()
    K = n // block
    B = np.zeros((K, n))
    for b in range(K):
        B[b, b * block:(b + 1) * block] = 1.0 / block
    D = B - B.mean(0)                                  # block mean minus mean over blocks
    num = (np.trace(D @ Csel @ D.T) + ((D @ msel) ** 2).sum()) / K
    f_sel, v_sel = pi + msel.mean(), Csel.mean()
    den = (f_sel * (1 - f_sel) - v_sel) / block        # E[f (1 - f)] / block
    return pi + mean_f, np.sqrt(var_f), np.sqrt(num / den) if den > 0 else np.nan


def folded_quantiles(mus, sds, qs=(0.1, 0.5, 0.9)):
    """Quantiles of |X| for X an equal mixture of Gaussians N(mu, sd)."""
    top = max(2.0, 1.5 * np.max(np.abs(mus) + 5 * np.asarray(sds)))
    grid = np.logspace(-5, np.log10(top), 6000)
    cdf = np.mean([stats.norm.cdf(grid, mu, sd) - stats.norm.cdf(-grid, mu, sd) for mu, sd in zip(mus, sds)], 0)
    return [np.interp(q, cdf, grid) for q in qs]


def _theory_graph(args):
    """One copy graph: (f - pi, sd of f, volatility, pi) for every kind, s and direction."""
    par, seed, s_grid = args
    g = copy_graph(par, seed)
    out = {}
    for kind in ('form', 'name'):
        W = weights(g['uses'] if kind == 'form' else g['names'])
        sel = g['first'] if kind == 'form' else np.arange(len(W))
        block = 50 if kind == 'form' else 100
        res = {}
        for s in s_grid:
            res[s] = [(f - p, sd, vol, p) for p in md.DIRECTIONS
                      for f, sd, vol in [statistics(*moments(W, s, p), p, sel, block)]]
        out[kind] = res
    return out


def theory(par, graphs=GRAPHS, s_grid=S_FINE, pool=None):
    """The exact moments on `graphs` copy graphs: kind -> s -> array over graphs and
    directions of (f - pi, sd of f, volatility, pi)."""
    jobs = [(par, 1000 + g, tuple(s_grid)) for g in range(graphs)]
    parts = pool.map(_theory_graph, jobs) if pool is not None else [_theory_graph(j) for j in jobs]
    out = {'s_grid': tuple(s_grid)}
    for kind in ('form', 'name'):
        out[kind] = {s: np.array([row for part in parts for row in part[kind][s]]) for s in s_grid}
    return out


def normalized_quantiles(th, kinds, N, qs=(0.1, 0.5, 0.9)):
    """Quantiles of |f - pi| / sqrt(pi (1 - pi) / N), the distance from the prior in
    units of the fluctuation of N independent choices, pooled over graphs and
    directions (Gaussian final share), along the s grid of the theory. `kinds` is
    one kind or several, pooled; N maps each kind to its number of uses."""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    out = []
    for s in th['s_grid']:
        mus, sds = [], []
        for kind in kinds:
            r = th[kind][s]
            sig0 = np.sqrt(r[:, 3] * (1 - r[:, 3]) / N[kind])
            mus.append(r[:, 0] / sig0)
            sds.append(np.maximum(r[:, 1], 1e-9) / sig0)
        out.append(folded_quantiles(np.concatenate(mus), np.concatenate(sds), qs))
    return np.array(th['s_grid']), np.array(out)


def graph_constants(par, graphs=10):
    """N (uses, or agents for names), V and tau of the copy graph, per kind."""
    out = {}
    G = [copy_graph(par, 1000 + g) for g in range(graphs)]
    for kind in ('form', 'name'):
        ks, lags, Ns = [], [], []
        for g in G:
            src = g['uses'] if kind == 'form' else g['names']
            k = np.zeros(len(src))
            for i, r in enumerate(src):
                if len(r):
                    k[r] += 1.0 / len(r)
                    lags.append(i - np.mean(r))
            ks.append(k[:int(0.7 * len(src))])          # uses with enough later readers
            Ns.append(len(src))
        k = np.concatenate(ks)
        out[kind] = dict(N=np.mean(Ns), V=k.var() / k.mean() ** 2, tau=np.mean(lags))
    return out


def closed_distance(s, N, V, directions=md.DIRECTIONS, q=(0.1, 0.5, 0.9)):
    """Quantiles of |f - pi| over the prior directions (Gaussian final share)."""
    p = np.array(directions)
    sd = np.sqrt(p * (1 - p) * (1 + (1 - s) ** 2 * V) / (N * s ** 2))
    return folded_quantiles(np.zeros(len(p)), sd, q)


def closed_volatility(s, N, B, V, tau):
    """Median volatility over the K = N // B blocks (np.std of the block shares
    over the binomial sd), from the spectrum of the forms."""
    K, a = int(N // B), 1 - s
    k = np.arange(1, K)
    Kw = 1 / (1 + 1j * 2 * np.pi * np.minimum(k, K - k) / (K * B) * tau)
    S = (1 + a ** 2 * V * abs(Kw) ** 2) / abs(1 - a * Kw) ** 2
    chi = np.sqrt(stats.chi2.median(K - 1) / K)        # median of the sample sd at K blocks
    return chi * np.sqrt(S.mean())


def borel_ccdf(c, kmax=400):
    """P(K >= k), k = 1..kmax, for the total progeny with Poisson(1 - c) offspring."""
    from scipy.special import gammaln
    mu, k = 1 - c, np.arange(1, kmax + 1)
    p = np.exp(-mu * k + (k - 1) * np.log(mu * k) - gammaln(k + 1))
    return k, np.cumsum(p[::-1])[::-1]


def curves(th, kind):
    dist, vol = [], []
    for s in th['s_grid']:
        r = th[kind][s]
        dist.append(folded_quantiles(r[:, 0], np.maximum(r[:, 1], 1e-9)))
        vol.append(np.nanmedian(r[:, 2]))
    return np.array(dist), np.array(vol)


def main():
    """Computes the exact moments on GRAPHS copy graphs and saves cache/model_theory.pkl,
    then prints the normalized distance against the simulation at the grid points."""
    import multiprocessing
    import os
    mod = load_cache('model.pkl')
    par, sw = mod['par'], mod['main']
    with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
        th = theory(par, pool=pool)
    save_cache('model_theory.pkl', th)
    N = {'form': round(par['n_agents'] * par['edits'] * par['usage']), 'name': par['n_agents']}
    for kind in ('form', 'name'):
        s_th, q = normalized_quantiles(th, kind, N)
        print(f'{kind}: normalized distance from the prior, median (theory / simulation)')
        for s in md.S_GRID:
            if s == 0:
                continue
            sim = np.concatenate([np.abs(sw[(kind, s, p)]['final'] - p) / np.sqrt(p * (1 - p) / N[kind])
                                  for p in md.DIRECTIONS])
            j = np.argmin(np.abs(s_th - s))
            print(f'  s={s:<5} {q[j, 1]:6.2f} / {np.median(sim):6.2f}')


if __name__ == '__main__':
    main()
