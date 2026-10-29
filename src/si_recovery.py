"""SI: the prior fit applied to the model's own record.

The record points of Fig. 3b place each convention at a prior fitted on the
same uses whose final share is measured. To see what that does, the model is
run on the quarter-decade grid of Fig. 3b (S_LOG x DIRECTIONS, RUNS_RECOVERY
runs), and every simulated convention goes through the pipeline of the record:
the exposure of conventions.prior_exposure (the page, the writer's own uses
included, or the uses by other agents in the last 100 edits), the fit of
conventions.fit_mu, a prior at zero floored at its 95% upper bound and left out
of the bins, and the selection rule of the paper where the model allows it
(MIN_USES uses and a minority form in MIN_MINORITY of them; the model's pages
are too short for the rule on pages). For each convention the distance from the
prior is computed twice, with the true (s, pi) and with the refitted ones.
Writes cache/si_recovery.pkl.
"""
import multiprocessing
import os

import numpy as np

import model as md
from common import Report, load_cache, save_cache
from conventions import MIN_MINORITY, MIN_USES, fit_mu, floor_mu, prior_exposure
from fig4_model import normalized

RUNS_RECOVERY = 20


def refit(rec, when):
    """The record pipeline on one simulated convention: (s, pi) refitted and
    whether a prior was floored. The model's form 1 is A; the record's form 0 is A."""
    pages, agents, a = (np.array(x) for x in zip(*rec))
    idx = np.array(when)
    form = 1 - a
    label, page = dict(zip(idx, agents)), dict(zip(idx, pages))
    share = prior_exposure(idx, form, label, page)
    took_a = a.astype(float)
    mu = fit_mu(share, took_a)
    mu_floor, floored = floor_mu(share, took_a, mu)
    return mu, mu_floor, floored


def _one(args):
    par, seed = args
    grid = {(s, p): (s * p, s * (1 - p)) for s in md.S_LOG for p in md.DIRECTIONS}
    r = md.run(par, seed, conventions=grid)
    out = []
    for (s, p), rec in r['record'].items():
        a = np.array([f for _, _, f in rec])
        n, f = len(a), a.mean()
        minority = min(f, 1 - f)
        row = dict(seed=seed, s=s, pi=p, n=n, f=f, devn_true=normalized(f, s * p, s * (1 - p), n),
                   kept=n >= MIN_USES and minority >= MIN_MINORITY)
        if minority > 0:
            mu, mu_floor, floored = refit(rec, r['when'][(s, p)])
            row.update(s_hat=min(sum(mu_floor), 1.0), pi_hat=mu_floor[0] / sum(mu_floor), floored=floored,
                       devn_hat=normalized(f, *mu_floor, n))
        else:
            row.update(s_hat=np.nan, pi_hat=np.nan, floored=True, devn_hat=np.nan)
        out.append(row)
    return out


def main():
    say = Report('SI: the prior fit on the model record')
    mod = load_cache('model.pkl')
    par = mod['par']
    with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
        rows = [row for res in pool.map(_one, [(par, 9000 + k) for k in range(RUNS_RECOVERY)]) for row in res]
    for s in md.S_LOG:
        sel = [r for r in rows if r['s'] == s]
        kept = [r for r in sel if r['kept'] and not r['floored']]
        rms = lambda v: np.sqrt(np.mean(np.square(v))) if len(v) else np.nan
        say(f's = {s:<7g} true RMS {rms([r["devn_true"] for r in sel]):6.2f}; refitted, kept and identified '
            f'({len(kept)} of {len(sel)}): RMS {rms([r["devn_hat"] for r in kept]):6.2f}, '
            f'median s_hat / s {np.median([r["s_hat"] / s for r in kept]) if kept else np.nan:.2f}; '
            f'floored {sum(r["floored"] for r in sel)}, below the selection rule {sum(not r["kept"] for r in sel)}')
    save_cache('si_recovery.pkl', dict(rows=rows, runs=RUNS_RECOVERY))
    say.write()


if __name__ == '__main__':
    main()
