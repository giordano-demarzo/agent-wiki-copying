"""Every run of the model used in the paper and the SI. Writes cache/model.pkl.

  main         the prior-strength axis of Fig. 4b-d (12 runs per point)
  pages        the distribution of handles per page of Fig. 4a (200 runs)
  windows      the model reading 30, 100, 300 or 1,000 feed lines (Fig. S3b)
  nulls        uniform choice among visible pages, or among all pages (S3)
  usage        the axis with usage rates of 0.5 and 0.1 (S7)
  last30       the fallback of an empty page taken from the last 30 uses (S7)
  half         no fallback: an empty page starts from rho = 1/2 (S7)
  fitted       each convention at its own base probabilities (Fig. S14)
"""
import multiprocessing
import os

import numpy as np

import model as md
from common import Report, load_cache, save_cache


def fmt(v, d=3):
    return '[' + ', '.join(f'{x:.{d}f}' for x in v) + ']'


def main():
    say = Report('The model')
    pages = load_cache('pages.pkl')
    forms = load_cache('forms.pkl')
    usage = float(np.mean(list(forms['usage'].values())))
    par = md.parameters(pages, usage)
    say(f'parameters: {par["n_agents"]} agents over {(par["window"][1] - par["window"][0]) / 86400:.1f} days, '
        f'{par["edits"]} edits on distinct pages within {par["span_h"]:.2f} h, c = {par["c"]:.3f}, '
        f'usage rate {par["usage"]:.3f}, {md.RUNS} runs per point of the axis')
    out = dict(par=par)
    with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
        out['pages'] = md.pages_only(par, pool, 200)
        p = out['pages']
        obs = pages['inputs']['handles_per_page']
        say(f'Fig. 4a, 200 runs: P(>=5,10,20,40) median {fmt([np.median(p[k]) for k in ("p5", "p10", "p20", "p40")])} '
            f'(observed {fmt([np.mean(obs >= k) for k in (5, 10, 20, 40)])}); 5-95% of P(>=20) '
            f'{fmt(np.percentile(p["p20"], [5, 95]))}; largest page median {np.median(p["max"]):.0f}, 5-95% '
            f'{np.percentile(p["max"], 5):.0f}-{np.percentile(p["max"], 95):.0f} (observed {obs.max()}); '
            f'pages {np.mean(p["n_pages"]):.0f} (observed {len(obs)})')

        out['windows'] = {m: md.pages_only(par, pool, 10, lines=m) for m in (30, 100, 300, 1000)}
        for m, w in out['windows'].items():
            say(f'window {m:5} lines, 10 runs: P(>=5,10,20,40) {fmt([np.median(w[k]) for k in ("p5", "p10", "p20", "p40")])}, '
                f'largest page {np.median(w["max"]):.0f}')
        out['nulls'] = {c: md.pages_only(par, pool, 10, choice=c) for c in ('pages', 'all')}
        for c, w in out['nulls'].items():
            say(f'uniform choice among {"visible" if c == "pages" else "all"} pages, 10 runs: P(>=10,20,40) '
                f'{fmt([np.median(w[k]) for k in ("p10", "p20", "p40")])}, largest page {np.median(w["max"]):.0f}')

        out['main'] = md.sweep(par, pool)
        out['usage'] = {u: md.sweep(par, pool, usage=u) for u in (0.5, 0.1)}
        out['last30'] = md.sweep(par, pool, fallback='last30')
        out['half'] = md.sweep(par, pool, empty_page='half')
        conv = {n: tuple(forms['fits'][n]) for n in forms['keep']}
        out['fitted'] = md.fitted(par, pool, conv)

    sw = out['main']
    for kind in ('form', 'name'):
        for stat in ('gap', 'vol'):
            med = md.curve(sw, kind, stat)[0]
            say(f'axis, {kind}, {stat}: ' + ' '.join(f'{s:g}:{m:.2f}' for s, m in zip(md.S_GRID, med)))
        say(f'axis, {kind}, distance from the prior: '
            + ' '.join(f'{s:g}:{m:.3f}' for s, m in zip(md.S_GRID, md.distance_curve(sw, kind)[0])))

    # S6: how much the outcome depends on the run (pi = 1/2)
    for s in (0.02, 0.1, 0.3, 0.6, 1.0):
        f = sw[('form', s, 0.5)]['final']
        say(f'final share of a form, s = {s:g}, pi = 1/2: from {f.min():.2f} to {f.max():.2f}')
    for kind in ('form', 'name'):
        say(f'2 s.d. of the final share across runs, {kind}: '
            + ' '.join(f'{s:g}:{2 * np.std(sw[(kind, s, 0.5)]["final"]):.2f}' for s in md.S_GRID))

    # S7: the direction of the prior
    for kind in ('form', 'name'):
        for stat in ('gap', 'vol'):
            for s in (0.2, 0.4, 0.6):
                say(f'direction, {kind}, {stat}, s = {s:g}: '
                    + ', '.join(f'pi {p}: {np.nanmedian(sw[(kind, s, p)][stat]):.2f}' for p in md.DIRECTIONS))
            across_pi = np.nanmean([np.nanstd([np.nanmedian(sw[(kind, s, p)][stat]) for p in md.DIRECTIONS])
                                    for s in md.S_GRID if s > 0])
            across_runs = np.nanmean([np.nanstd(sw[(kind, s, 0.5)][stat]) for s in md.S_GRID if s > 0])
            say(f'direction, {kind}, {stat}: s.d. of the median across directions {across_pi:.2f}, '
                f's.d. across runs at pi = 1/2 {across_runs:.2f} (means over s > 0)')

    # S7: the usage rate and the fallback of an empty page
    at = lambda w, kind, stat, s: np.nanmedian(np.concatenate([w[(kind, s, p)][stat] for p in md.DIRECTIONS]))
    dist = lambda w, s: np.median(np.concatenate([np.abs(w[('form', s, p)]['final'] - p) for p in md.DIRECTIONS]))
    for s in (0.2, 0.4):
        say(f'usage rate, s = {s:g}: co-occurrence {at(out["usage"][0.5], "form", "gap", s):.2f} (0.5), '
            f'{at(sw, "form", "gap", s):.2f} ({usage:.2f}), {at(out["usage"][0.1], "form", "gap", s):.2f} (0.1); '
            f'volatility {at(out["usage"][0.5], "form", "vol", s):.2f}, {at(sw, "form", "vol", s):.2f}, '
            f'{at(out["usage"][0.1], "form", "vol", s):.2f}')
    for key, lab in (('last30', 'last 30 uses anywhere'), ('half', 'rho = 1/2 on an empty page')):
        w = out[key]
        say(f'{lab}, s = 0.2: co-occurrence {at(w, "form", "gap", 0.2):.2f} (model {at(sw, "form", "gap", 0.2):.2f}), '
            f'volatility {at(w, "form", "vol", 0.2):.2f} ({at(sw, "form", "vol", 0.2):.2f}), distance from the prior '
            f'{dist(w, 0.2):.3f} ({dist(sw, 0.2):.3f}); 2 s.d. of the final share at s = 0.02, pi = 1/2 '
            f'{2 * np.std(w[("form", 0.02, 0.5)]["final"]):.2f} ({2 * np.std(sw[("form", 0.02, 0.5)]["final"]):.2f})')

    # Fig. S14: each convention at its own base probabilities, against the record
    axis = {e['name']: e for e in forms['axis']}
    for stat, log in (('gap', True), ('vol', False), ('final', False)):
        obs, med, inside = [], [], []
        for n, runs in out['fitted'].items():
            o = axis[n]['f'] if stat == 'final' else axis[n][stat]
            m = runs[stat][np.isfinite(runs[stat])]
            if not np.isfinite(o) or len(m) < 6 or (log and o <= 0):
                continue
            obs.append(o)
            med.append(np.median(m))
            inside.append(np.percentile(m, 10) <= o <= np.percentile(m, 90))
        obs, med = np.array(obs), np.array(med)
        tx, ty = (np.log(obs), np.log(med)) if log else (obs, med)
        mean = (lambda x: np.exp(x.mean())) if log else np.mean
        say(f'each convention, {stat}: n={len(obs)}, correlation {np.corrcoef(tx, ty)[0, 1]:.2f}, mean observed '
            f'{mean(tx):.2f}, mean model {mean(ty):.2f}, observed inside the 10-90% band of its runs {np.mean(inside):.2f}')
    va = [np.nanmedian(r['vol_all']) for n, r in out['fitted'].items() if np.isfinite(axis[n]['vol'])]
    vf = [np.nanmedian(r['vol']) for n, r in out['fitted'].items() if np.isfinite(axis[n]['vol'])]
    say(f'each convention, model volatility: all uses {np.mean(va):.2f}, first uses {np.mean(vf):.2f}')

    save_cache('model.pkl', out)
    say.write()


if __name__ == '__main__':
    main()
