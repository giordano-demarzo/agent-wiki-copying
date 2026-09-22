"""Identification of copying: linear probability models in which the exposure
is split by task (own or other) and by visibility (on the page or in the last
100 feed lines, against older). Each model is estimated with a baseline fixed
effect and again with fixed effects for the task family and the three-hour
block. Writes cache/identification.pkl (Table 1, Tables S5-S7, Fig. S15) and,
with --full, cache/identification_full.pkl (Table S1).
"""
import collections
import sys

import numpy as np

import names as nm
from common import FEED_LINES, Dataset, Report, T, block_of, demean, fit, load_cache, save_cache, share
from conventions import sequence

MATES_H = nm.MATES_H


def form_records(ds, keep):
    """One record per handle's first use of a convention: the outcome (form A)
    and the share of A on the page, among same-task uses in the feed, among
    same-task uses of the previous six hours that had scrolled off, and among
    other tasks' uses in the feed."""
    ts_all = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    pos = {id(r): k for k, r in enumerate(ds.revs_all)}
    rows = []
    for name in keep:
        idx, form = sequence(ds.revs, name)
        use_pos = np.array([pos[id(ds.revs[i])] for i in idx])
        use_lab = [ds.revs[i]['label'] for i in idx]
        use_pg = [ds.revs[i]['page_id'] for i in idx]
        use_fam = np.array([ds.first_family[h] for h in use_lab])
        a = (form == 0).astype(float)
        seen = set()
        for k in range(len(idx)):
            h, pg, p = use_lab[k], use_pg[k], use_pos[k]
            if h in seen:
                continue
            seen.add(h)
            t = ts_all[p]
            earlier = [j for j in range(k) if use_pos[j] < p and use_lab[j] != h]
            on_page = [j for j in earlier if use_pg[j] == pg]
            in_feed = [j for j in earlier if p - FEED_LINES <= use_pos[j] and use_pg[j] != pg]
            same_feed = [j for j in in_feed if use_fam[j] == use_fam[k]]
            other_feed = [j for j in in_feed if use_fam[j] != use_fam[k]]
            same_out = [j for j in earlier if use_pos[j] < p - FEED_LINES and ts_all[use_pos[j]] >= t - MATES_H * 3600
                        and use_fam[j] == use_fam[k] and use_pg[j] != pg]
            rows.append((name, use_fam[k], block_of(t), h, a[k], share(a, on_page, 2), share(a, same_feed, 2),
                         share(a, same_out, 2), share(a, other_feed, 3)))
    c = list(zip(*rows))
    return dict(convention=np.array(c[0]), family=np.array(c[1]), block=np.array(c[2]), handle=np.array(c[3]),
                y=np.array(c[4]), X=np.column_stack([np.array(x, float) for x in c[5:]]))


def page_records(ds):
    """One record per candidate page within a decision to append to a task page
    new to the handle. Candidates are the existing pages of the handle's own
    task and the visible pages of other tasks."""
    ts_all = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    pid_all = [r['page_id'] for r in ds.revs_all]
    fam_page = {pid: p['page_family'] for pid, p in ds.pages.items()}
    seen = collections.defaultdict(set)
    created_at, handles_on, rows, n_dec = {}, collections.defaultdict(set), [], 0
    for k, r in enumerate(ds.revs_all):
        if not ds.is_task(r) or r['label'] not in ds.index:
            continue
        h, pg = r['label'], r['page_id']
        created_at.setdefault(pg, k)
        if pg not in seen[h]:
            seen[h].add(pg)
            if r['diff_base_reason'] != 'page_created' and created_at[pg] < k:
                window = collections.Counter(pid_all[max(0, k - FEED_LINES):k])
                own = ds.first_family[h]
                cands = ({p for p in created_at if created_at[p] < k and fam_page.get(p) == own}
                         | {p for p in window if fam_page.get(p) in ds.task_families})
                for p in cands:
                    rows.append((n_dec, own, block_of(ts_all[k]), int(p == pg), window.get(p, 0) / FEED_LINES,
                                 int(p in window), int(fam_page.get(p) == own), np.log1p(len(handles_on[p]))))
                n_dec += 1
        handles_on[pg].add(h)
    c = list(zip(*rows))
    return dict(decision=np.array(c[0]), family=np.array(c[1]), block=np.array(c[2]), y=np.array(c[3], float),
                share=np.array(c[4]), visible=np.array(c[5]), own=np.array(c[6]), log_handles=np.array(c[7]),
                n_decisions=n_dec)


def fe(*arrays):
    """One grouping variable out of several: their combination."""
    return np.array(['|'.join(map(str, t)) for t in zip(*arrays)])


def fit_forms(r):
    m = np.all(~np.isnan(r['X']), axis=1)
    X, y = [r['X'][m, j] for j in range(4)], r['y'][m]
    conv, tb = r['convention'][m], fe(r['family'][m], r['block'][m])
    out = dict(n=int(m.sum()))
    out['b0'], out['se0'], out['r2_0'] = fit(X, y, (conv,))
    out['b1'], out['se1'], out['r2_1'] = fit(X, y, (conv, tb))
    out['bx'], out['sex'], out['r2_x'] = fit(X, y, (fe(conv, tb),))
    return out


def fit_names(r):
    X = np.column_stack([r['page'], r['feed'], r['out'], r['other']])
    m = np.all(~np.isnan(X), axis=1)
    cols, y = [X[m, j] for j in range(4)], r['y'][m]
    feat, tb, handle = r['feature'][m], fe(r['family'][m], r['block'][m]), r['handle'][m]
    out = dict(n=int(m.sum()))
    out['b0'], out['se0'], out['r2_0'] = fit(cols, y, (feat,), cluster=handle)
    out['b1'], out['se1'], out['r2_1'] = fit(cols, y, (feat, tb), cluster=handle)
    return out


def fit_pages(r):
    own = r['own'] == 1
    R = np.column_stack([r['y'], r['share'], r['visible'], r['own'], r['log_handles']])
    D = demean(R, r['decision'])
    cols = [D[:, 1] * own, D[:, 1] * (~own), D[:, 2] * own, D[:, 3], D[:, 4]]
    tb = fe(r['family'], r['block'])
    out = dict(n=len(r['y']), n_decisions=r['n_decisions'])
    out['b0'], out['se0'], out['r2_0'] = fit(cols, D[:, 0])
    out['b1'], out['se1'], out['r2_1'] = fit(cols, D[:, 0], (tb,))
    return out


def name_windows(ds, feats):
    """The visibility test at three windows: task-mates born in the previous
    1.5, 3 or 6 h, split into page authors, feed and out of view; and handles
    of other tasks born in the window, in the feed against out of view."""
    birth = np.array([ds.birth[h].timestamp() for h in ds.handles])
    fam = np.array([ds.first_family[h] for h in ds.handles])
    out = {}
    for w in (1.5, 3, 6):
        groups = nm.exposure_groups(ds, mates_h=w)
        rec, rec2 = [], []
        for i, g in enumerate(groups):
            if not nm.in_sample(g):
                continue
            others = np.where((fam != fam[i]) & (birth < birth[i]) & (birth >= birth[i] - w * 3600))[0]
            o_feed = [j for j in others if j in g['feed']]
            o_out = [j for j in others if j not in g['feed'] and j not in g['authors']]
            for t, v in feats.items():
                rec.append((t, fam[i], g['block'], i, v[i], share(v, g['on_page'], 2), share(v, g['in_feed'], 2),
                            share(v, g['out_of_view'], 2)))
                rec2.append((t, fam[i], g['block'], i, v[i], share(v, o_feed, 3), share(v, o_out, 3)))
        res = {}
        for key, rows, k in (('same', rec, 3), ('other', rec2, 2)):
            c = list(zip(*rows))
            X = np.column_stack([np.array(x, float) for x in c[5:]])
            m = np.all(~np.isnan(X), axis=1)
            y = np.array(c[4], float)[m]
            groups_fe = (np.array(c[0])[m], fe(np.array(c[1])[m], np.array(c[2])[m]))
            b, se, _ = fit([X[m, j] for j in range(k)], y, groups_fe, cluster=np.array(c[3])[m])
            res[key] = dict(b=b, se=se, n=int(m.sum()))
        out[w] = res
    return out


def main(full=False):
    say = Report('Identification of copying' + (', all handles' if full else ''))
    ds = Dataset(full=full)
    forms = load_cache('forms_full.pkl' if full else 'forms.pkl')
    names_ = load_cache('names.pkl')
    generic = [p['name'] for p in names_['pieces'] if p['cls'] == 'generic']
    feats = {t: nm.indicators(ds, [t]) for t in generic}
    feats['date'] = np.array([bool(nm.DATE.search(h)) for h in ds.handles], float)

    pages = fit_pages(page_records(ds))
    groups = nm.exposure_groups(ds)
    name_fit = fit_names(nm.records(ds, feats, groups))
    form_fit = fit_forms(form_records(ds, forms['keep']))
    labels = dict(pages=['feed share, own-task page', 'feed share, other-task page', 'visible at all, own-task page',
                         'own-task page', 'log handles already there'],
                  names=['same task, authors of the page', 'same task, in the feed', 'same task, out of view',
                         'other tasks, in the feed'],
                  forms=['on the page', 'same task, in the feed', 'same task, out of view', 'other tasks, in the feed'])
    for key, res, unit in (('pages', pages, 'candidates'), ('names', name_fit, 'records'), ('forms', form_fit, 'first uses')):
        extra = f', {res["n_decisions"]} decisions' if key == 'pages' else ''
        say(f'{key}: {res["n"]} {unit}{extra}; R2 {res["r2_0"]:.3f} (baseline FE), {res["r2_1"]:.3f} (+ task x 3 h FE)')
        for j, lab in enumerate(labels[key]):
            say(f'  {lab:32} {res["b0"][j]:7.3f} +- {res["se0"][j]:.3f} (se)   + task x 3 h: {res["b1"][j]:7.3f} '
                f'+- {res["se1"][j]:.3f} (se), 95% half-width {1.96 * res["se1"][j]:.2f}')
    say(f'forms, fully interacted convention x task x 3 h FE (R2 {form_fit["r2_x"]:.2f}): '
        + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(form_fit['bx'], form_fit['sex'])))
    out = dict(pages=pages, names=name_fit, forms=form_fit, labels=labels, n_features=len(feats))

    if not full:
        r = page_records(ds)
        own_vis, own_out = (r['own'] == 1) & (r['visible'] == 1), (r['own'] == 1) & (r['visible'] == 0)
        say(f'own-task pages: chosen {r["y"][own_vis].mean():.3f} of the time when visible (n={own_vis.sum()}), '
            f'{r["y"][own_out].mean():.3f} when not (n={own_out.sum()}); ratio {r["y"][own_vis].mean() / r["y"][own_out].mean():.0f}')
        for a, b in ((0.01, 0.02), (0.02, 0.04), (0.04, 0.1), (0.1, 0.2)):
            mo = own_vis & (r['share'] >= a) & (r['share'] < b)
            mt = (r['own'] == 0) & (r['share'] >= a) & (r['share'] < b)
            say(f'  feed share {a:.2f}-{b:.2f}: own task {r["y"][mo].mean():.3f} at {r["share"][mo].mean():.3f} '
                f'(n={mo.sum()}), other task {r["y"][mt].mean():.3f} at {r["share"][mt].mean():.3f} (n={mt.sum()})')
        say(f'  slope ratio own/other task {pages["b1"][0] / pages["b1"][1]:.1f}')
        win = name_windows(ds, feats)
        for w, res in win.items():
            s_, o_ = res['same'], res['other']
            say(f'names, task-mates born in the previous {w:g} h (n={s_["n"]}): page {s_["b"][0]:.2f} +- {s_["se"][0]:.2f}, '
                f'feed {s_["b"][1]:.2f} +- {s_["se"][1]:.2f}, out of view {s_["b"][2]:.2f} +- {s_["se"][2]:.2f}; '
                f'other tasks (n={o_["n"]}): in the feed {o_["b"][0]:.2f} +- {o_["se"][0]:.2f}, '
                f'out of view {o_["b"][1]:.2f} +- {o_["se"][1]:.2f}')
        out['windows'] = win
    save_cache('identification_full.pkl' if full else 'identification.pkl', out)
    say.write()


if __name__ == '__main__':
    main(full='--full' in sys.argv)
