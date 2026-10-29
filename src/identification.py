"""Identification of copying: linear probability models in which the exposure
is split by task (own or other) and by visibility (on the page or in the last
100 feed lines, against older). Names and forms are estimated with intercepts
per feature or convention and again with task-family by three-hour-block
effects added; pages within the arrival, whose intercept nests both. Standard
errors are clustered by handle throughout. Writes cache/identification.pkl (Table 1, Tables S5-S7, Fig. S15) and,
with --full, cache/identification_full.pkl (Table S1).
"""
import collections
import sys

import numpy as np

import common
import names as nm
from common import (FEED_LINES, Dataset, Report, T, block_of, conditional_logit, demean, fit, fit_full, load_cache,
                    save_cache, share, wald)
from conventions import sequence

MATES_H = nm.MATES_H
WINDOWS = (1.5, 3, 6, 12)   # hours: the task-mate windows of the visibility test


def form_records(ds, keep, kmin=nm.MIN_GROUP, task='handle'):
    """One record per handle's first use of a convention: the outcome (form A)
    and the share of A on the page, among same-task uses in the feed, among
    same-task uses of the previous six hours that had scrolled off, and among
    other tasks' uses in the feed. Each share needs at least `kmin` uses.
    The task of a use is the family of the handle's first task edit, or with
    task='page' the family of the page it is written on (for task pages)."""
    ts_all = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    pos = {id(r): k for k, r in enumerate(ds.revs_all)}
    rows = []
    for name in keep:
        idx, form = sequence(ds.revs, name)
        use_pos = np.array([pos[id(ds.revs[i])] for i in idx])
        use_lab = [ds.revs[i]['label'] for i in idx]
        use_pg = [ds.revs[i]['page_id'] for i in idx]
        if task == 'page':
            use_fam = np.array([ds.family(ds.revs[i]) if ds.is_task(ds.revs[i]) else ds.first_family[ds.revs[i]['label']]
                                for i in idx])
        else:
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
            rows.append((name, use_fam[k], block_of(t), h, a[k], share(a, on_page, kmin), share(a, same_feed, kmin),
                         share(a, same_out, kmin), share(a, other_feed, kmin)))
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
    created_at, handles_on, rows, n_dec, outside = {}, collections.defaultdict(set), [], 0, []
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
                # a page of another task outside the feed is not a candidate: such a decision has no
                # chosen alternative in its choice set and is left out rather than given one
                if pg not in cands:
                    outside.append(fam_page.get(pg) != own)
                    handles_on[pg].add(h)
                    continue
                for p in cands:
                    rows.append((n_dec, own, block_of(ts_all[k]), int(p == pg), window.get(p, 0) / FEED_LINES,
                                 int(p in window), int(fam_page.get(p) == own), np.log1p(len(handles_on[p])), h))
                n_dec += 1
        handles_on[pg].add(h)
    c = list(zip(*rows))
    return dict(decision=np.array(c[0]), family=np.array(c[1]), block=np.array(c[2]), y=np.array(c[3], float),
                share=np.array(c[4]), visible=np.array(c[5]), own=np.array(c[6]), log_handles=np.array(c[7]),
                handle=np.array(c[8]), n_decisions=n_dec, n_outside=len(outside),
                n_outside_other_task=int(sum(outside)))


def fe(*arrays):
    """One grouping variable out of several: their combination."""
    return np.array(['|'.join(map(str, t)) for t in zip(*arrays)])


CONTRASTS = {'page vs out of view': [1, 0, -1, 0], 'feed vs out of view': [0, 1, -1, 0],
             'own task vs other tasks, in the feed': [0, 1, 0, -1]}


def checks(X, y, groups, handle, tb):
    """On the task x 3 h specification: the covariance for the Wald tests of
    the contrasts that the text relies on, and the two-way clustered standard
    errors (handle and task x 3 h)."""
    b, cov, _ = fit_full(X, y, groups, cluster=handle)
    pad = lambda c: list(c) + [0] * (len(b) - len(c))
    out = {name: wald(b, cov, pad(c)) for name, c in CONTRASTS.items()}
    _, cov2, _ = fit_full(X, y, groups, cluster=(handle, tb))
    return dict(wald=out, se_twoway=np.sqrt(np.diag(cov2)))


def fit_forms(r):
    m = np.all(~np.isnan(r['X']), axis=1)
    X, y = [r['X'][m, j] for j in range(4)], r['y'][m]
    conv, tb = r['convention'][m], fe(r['family'][m], r['block'][m])
    out = dict(n=int(m.sum()))
    handle = r['handle'][m]
    out['b0'], out['se0'], out['r2_0'] = fit(X, y, (conv,), cluster=handle)
    out['b1'], out['se1'], out['r2_1'] = fit(X, y, (conv, tb), cluster=handle)
    out['bx'], out['sex'], out['r2_x'] = fit(X, y, (fe(conv, tb),), cluster=handle)
    _, out['cov1'], _ = fit_full(X, y, (conv, tb), cluster=handle)
    _, out['covx'], _ = fit_full(X, y, (fe(conv, tb),), cluster=handle)
    out.update(checks(X, y, (conv, tb), handle, tb))
    return out


def fit_names(r):
    """A newcomer whose first page had fewer than two earlier task-mates keeps
    its record with the page share at zero and an indicator (fifth column)."""
    X = np.column_stack([r['page0'], r['feed'], r['out'], r['other'], r['no_page']])
    m = np.all(~np.isnan(X), axis=1)
    cols, y = [X[m, j] for j in range(5)], r['y'][m]
    feat, tb, handle = r['feature'][m], fe(r['family'][m], r['block'][m]), r['handle'][m]
    out = dict(n=int(m.sum()))
    out['b0'], out['se0'], out['r2_0'] = fit(cols, y, (feat,), cluster=handle)
    out['b1'], out['se1'], out['r2_1'] = fit(cols, y, (feat, tb), cluster=handle)
    # feature x task intercepts added: absorbs a prompt that favours one feature in one task
    out['b2'], out['se2'], out['r2_2'] = fit(cols, y, (fe(feat, r['family'][m]), tb), cluster=handle)
    b2, cov2, _ = fit_full(cols, y, (fe(feat, r['family'][m]), tb), cluster=handle)
    _, out['cov1'], _ = fit_full(cols, y, (feat, tb), cluster=handle)
    out['cov2'] = cov2
    out['wald2'] = {name: wald(b2, cov2, list(c) + [0] * (len(b2) - len(c))) for name, c in CONTRASTS.items()}
    # feature x block intercepts as well: absorbs a drift of the feature across the whole wiki
    out['b3'], out['se3'], out['r2_3'] = fit(cols, y, (fe(feat, r['family'][m]), fe(feat, r['block'][m]), tb), cluster=handle)
    out.update(checks(cols, y, (feat, tb), handle, tb))
    return out


def fit_pages(r):
    """Within-arrival estimator of Eq. (regpages): the regressors, interactions
    included, and the outcome are demeaned within the arrival, which is the
    same as an intercept per arrival. That intercept nests the task family and
    the three-hour block of the arrival, so there is one specification; it is
    stored under both keys for the tables."""
    own = r['own'] == 1
    R = np.column_stack([r['y'], r['share'] * own, r['share'] * (~own), r['visible'] * own, r['own'],
                         r['log_handles']])
    D = demean(R, r['decision'])
    cols = [D[:, j] for j in range(1, 6)]
    out = dict(n=len(r['y']), n_decisions=r['n_decisions'], nested=True)
    out['b0'], out['se0'], out['r2_0'] = fit(cols, D[:, 0], cluster=r['handle'])
    out['b1'], out['se1'], out['r2_1'] = out['b0'], out['se0'], out['r2_0']
    b, cov, _ = fit_full(cols, D[:, 0], cluster=r['handle'])
    out['wald'] = {'own task vs other task, share of the feed': wald(b, cov, [1, -1, 0, 0, 0])}
    _, cov2, _ = fit_full(cols, D[:, 0], cluster=(r['handle'], fe(r['family'], r['block'])))
    out['se_twoway'] = np.sqrt(np.diag(cov2))
    # the textbook model for a choice among alternatives: conditional logit within the arrival
    bl, covl, r2l = conditional_logit(R[:, 1:], r['y'], r['decision'], cluster=r['handle'])
    out['clogit'] = dict(b=bl, se=np.sqrt(np.diag(covl)), pseudo_r2=r2l)
    return out


def name_windows(ds, feats):
    """The visibility test at four windows: task-mates born in the previous
    1.5, 3, 6 or 12 h, split into page authors, feed and out of view; and handles
    of other tasks born in the window, in the feed against out of view."""
    birth = np.array([ds.birth[h].timestamp() for h in ds.handles])
    fam = np.array([ds.first_family[h] for h in ds.handles])
    out = {}
    for w in WINDOWS:
        groups = nm.exposure_groups(ds, mates_h=w)
        rec, rec2 = [], []
        for i, g in enumerate(groups):
            if not nm.in_sample(g):
                continue
            others = np.where((fam != fam[i]) & (birth < birth[i]) & (birth >= birth[i] - w * 3600))[0]
            o_feed = [j for j in others if j in g['feed']]
            o_out = [j for j in others if j not in g['feed'] and j not in g['authors']]
            for t, v in feats.items():
                k = nm.MIN_GROUP
                on_page = share(v, g['on_page'], k)
                rec.append((t, fam[i], g['block'], i, v[i], np.nan_to_num(on_page), share(v, g['in_feed'], k),
                            share(v, g['out_of_view'], k), float(np.isnan(on_page))))
                rec2.append((t, fam[i], g['block'], i, v[i], share(v, o_feed, k), share(v, o_out, k)))
        res = {}
        for key, rows, k in (('same', rec, 4), ('other', rec2, 2)):
            c = list(zip(*rows))
            X = np.column_stack([np.array(x, float) for x in c[5:]])
            m = np.all(~np.isnan(X), axis=1)
            y = np.array(c[4], float)[m]
            groups_fe = (np.array(c[0])[m], fe(np.array(c[1])[m], np.array(c[2])[m]))
            b, cov, _ = fit_full([X[m, j] for j in range(k)], y, groups_fe, cluster=np.array(c[3])[m])
            res[key] = dict(b=b, se=np.sqrt(np.diag(cov)), cov=cov, n=int(m.sum()))
        out[w] = res
    return out


def sensitivity(ds, feats, keep):
    """Table 1 (task x block specification) for names and forms under other
    analysis choices: the minimum group for a share (1 or 3 instead of 2), the
    length of the blocks (1, 6 or 12 h instead of 3), and, for forms, the task
    of a use taken from its page rather than from the handle."""
    def both(kmin=nm.MIN_GROUP, task='handle'):
        n = fit_names(nm.records(ds, feats, nm.exposure_groups(ds), kmin=kmin))
        f = fit_forms(form_records(ds, keep, kmin=kmin, task=task))
        return dict(names=dict(b=n['b1'][:4], se=n['se1'][:4], n=n['n']),
                    forms=dict(b=f['b1'], se=f['se1'], n=f['n']))
    out = {'main': both()}
    for k in (1, 3):
        out[f'at least {k} instance' + ('s' if k > 1 else '')] = both(kmin=k)
    main_h = common.BLOCK_H
    try:
        for h in (1, 6, 12):
            common.BLOCK_H = h      # block_of reads it: the blocks of every record
            out[f'blocks of {h} h'] = both()
    finally:
        common.BLOCK_H = main_h
    out['task of a use from its page'] = both(task='page')
    out['task of a use from its page']['names'] = None   # a name is chosen at the first task edit: no alternative
    return out


def main(full=False):
    say = Report('Identification of copying' + (', all handles' if full else ''))
    ds = Dataset(full=full)
    forms = load_cache('forms_full.pkl' if full else 'forms.pkl')
    names_ = load_cache('names.pkl')
    generic = [p['name'] for p in names_['pieces'] if p['cls'] == 'generic']
    feats = {t: nm.indicators(ds, [t]) for t in generic}
    feats['date'] = np.array([bool(nm.DATE.search(h)) for h in ds.handles], float)

    page_rec = page_records(ds)
    pages = fit_pages(page_rec)
    pages.update(n_outside=page_rec['n_outside'], n_outside_other_task=page_rec['n_outside_other_task'])
    say(f'pages: {page_rec["n_outside"]} appends left out because the chosen page is not a candidate '
        f'({page_rec["n_outside_other_task"]} of them a page of another task outside the feed)')
    groups = nm.exposure_groups(ds)
    name_fit = fit_names(nm.records(ds, feats, groups))
    form_fit = fit_forms(form_records(ds, forms['keep']))
    labels = dict(pages=['share of the feed, own task', 'share of the feed, other task', 'in the feed, own task',
                         'own task', 'log(1 + handles on the page)'],
                  names=['share among page authors, own task', 'share in the feed, own task',
                         'share out of view, own task', 'share in the feed, other tasks',
                         'first page without earlier task-mates'],
                  forms=['share on the page', 'share in the feed, own task', 'share out of view, own task',
                         'share in the feed, other tasks'])
    for key, res, unit in (('pages', pages, 'candidates'), ('names', name_fit, 'records'), ('forms', form_fit, 'first uses')):
        extra = f', {res["n_decisions"]} decisions' if key == 'pages' else ''
        say(f'{key}: {res["n"]} {unit}{extra}; R2 {res["r2_0"]:.3f} (baseline intercepts), {res["r2_1"]:.3f} (+ task x 3 h)'
            + (' [the arrival intercept nests task x 3 h: one specification]' if res.get('nested') else ''))
        for j, lab in enumerate(labels[key]):
            say(f'  {lab:32} {res["b0"][j]:7.3f} +- {res["se0"][j]:.3f} (se)   + task x 3 h: {res["b1"][j]:7.3f} '
                f'+- {res["se1"][j]:.3f} (se), 95% half-width {1.96 * res["se1"][j]:.2f}')
    say(f'names, feature x task + task x 3 h intercepts (R2 {name_fit["r2_2"]:.2f}): '
        + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(name_fit['b2'], name_fit['se2'])))
    say(f'names, feature x task + feature x 3 h + task x 3 h intercepts (R2 {name_fit["r2_3"]:.2f}): '
        + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(name_fit['b3'], name_fit['se3'])))
    for name, (stat, p) in name_fit['wald2'].items():
        say(f'names with feature x task intercepts, Wald {name}: chi2 {stat:.1f}, p {p:.2g}')
    for key, res in (('pages', pages), ('names', name_fit), ('forms', form_fit)):
        for name, (stat, p) in res['wald'].items():
            say(f'{key}, Wald {name}: chi2 {stat:.1f}, p {p:.2g}')
        say(f'{key}, two-way clustered se (handle, task x 3 h): ' + ', '.join(f'{x:.3f}' for x in res['se_twoway']))
    say('pages, conditional logit (clustered by handle), pseudo-R2 %.2f: ' % pages['clogit']['pseudo_r2']
        + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(pages['clogit']['b'], pages['clogit']['se'])))
    say(f'forms, fully interacted convention x task x 3 h FE (R2 {form_fit["r2_x"]:.2f}): '
        + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(form_fit['bx'], form_fit['sex'])))
    out = dict(pages=pages, names=name_fit, forms=form_fit, labels=labels, n_features=len(feats))

    if not full:
        r = page_records(ds)
        own_vis, own_out = (r['own'] == 1) & (r['visible'] == 1), (r['own'] == 1) & (r['visible'] == 0)
        out['own_visible'] = (r['y'][own_vis].mean(), int(own_vis.sum()), r['y'][own_out].mean(), int(own_out.sum()))
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
        sens = sensitivity(ds, feats, forms['keep'])
        for lab, r in sens.items():
            say(f'sensitivity, {lab}: ' + '; '.join(
                f'{key} (n={r[key]["n"]}) ' + ', '.join(f'{b:.2f} +- {1.96 * s:.2f}' for b, s in zip(r[key]['b'], r[key]['se']))
                for key in ('names', 'forms') if r[key] is not None))
        out['sensitivity'] = sens
        # what a newcomer of the name analysis saw, and the regression with the other classes of pieces
        sample = [g for g in groups if nm.in_sample(g)]
        own = np.array([len(g['in_feed_any']) for g in sample])
        out['seen'] = dict(own=own.mean(), other=np.mean([len(g['other_tasks']) for g in sample]), none_own=np.mean(own == 0))
        say(f'newcomers of the name analysis saw on average {own.mean():.1f} task-mates and '
            f'{out["seen"]["other"]:.1f} handles of other tasks in the feed; {out["seen"]["none_own"]:.2f} saw no task-mate')
        cls = {p['name']: p['cls'] for p in names_['pieces']}
        out['pieces'] = {}
        for lab, keep in (('all', [p for p, c in cls.items() if c in ('generic', 'task-specific')]),
                          ('task-specific', [p for p, c in cls.items() if c == 'task-specific'])):
            fs = {t: nm.indicators(ds, [t]) for t in keep}
            if lab == 'all':
                fs['date'] = feats['date']
            r = fit_names(nm.records(ds, fs, groups))
            out['pieces'][lab] = dict(b=r['b1'][:4], n=r['n'])
            say(f'names with the {lab} pieces ({len(fs)} features, n={r["n"]}): ' + ', '.join(f'{b:.2f}' for b in r['b1'][:4]))
    save_cache('identification_full.pkl' if full else 'identification.pkl', out)
    say.write()


if __name__ == '__main__':
    main(full='--full' in sys.argv)
