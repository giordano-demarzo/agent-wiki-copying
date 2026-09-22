"""SI, how to write (Section S5). Writes cache/si_forms.pkl.

1. For every kept convention: the first-use response records, the share of
   form A among the first uses of each block of 50 newborns (the heatmap), the
   late share of the prevailing form against the bias ceiling, the response
   measured against the page as it was six hours before the use, and the
   same-page agreement with the forms shuffled within three-hour blocks.
2. The slopes by class and by definition of a record (Table S4).
3. The family control: the page against the other pages of the same task, for
   forms and for names.
"""
import collections

import numpy as np

import names as nm
from common import Dataset, Report, T, block_of, fit, load_cache, save_cache, share, wilson
from conventions import CLASSES, CONV, FEED_USES, MIN_IN_VIEW, exposure, sequence

BLOCK = 50
SHIFT_H = 6
K_FAMILY = 30     # the family's other pages: the last 30 uses of the convention there, by other handles


def per_convention(ds, keep, fits):
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    ts = np.array([T(r['time']).timestamp() for r in ds.revs])
    rank = {h: i for i, h in enumerate(ds.handles)}
    n_blocks = int(np.ceil(len(ds.handles) / BLOCK))
    out = {}
    for name in keep:
        idx, form = sequence(ds.revs, name)
        took_a = (form == 0).astype(int)
        share_all, share_page = exposure(idx, form, label, page)
        first, seen = [], set()
        for k in range(len(idx)):
            if label[idx[k]] not in seen:
                first.append(k)
                seen.add(label[idx[k]])
        first = np.array(first)
        records = np.column_stack([share_page[first], took_a[first]])
        records = records[~np.isnan(records[:, 0])]
        blocks = [[] for _ in range(n_blocks)]
        for k in first:
            blocks[rank[label[idx[k]]] // BLOCK].append(took_a[k])
        heat = np.array([np.mean(b) if len(b) >= 3 else np.nan for b in blocks])
        fa = took_a[first]
        prevailing = int(fa.mean() >= 0.5)
        late = fa[-max(1, len(fa) // 5):].mean()
        mu = fits[name]
        s = mu[0] + mu[1]
        pi = mu[0] / s if s > 0 else 0.5
        # the page as it was six hours before the use
        history, shifted, first_set = collections.defaultdict(list), [], set(first)
        for k in range(len(idx)):
            pg, t = page[idx[k]], ts[idx[k]]
            earlier = [f for tt, f in history[pg] if tt < t - SHIFT_H * 3600]
            if k in first_set and len(earlier) >= MIN_IN_VIEW:
                shifted.append((1 - np.mean(earlier), took_a[k]))
            history[pg].append((t, form[k]))
        out[name] = dict(cls=CONV[name][0], records=records, heat=heat,
                         late=late if prevailing else 1 - late, s=s, ceiling=max(pi, 1 - pi),
                         shifted=np.array(shifted) if shifted else np.zeros((0, 2)),
                         all_records=np.column_stack([share_page, took_a]), n_first=len(fa))
    return out


def slope(records):
    v = records[~np.isnan(records[:, 0])]
    return np.polyfit(v[:, 0], v[:, 1], 1)[0] if len(v) >= 30 else np.nan


def form_family_records(ds, keep):
    """One record per use: share of A on the page, on the other pages of the
    same family (last K_FAMILY uses by other handles), and in the feed."""
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    family = [ds.family(r) for r in ds.revs]
    day = [r['time'][:10] for r in ds.revs]
    rows = []
    for name in keep:
        idx, form = sequence(ds.revs, name)
        history, seen = {}, set()
        tally = lambda js: np.array([np.sum(form[js] == 0), np.sum(form[js] == 1)], float)
        sh = lambda c: c[0] / c.sum() if c.sum() >= MIN_IN_VIEW else np.nan
        for k, i in enumerate(idx):
            h, pg, fam = label[i], page[i], family[i]
            others = [j for j in range(k - 1, -1, -1) if label[idx[j]] != h]
            fam_others = [j for j in others if family[idx[j]] == fam and page[idx[j]] != pg][:K_FAMILY]
            on_page = history.get(pg, np.zeros(2))
            rows.append((name, CONV[name][0], sh(on_page), sh(tally(fam_others)), sh(tally(others[:FEED_USES])),
                         int(form[k] == 0), h not in seen, fam, day[i]))
            seen.add(h)
            step = np.zeros(2)
            step[form[k]] = 1
            history[pg] = on_page + step
    c = list(zip(*rows))
    return dict(name=np.array(c[0]), cls=np.array(c[1]), page=np.array(c[2], float), family=np.array(c[3], float),
                feed=np.array(c[4], float), y=np.array(c[5], float), first=np.array(c[6], bool),
                fam=np.array(c[7]), day=np.array(c[8]))


def conflicts(a, b, y):
    """Among records where the two shares sit on opposite sides of one half,
    the probability that the outcome follows the first."""
    m = ~np.isnan(a) & ~np.isnan(b) & ((a > 0.5) != (b > 0.5)) & (a != 0.5) & (b != 0.5)
    follow = (y[m] == (a[m] > 0.5)).astype(int)
    return (*wilson(follow.sum(), len(follow)), int(len(follow)))


def main():
    say = Report('SI: how to write')
    ds = Dataset()
    forms = load_cache('forms.pkl')
    keep, fits = forms['keep'], forms['fits']
    conv = per_convention(ds, keep, fits)

    variants = {}
    for cls in CLASSES:
        names_ = [n for n in keep if CONV[n][0] == cls]
        first3 = np.vstack([conv[n]['records'] for n in names_])
        allr = np.vstack([conv[n]['all_records'] for n in names_])
        shifted = np.vstack([conv[n]['shifted'] for n in names_])
        variants[cls] = dict(first=slope(first3), all=slope(allr), shifted=slope(shifted), n_first=len(first3),
                             n_all=int((~np.isnan(allr[:, 0])).sum()), n_shifted=len(shifted))
        say(f'{cls:9}: slope on first uses {variants[cls]["first"]:.2f} (n={len(first3)}), all uses '
            f'{variants[cls]["all"]:.2f} (n={variants[cls]["n_all"]}), page {SHIFT_H} h earlier '
            f'{variants[cls]["shifted"]:.2f} (n={len(shifted)})')
    for n in keep:
        c = conv[n]
        say(f'  {n:26} s {c["s"]:.2f} bias ceiling {c["ceiling"]:.2f} late share of the prevailing form {c["late"]:.2f}')
    pw = {p['name']: p for p in forms['patchwork']}
    say('shuffle control: observed same-page agreement minus the time-shuffled one: '
        + ', '.join(f'{n} {pw[n]["same"] - pw[n]["shuffled"]:+.2f}' for n in keep))

    # the co-occurrence ratio over pairs of uses by different handles only
    from common import co_occurrence
    allp, diffp = [], []
    for n in keep:
        idx, form = sequence(ds.revs, n)
        pages = np.array([ds.revs[i]['page_id'] for i in idx])
        who = np.array([ds.revs[i]['label'] for i in idx])
        a = (form == 0).astype(float)
        if min(a.sum(), len(a) - a.sum()) < 10:
            continue
        if a.mean() > 0.5:
            a = 1 - a
        same = pages[:, None] == pages[None, :]
        np.fill_diagonal(same, False)
        other = who[:, None] != who[None, :]
        both = a[:, None] * a[None, :]
        allp.append(co_occurrence(pages, a))
        diffp.append(both[same & other].mean() / both[~same & other].mean())
    say(f'co-occurrence ratio, geometric mean over {len(allp)} conventions: all pairs {np.exp(np.mean(np.log(allp))):.2f}, '
        f'pairs of uses by different handles {np.exp(np.mean(np.log(diffp))):.2f}')

    r = form_family_records(ds, keep)
    fam = {}
    for cls in CLASSES + ('pooled',):
        m = r['first'] & ((r['cls'] == cls) if cls != 'pooled' else True)
        fam[cls] = conflicts(r['page'][m], r['family'][m], r['y'][m])
        say(f'page against the family\'s other pages, {cls:9}: follow the page {fam[cls][0]:.2f} '
            f'[{fam[cls][1]:.2f}, {fam[cls][2]:.2f}] (n={fam[cls][3]})')
    ff = conflicts(r['family'][r['first']], r['feed'][r['first']], r['y'][r['first']])
    say(f'family\'s other pages against the feed, pooled: follow the family {ff[0]:.2f} [{ff[1]:.2f}, {ff[2]:.2f}] (n={ff[3]})')
    m = r['first'] & ~np.isnan(r['page']) & ~np.isnan(r['family']) & ~np.isnan(r['feed'])
    b, se, _ = fit([r['page'][m], r['family'][m], r['feed'][m]], r['y'][m])
    say(f'joint fit on first uses with all three (n={m.sum()}): page {b[0]:.2f}+-{se[0]:.2f}, family '
        f'{b[1]:.2f}+-{se[1]:.2f}, feed {b[2]:.2f}+-{se[2]:.2f}')
    m2 = r['first'] & ~np.isnan(r['page']) & ~np.isnan(r['family'])
    b2, se2, _ = fit([r['page'][m2], r['family'][m2]], r['y'][m2])
    say(f'  page and family only (n={m2.sum()}): page {b2[0]:.2f}, family {b2[1]:.2f}')
    by_class = {}
    for cls in CLASSES:
        mc = m2 & (r['cls'] == cls)
        bc, sec, _ = fit([r['page'][mc], r['family'][mc]], r['y'][mc])
        by_class[cls] = (bc, sec, int(mc.sum()))
        say(f'  {cls:9}: page {bc[0]:.2f}, family {bc[1]:.2f} (n={mc.sum()})')
    m = r['first'] & ~np.isnan(r['page'])
    strata = np.array([f'{a}|{b_}|{c}' for a, b_, c in zip(r['name'][m], r['fam'][m], r['day'][m])])
    b_fe, se_fe, _ = fit([r['page'][m]], r['y'][m], (strata,))
    plain = np.polyfit(r['page'][m], r['y'][m], 1)[0]
    say(f'response to the page share: slope {plain:.2f}; within convention x family x day ({len(set(strata))} strata) '
        f'{b_fe[0]:.2f}+-{se_fe[0]:.2f} (n={m.sum()})')
    split = {}
    for lab, cond in (('family says A', r['family'] > 0.5), ('family says B', r['family'] < 0.5)):
        mm = r['first'] & ~np.isnan(r['page']) & ~np.isnan(r['family']) & cond
        split[lab] = np.column_stack([r['page'][mm], r['y'][mm]])

    # names: the page's authors against same-family handles active recently on other pages
    d = load_cache('names.pkl')
    feats, groups = d['features'], d['groups']
    famv = np.array([ds.first_family[h] for h in ds.handles])
    rows = []
    for i, g in enumerate(groups):
        if not nm.in_sample(g):
            continue
        fam_others = [j for j in g['mates'] if j not in g['authors']]
        for t, v in feats.items():
            rows.append((v[i], share(v, list(g['authors']), 3), share(v, fam_others, 3), share(v, list(g['feed']), 3), i))
    R = np.array(rows, float)
    m = np.all(~np.isnan(R[:, 1:4]), axis=1)
    bn, sen, _ = fit([R[m, 1], R[m, 2], R[m, 3]], R[m, 0], cluster=R[m, 4])
    cn = conflicts(R[:, 1], R[:, 2], R[:, 0])
    say(f'names, joint fit (n={m.sum()}): page\'s authors {bn[0]:.2f}+-{sen[0]:.2f}, same-task handles on other pages '
        f'{bn[1]:.2f}+-{sen[1]:.2f}, feed {bn[2]:.2f}+-{sen[2]:.2f}; conflicts: follow the page {cn[0]:.2f} '
        f'[{cn[1]:.2f}, {cn[2]:.2f}] (n={cn[3]})')

    save_cache('si_forms.pkl', dict(conv=conv, variants=variants, keep=keep, fits=fits,
                                    family=dict(conflict=fam, conflict_feed=ff, joint=(b, se, int(m.sum())),
                                                by_class=by_class, split=split, fe=(plain, b_fe[0], se_fe[0])),
                                    names=dict(joint=(bn, sen), conflict=cn),
                                    block_days=[ds.birth[ds.handles[min(i * BLOCK, len(ds.handles) - 1)]].strftime('%d %b')
                                                for i in range(int(np.ceil(len(ds.handles) / BLOCK)))]))
    say.write()


if __name__ == '__main__':
    main()
