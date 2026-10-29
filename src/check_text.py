"""Check that the numbers quoted in paper/main.tex and paper/si.tex are the
ones the pipeline produces.

Every check names a file, a regular expression whose groups capture the
numbers as the text states them, and the values recomputed from the cache.
A captured number agrees with its value when it rounds to it at the precision
the text uses (percentages and number words are handled). Table 1 and the SI
tables are generated, so they are not checked here.

    python src/check_text.py        exits with 1 if any check fails
"""
import os
import re
import sys

import numpy as np

import model as md
from common import Dataset, ROOT, load_cache

WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
         'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'eighteen': 18, 'twenty': 20, 'fifty': 50, 'a quarter': 0.25, 'a third': 1 / 3, 'half': 0.5,
         'two thirds': 2 / 3, 'a fifth': 0.2, 'one in seven': 1 / 7, 'a few dozen': None}


def number(text):
    """A captured token as a number: digits with commas, a percentage, or a number word."""
    t = text.replace('{,}', '').replace(',', '').replace('\\%', '').replace('$', '').strip().lower()
    if t in WORDS:
        return WORDS[t], 0
    m = re.fullmatch(r'-?\d+(?:\.(\d+))?', t)
    if not m:
        raise ValueError(text)
    return float(t), len(m.group(1) or '')


def agree(captured, value):
    """The text's number rounds to the pipeline's value at the text's precision."""
    x, decimals = number(captured)
    if x is None:
        return True
    if isinstance(value, tuple) and value[0] == 'rel':
        return abs(x - value[1]) <= value[2] * abs(value[1])
    if isinstance(value, str):      # a percentage: the value is a share, the text a percentage
        value = float(value) * 100
    return abs(x - value) <= 0.5 * 10 ** (-decimals) + 1e-9


def expected_values():
    ds = Dataset()
    pages, names, forms = load_cache('pages.pkl'), load_cache('names.pkl'), load_cache('forms.pkl')
    ident, ident_full, forms_full = load_cache('identification.pkl'), load_cache('identification_full.pkl'), load_cache('forms_full.pkl')
    mod, pop, sin, sif = load_cache('model.pkl'), load_cache('si_population.pkl'), load_cache('si_names.pkl'), load_cache('si_forms.pkl')
    inp, sw = pages['inputs'], mod['main']
    hpp = inp['handles_per_page']
    v = {}
    # the record and the population
    v['revs_all'] = len(ds.revs_all); v['handles_all'] = len(ds.birth); v['handles'] = len(ds.handles)
    v['edits'] = len(ds.revs); v['task_edits'] = inp['n_edits']; v['task_pages'] = inp['n_pages']
    v['families'] = len({ds.family(r) for r in ds.revs if ds.is_task(r)})
    f2 = load_cache('fig2.pkl'); v['top_late'], v['even_late'] = f2['top_late'], f2['even_late']
    import json, collections as _c
    net = json.load(open(os.path.join(ROOT, 'cache', 'network.json'))); v['giant'] = len(net['nodes'])
    editors = [sorted({r['label'] for r in rs}) for rs in ds.by_page(task_only=True).values()]
    linked = {h for hs in editors if len(hs) > 1 for h in hs}; v['isolated'] = len(ds.handles) - len(linked)
    import network as _net
    parts = _net.components(sorted(linked), sorted({p for hs in editors for p in __import__('itertools').combinations(hs, 2)}))
    v['small'] = len(linked) - len(parts[0]); v['n_small'] = len(parts) - 1
    v['excluded'] = len(ds.swarm); v['excluded_18'] = sum(1 for h in ds.swarm if (ds.birth[h].month, ds.birth[h].day) == (6, 18))  # not strftime: locale-dependent
    nfam = [len({ds.family(r) for r in ds.by_label[h] if ds.is_task(r)}) for h in ds.handles]
    v['one_family'], v['two_families'], v['three_families'] = nfam.count(1), nfam.count(2), sum(1 for k in nfam if k >= 3)
    v['span_h'] = np.median(pop['handle_span']) / 60; v['n_span'] = len(pop['handle_span'])
    v['span_6h'] = str(np.mean(pop['handle_span'] > 360)); v['span_1d'] = str(np.mean(pop['handle_span'] > 1440))
    v['edits_median'] = np.median(pop['edits_per']); v['edits_one'] = str(np.mean(pop['edits_per'] == 1)); v['edits_max'] = pop['edits_per'].max()
    v['page_span_h'] = np.median(pop['page_span']) / 60; v['readable_d'] = np.median(pop['readable']) / 1440
    v['pages_touched'] = len(pop['readable']); v['pages_deleted'] = pop['n_deleted']
    v['readable_1d'] = str(np.mean(pop['readable'] > 1440)); v['readable_1w'] = str(np.mean(pop['readable'] > 10080))
    fam = sorted(pop['families'], key=lambda r: -r[1]); v['largest_family'] = fam[0][1]
    top7 = {r[0] for r in fam[:7]}
    v['top7_handles'] = len({h for h in ds.handles if ds.first_family[h] in top7 or any(ds.family(r) in top7 for r in ds.by_label[h] if ds.is_task(r))})
    # where to write
    c = pages['counts']; v['creations'], v['appends'] = c['n_create'], c['n_append']; v['decisions'] = c['n_create'] + c['n_append']
    v['in_window'] = str(c['n_in_window'] / c['n_append']); v['c'] = inp['c']; v['pages_per'] = inp['pages_per'].mean()
    v['p_tail'] = [np.mean(hpp >= k) for k in (5, 10, 20, 40)]; v['largest'] = hpp.max()
    for w, d in pages['windows'].items():
        v[f'inside_{w}'] = str(d['inside']); v[f'slope_{w}'] = d['slope']
    cr = {k: np.array([d[k] for d in pages['creation']]) for k in ('created', 'first', 'visible', 'own')}
    v['c_first'], v['c_later'] = cr['created'][cr['first'] == 1].mean(), cr['created'][cr['first'] == 0].mean()
    v['by_active'] = [b[3] for b in pages['by_active']]; v['by_active_n0'] = pages['by_active'][0][2]
    v['by_day_min'], v['by_day_max'] = min(pages['by_day'].values()), max(pages['by_day'].values())
    vis = cr['visible'] == 1; v['n_vis'], v['n_novis'] = int(vis.sum()), int((~vis).sum())
    v['c_vis'] = str(cr['created'][vis].mean()); v['c_novis'] = str(cr['created'][~vis].mean())
    later = cr['first'] == 0; v['n_later'] = int(later.sum()); v['own_later'] = str(cr['own'][later].mean())
    v['n_attach'] = pages['n_attach']; k = pages['kernel']; v['kernel'] = list(k[:5, 1] / pages['ref'])
    # how to sign
    pieces = names['pieces']; v['n_pieces'] = len(pieces)
    v['n_months'] = sum(p['cls'] == 'month' for p in pieces); v['n_specific'] = sum(p['cls'] == 'task-specific' for p in pieces)
    v['n_generic'] = sum(p['cls'] == 'generic' for p in pieces); v['n_features'] = len(names['features'])
    over = sorted(p['over'] for p in pieces if p['cls'] != 'month'); v['gap_below'] = max(o for o in over if o < 4); v['gap_above'] = min(o for o in over if o >= 4)
    fs = [names['stats'][t]['s'] for t in names['features']]; v['s_features'] = (min(fs), max(fs))
    v['any_feature'] = int((sum(names['features'].values()) > 0).sum())
    # how to write
    rows = forms['rows']; v['candidates'] = len(rows); v['kept'] = len(forms['keep']); v['dropped'] = len(rows) - len(forms['keep'])
    v['under_100'] = sum(r['uses'] < 100 for r in rows.values()); v['never'] = sum(r['uses'] >= 100 and r['minority'] == 0 for r in rows.values())
    width = lambda ns: np.median([rows[n]['s_interval'][2] - rows[n]['s_interval'][1] for n in ns])
    v['width_kept'] = width(forms['keep']); v['width_dropped'] = width([n for n in rows if not rows[n]['kept'] and rows[n]['s_interval']])
    ks = forms['keep']; v['uses_range'] = (min(rows[n]['uses'] for n in ks), max(rows[n]['uses'] for n in ks))
    v['pages_range'] = (min(rows[n]['n_pages'] for n in ks), max(rows[n]['n_pages'] for n in ks)); v['s_range'] = (min(sum(rows[n]['mu']) for n in ks), max(sum(rows[n]['mu']) for n in ks))
    v['mu_Rn'] = rows['Rn / #n']['mu'][0]; v['mu_CONFIRMED'] = rows['CONFIRMED / confirmed']['mu'][0]
    pw = forms['patchwork']; v['same'], v['diff'] = np.mean([p['same'] for p in pw]), np.mean([p['diff'] for p in pw])
    v['usage'] = np.mean(list(forms['usage'].values())); v['usage_range'] = (min(forms['usage'].values()), max(forms['usage'].values()))
    v['kept_full'] = len(forms_full['keep'])
    # the answers in the record
    _d = load_cache('data.pkl'); v['release'] = (len(_d['revs']), len(_d['pages']))
    v['n_first_curve'] = sum(len(x) for x in forms['by_class'].values())
    ans = load_cache('answers_record.pkl'); v['ans_uses'], v['ans_pages_all'], v['ans_handles_all'] = ans['n_uses'], ans['n_pages'], ans['n_handles']
    v['ans_next'] = ans['uses_before'] - 1; v['ans_handles'] = ans['handles_before']; v['ans_pages'] = ans['pages_before']
    # identification
    _idf = load_cache('identification.pkl')['forms']; v['forms_n'] = _idf['n']; v['forms_r2'] = [_idf['r2_0'], _idf['r2_1'], _idf['r2_x']]
    p, n, f = ident['pages'], ident['names'], ident['forms']
    v['n_records'], v['n_first'], v['n_cand'], v['n_dec'] = n['n'], f['n'], p['n'], p['n_decisions']
    v['own_vis_ratio'] = p['b1'][0] / p['b1'][1]; v['r2'] = (f['r2_1'], n['r2_1'], p['r2_1']); v['fx'] = list(f['bx'])
    v['out_full'], v['out_pop'] = ident_full['names']['b1'][2], n['b1'][2]; v['full_names'] = list(ident_full['names']['b1'])
    th = load_cache('si_thresholds.pkl'); v['th_page'] = (min(r['b'][0] for r in th), max(r['b'][0] for r in th)); v['th_kept'] = (min(r['kept'] for r in th), max(r['kept'] for r in th))
    v['th_slope'] = (min(r['slope'] for r in th), max(r['slope'] for r in th))
    pr = load_cache('si_prior.pkl')['conventions']
    v['habit_more'] = sum(c['repeat'] > c['predicted'] for c in pr.values()); v['n_conv'] = len(pr)
    v['first_larger'] = sum(c['s_first'] > c['s_all'] for c in pr.values())
    v['sec_first'], v['sec_later'] = 100 * pr['seconds / sec']['f_first'], 100 * pr['seconds / sec']['f_later']
    v['neff'] = (min(c['n_eff'] for c in pr.values()), max(c['n_eff'] for c in pr.values()), 100 * np.median([c['n_eff'] / c['uses'] for c in pr.values()]))
    import fig4_model
    xs, ys, _, _ = fig4_model.binned([dict(kind=c['cls'], s_floor=c['s_first_floor'], devn=c['devn_first']) for c in pr.values()],
                                     fig4_model.FORMS, 'devn', True, True, fig4_model.EDGES_LOG, 's_floor')
    mid = [k for k in range(len(xs)) if 0.2 < xs[k] < 0.5]
    v['first_bins'] = [ys[mid[0]], xs[mid[0]], ys[mid[1]], xs[mid[1]], 1 / xs[mid[0]], 1 / xs[mid[1]]]
    rc = load_cache('si_recovery.pkl')['rows']
    s0 = min(r['s'] for r in rc)
    v['rec_floored'] = (sum(r['floored'] for r in rc if r['s'] == s0), sum(1 for r in rc if r['s'] == s0), s0)
    v['rec_none_from'] = min(s_ for s_ in {r['s'] for r in rc} if not any(r['floored'] for r in rc if r['s'] >= s_))
    s3 = sorted({r['s'] for r in rc})[2]
    v['rec_factor'] = (np.median([r['s_hat'] / s3 for r in rc if r['s'] == s3 and r['kept'] and not r['floored']]), s3)
    rp = [dict(kind='coined', s_floor=r['s_hat'], devn=r['devn_hat']) for r in rc if r['kept'] and not r['floored'] and np.isfinite(r['devn_hat'])]
    xs, ys, _, _ = fig4_model.binned(rp, {'coined'}, 'devn', True, True, fig4_model.EDGES_LOG, 's_floor')
    ratio = ys * xs
    v['rec_ratio'] = (ratio.min(), ratio.max(), ratio[-1], 1 / ratio.min() if ratio.min() < 1 else ratio.max(), 100 * (1 - ratio[-1]))
    ue = [r['devn_hat'] for r in rc if r['kept'] and not r['floored'] and 0.56 <= r['s_hat'] < 1.0]
    v['rec_below'] = 100 * np.mean(np.array(ue) < 0.03)
    jn = names['joint']; v['joint'] = (jn['b'][0], jn['se'][0], jn['b'][1], jn['se'][1], jn['n'])
    v['n1'] = list(n['b1']); v['n2'], v['n3'] = list(n['b2']), list(n['b3']); v['c_page'] = (n['b2'][0] - n['b2'][2], n['b1'][0] - n['b1'][2], n['b2'][1] - n['b2'][2], n['b1'][1] - n['b1'][2])
    v['c3'] = (n['b3'][0] - n['b3'][2], n['b3'][1] - n['b3'][2]); cl = p['clogit']['b']; v['logodds'] = cl[2] + cl[3]; v['clogit'] = list(cl)
    for w, r in ident['windows'].items():
        v[f'win_{w}'] = list(r['same']['b']); v[f'win_other_{w}'] = list(r['other']['b'])
    v['seen'], v['pieces'], v['sens'] = ident['seen'], ident['pieces'], ident['sensitivity']
    v['names_r2'] = [n['r2_0'], n['r2_1'], n['r2_2'], n['r2_3']]; v['n_names'] = n['n']
    v['view_fit'] = None   # the joint fit of Fig. 3b is recomputed in fig3_rule; checked through si_names below
    # the model
    par = mod['par']; v['agents'] = par['n_agents']; v['days'] = (par['window'][1] - par['window'][0]) / 86400; v['edits_model'] = par['edits']
    pg = mod['pages']; v['largest_model'] = np.median(pg['max']); v['largest_5_95'] = np.percentile(pg['max'], [5, 95]); v['p20_model'] = np.median(pg['p20'])
    v['model_pages'] = np.mean(pg['n_pages'])
    for m, w in mod['windows'].items():
        v[f'mw_{m}'] = [np.median(w[k]) for k in ('p5', 'p10', 'p20', 'p40')]; v[f'mw_max_{m}'] = np.median(w['max'])
    for cch, w in mod['nulls'].items():
        v[f'null_{cch}'] = [np.median(w[k]) for k in ('p10', 'p20', 'p40')]; v[f'null_max_{cch}'] = np.median(w['max'])
    for s in (0.02, 0.1, 0.3):
        fin = sw[('form', s, 0.5)]['final']; v[f'final_{s}'] = (fin.min(), fin.max())
    lo6, hi6 = sw[('form', 0.6, 0.5)]['final'], sw[('form', 1.0, 0.5)]['final']; v['final_06_1'] = (min(lo6.min(), hi6.min()), max(lo6.max(), hi6.max()))
    v['sd2'] = {s: 2 * np.std(sw[('form', s, 0.5)]['final']) for s in md.S_GRID}
    at = lambda w, kind, stat, s, p=None: np.nanmedian(w[(kind, s, p)][stat]) if p else np.nanmedian(np.concatenate([w[(kind, s, q)][stat] for q in md.DIRECTIONS]))
    v['dir_vol_02'] = [at(sw, 'form', 'vol', 0.2, q) for q in md.DIRECTIONS]; v['dir_vol_06'] = [at(sw, 'form', 'vol', 0.6, q) for q in md.DIRECTIONS]
    v['dir_gap_02'] = [at(sw, 'form', 'gap', 0.2, q) for q in md.DIRECTIONS]; v['dir_gap_04'] = [at(sw, 'form', 'gap', 0.4, q) for q in md.DIRECTIONS]
    spread = lambda kind, stat: (np.nanmean([np.nanstd([at(sw, kind, stat, s, q) for q in md.DIRECTIONS]) for s in md.S_GRID if s > 0]),
                                 np.nanmean([np.nanstd(sw[(kind, s, 0.5)][stat]) for s in md.S_GRID if s > 0]))
    v['spread_form_vol'], v['spread_name_vol'] = spread('form', 'vol'), spread('name', 'vol')
    v['spread_form_gap'], v['spread_name_gap'] = spread('form', 'gap'), spread('name', 'gap')
    u5, u1 = mod['usage'][0.5], mod['usage'][0.1]
    v['usage_gap_02'] = [at(u5, 'form', 'gap', 0.2), at(sw, 'form', 'gap', 0.2), at(u1, 'form', 'gap', 0.2)]
    v['usage_gap_04'] = [at(u5, 'form', 'gap', 0.4), at(sw, 'form', 'gap', 0.4), at(u1, 'form', 'gap', 0.4)]
    v['usage_vol_02'] = [at(u5, 'form', 'vol', 0.2), at(sw, 'form', 'vol', 0.2), at(u1, 'form', 'vol', 0.2)]
    dist = lambda w, s: np.median(np.concatenate([np.abs(w[('form', s, q)]['final'] - q) for q in md.DIRECTIONS]))
    l30, half = mod['last30'], mod['half']
    v['l30'] = [at(l30, 'form', 'gap', 0.2), at(sw, 'form', 'gap', 0.2), at(l30, 'form', 'vol', 0.2), at(sw, 'form', 'vol', 0.2), dist(l30, 0.2), dist(sw, 0.2)]
    v['half'] = [at(half, 'form', 'gap', 0.2), at(half, 'form', 'vol', 0.2), 2 * np.std(half[('form', 0.02, 0.5)]['final']), 2 * np.std(sw[('form', 0.02, 0.5)]['final'])]
    axis = {e['name']: e for e in forms['axis']}
    for stat, log in (('gap', True), ('vol', False)):
        obs, med, inside = [], [], []
        for nme, runs in mod['fitted'].items():
            o = axis[nme][stat]; m = runs[stat][np.isfinite(runs[stat])]
            if not np.isfinite(o) or len(m) < 6 or (log and o <= 0):
                continue
            obs.append(o); med.append(np.median(m)); inside.append(np.percentile(m, 10) <= o <= np.percentile(m, 90))
        obs, med = np.array(obs), np.array(med); tx, ty = (np.log(obs), np.log(med)) if log else (obs, med)
        v[f'pc_{stat}'] = (np.corrcoef(tx, ty)[0, 1], np.exp(ty.mean()) if log else ty.mean(), np.exp(tx.mean()) if log else tx.mean(), np.mean(inside))
    # SI names and forms
    v['skeleton'] = sin.get('skeleton'); v['first_arrivals'] = sin.get('first_arrivals')
    fam_ = sif['family']; v['fam_conflict'] = fam_['conflict']['pooled']; v['fam_joint'] = list(fam_['joint'][0]); v['fam_n'] = fam_['joint'][2]
    v['fam_by_class'] = {c: list(b[0]) for c, b in fam_['by_class'].items()}; v['fam_fe'] = fam_['fe']
    v['names_fam'] = list(sif['names']['joint'][0]); v['names_conflict'] = sif['names']['conflict']
    v['variants'] = sif['variants']
    from fig3_rule import binned_forms
    _in = _tot = 0
    for _n in sif['keep']:
        _x, _p, _lo, _hi = binned_forms(sif['conv'][_n]['fit_records']); _mu = sif['fits'][_n]
        _pred = _mu[0] + (1 - _mu[0] - _mu[1]) * _x
        _in += int(np.sum((_lo <= _pred) & (_pred <= _hi))); _tot += len(_x)
    v['each_inside'] = (_in, _tot)
    # the reorganized SI, the Extended Data and S7
    from common import contrast
    v['fam_conflict_n'] = fam_['conflict']['pooled'][3]; v['fam_se'] = list(fam_['joint'][1]); v['fam_contrast'] = fam_['contrast']
    v['names_fam_n'] = sif['names'].get('n_joint'); v['names_contrast'] = sif['names']['contrast']
    wc = [contrast(r['same']['b'], r['same']['cov'], [0, 1, -1, 0])['est'] for r in ident['windows'].values()]
    v['win_contrast'] = (min(wc), max(wc))
    v['page_outside'] = (p.get('n_outside'), p['n_decisions'] + p.get('n_outside', 0)); v['own_visible'] = ident.get('own_visible')
    v['nx_contrast'] = [contrast(f['bx'], f['covx'], c) for c in ([1, 0, -1, 0], [0, 1, -1, 0])]
    f2 = load_cache('fig2.pkl'); v['ed1a'] = f2
    pw2 = load_cache('ed_patchwork.pkl')['rows']
    v['pw_above'] = (sum(r['same'] > r['time']['hi'] for r in pw2), sum(r['same'] > r['task']['hi'] for r in pw2), len(pw2))
    v['pw_excess'] = (np.median([r['same'] - r['time']['median'] for r in pw2]), np.median([r['same'] - r['task']['median'] for r in pw2]))
    v['pw_movable'] = (min(r['movable'] for r in pw2), max(r['movable'] for r in pw2))
    for stat in ('gap', 'vol', 'final'):
        obs, med, inside = [], [], []
        for nme, runs in mod['fitted'].items():
            o = axis[nme]['f'] if stat == 'final' else axis[nme][stat]; m = runs[stat][np.isfinite(runs[stat])]
            if not np.isfinite(o) or len(m) < 6 or (stat == 'gap' and o <= 0):
                continue
            obs.append(o); med.append(np.median(m)); inside.append(np.percentile(m, 10) <= o <= np.percentile(m, 90))
        tx, ty = (np.log(obs), np.log(med)) if stat == 'gap' else (np.array(obs), np.array(med))
        v[f'inside_{stat}'] = (sum(inside), len(inside), np.corrcoef(tx, ty)[0, 1])
    lab = load_cache('si_lab.pkl'); ag = lab['agreement']
    tot = lambda key, k: sum(ag[n_][key][k] for n_ in ag)
    v['lab_oss'] = (tot('oss', 'checked'), tot('oss', 'wrong_checked'), tot('oss', 'right_unchecked'))
    v['lab_sol'] = (tot('sol', 'wrong_checked'), tot('sol', 'checked'))
    v['lab_mixed'] = lab['mixed']; v['lab_chains'] = lab['chains']; v['lab_copy'] = lab['copying']
    c20 = [x[0] for x in lab['mixed'][20]['per_country'].values()]; v['lab_country20'] = (min(c20), max(c20))
    v['lab_oss_curve'] = [ag[n_]['oss']['checked'] / ag[n_]['oss']['valid'] for n_ in (0, 32)]
    return v


def checks(v):
    """(file, pattern, values): every group of the pattern must agree with the value at the same position."""
    M, S, E = 'main.tex', 'si.tex', 'extended_data.tex'
    win = lambda w: v[f'win_{w}'][:3]
    return [
        # main text: the record and the population
        (M, r'and the next (\d+) answers, by (\d+) agents on (\d+) pages, all repeated it', [v['ans_next'], v['ans_handles'], v['ans_pages']]),
        (M, r'It contains ([\d,]+) stored revisions of ([\d,]+) wiki pages', list(v['release'])),
        (M, r'the record contains ([\d,]+) edits attributed to ([\d,]+) distinct usernames', [v['edits'] + sum(1 for r in ds_revs_swarm(v)), v['handles_all']]),
        (M, r'We thus study the ([\d,]+) handles', [v['handles']]),
        (M, r'They made ([\d,]+) edits across the wiki, including ([\d,]+) edits on (\d+) pages associated with (\d+) task families', [v['edits'], v['task_edits'], v['task_pages'], v['families']]),
        (M, r'(\d+) handles contributed to one family, (\d+) to two and (\d+) to three or more', [v['one_family'], v['two_families'], v['three_families']]),
        (M, r'edits on (\d+) task pages belonging to (\d+) tasks', [v['task_pages'], v['families']]),
        # where to write
        (M, r'(\w+) out of (\w+) appends went to a page still in the last hundred lines', [round(10 * float(v['in_window'])), 10]),
        (M, r'The chosen page appears within this window for (\d+)\\% of appends; the corresponding coverage is (\d+)\\%, (\d+)\\% and (\d+)\\% for windows of 15, 30 and 300',
         [v['inside_100'], v['inside_15'], v['inside_30'], v['inside_300']]),
        (M, r'Of ([\d,]+) arrivals, (\d+) create a page and ([\d,]+) append', [v['decisions'], v['creations'], v['appends']]),
        (M, r'\$p_c=(\d+)/(\d+)=(0\.\d+)\$', [v['creations'], v['decisions'], v['c']]),
        (M, r'\$p_c=(0\.\d+)\$', [v['c']]),
        # how to sign and how to write
        (M, r'retaining pieces found in at least (\d+) handles', [15]),
        (M, r'we end up with (\d+) name features', [v['n_features']]),
        (M, r'identify (\d+) candidate pairs of competing expressions', [v['candidates']]),
        (M, r'We then retain (\d+) pairs with at least 100 uses', [v['kept']]),
        (M, r'This gives ([\d,]+) first-use observations across the (\d+) conventions', [v['n_first_curve'], v['kept']]),
        (M, r'we took (\d+) conventions in which two forms', [v['kept']]),
        (M, r'still writes \\texttt\{R4\} one time in (\w+)', [('rel', 1 / v['mu_Rn'], 0.1)]),
        (M, r'still writes \\texttt\{CONFIRMED\} one time in (\w+)', [('rel', 1 / v['mu_CONFIRMED'], 0.1)]),
        # the model
        (M, r'The model contains ([\d,]+) agents, matching', [v['agents']]),
        (M, r'Agents arrive uniformly over (\d\.\d) days', [v['days']]),
        (M, r'probability \$u=(0\.\d+)\$ of using a given convention', [v['usage']]),
        (M, r'percentile of (\d+) simulated runs', [200]),
        (M, r'We run (\d+) independent simulations to study page popularity and (\d+) simulations per parameter setting', [200, md.RUNS]),
        # the answers and the laboratory
        (M, r'This yields (\d+) uses on (\d+) pages by (\d+) handles', [v['ans_uses'], v['ans_pages_all'], v['ans_handles_all']]),
        (M, r'followed by (\d+) additional uses of the padded answer', [v['ans_next']]),
        (M, r'checking declined from nearly universal on an empty page to (\d+)\\% with 32 agreeing notes', [str(v['lab_oss_curve'][1])]),
        (M, r'The laboratory dataset contains (\d+) prepared trials: (\d+) trials manipulating page frequencies and name-feature frequencies in the feed, (\d+) related trials using anonymous handles, (\d+) trials manipulating convention frequencies on task pages, and (\d+) empty-page',
         [v['lab_copy']['trials'], *[v['lab_copy']['kinds'][k] for k in ('P', 'P0', 'W', 'W0')]]),
        (M, r'GPT-6\.1 Sol completed (\d+) of (\d+) prepared trials, including (\d+) of (\d+) empty-page trials; (\w+) gpt-oss-20b responses could not be parsed',
         [v['lab_copy']['models']['gpt-6.1-sol']['replies'], v['lab_copy']['trials'], v['lab_copy']['models']['gpt-6.1-sol']['by_kind']['W0'],
          v['lab_copy']['kinds']['W0'], v['lab_copy']['models']['gpt-oss-20b']['errors']]),
        (M, r'for a total of ([\d,]+) agent decisions', [sum(c['agents'] for c in v['lab_chains'].values())]),
        # SI: S1
        (S, r'median of (\d\.\d+)~h among the (\d+) handles', [v['span_h'], v['n_span']]),
        (S, r'(\d+)\\% of these spans exceed 6~h and (\d\.\d)\\% a day', [v['span_6h'], v['span_1d']]),
        (S, r'median of (\d) edits, (\d+)\\% made one, and the most active made (\d+)', [v['edits_median'], v['edits_one'], v['edits_max']]),
        (S, r'median of (\d\.\d)~h, but stayed readable for a median of (\d+) days', [v['page_span_h'], v['readable_d']]),
        (S, r'of the ([\d,]+) pages touched by the population, ([\d,]+) were eventually deleted', [v['pages_touched'], v['pages_deleted']]),
        (S, r'(\d+)\\% were still readable one day after their creation', [v['readable_1d']]),
        (S, r'gathered (\d+) handles, and (\d+) of the ([\d,]+) handles worked on at least one of the seven largest', [v['largest_family'], v['top7_handles'], v['handles']]),
        (S, r'(\d+) handles wrote on one family, (\d+) on two and (\d+) on three or more', [v['one_family'], v['two_families'], v['three_families']]),
        (S, r'the response slopes stay between (0\.\d+) and (0\.\d+), and the regression', [None, None]),
        (S, r'For names the feed carries (0\.\d+), the authors of the first page (0\.\d+), the task-mates out of view (0\.\d+) and other tasks (0\.\d+)\. The coefficient of the task-mates out of view is larger than in the population, where it is (0\.\d+)',
         [v['full_names'][1], v['full_names'][0], v['full_names'][2], v['full_names'][3], v['out_pop']]),
        # SI: S2
        (S, r'rises from (\d+)\\% at 15 lines to (\d+)\\% at 30, (\d+)\\% at 100 and (\d+)\\% at 300, and the fitted slope is (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+)',
         [v['inside_15'], v['inside_30'], v['inside_100'], v['inside_300'], v['slope_15'], v['slope_30'], v['slope_100'], v['slope_300']]),
        (S, r'is (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 30 lines, (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 100 \(medians over 10 runs; the 200 runs of Fig.~\\ref\{fig3\}a give (0\.\d+) for 20 handles\), (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 300, and (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 1,000, against (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) observed',
         [*v['mw_30'], *v['mw_100'], v['p20_model'], *v['mw_300'], *v['mw_1000'], *v['p_tail']]),
        (S, r'with medians of (\d+), (\d+), (\d+) and (\d+) handles against (\d+) observed', [v['mw_max_30'], v['mw_max_100'], v['mw_max_300'], v['mw_max_1000'], v['largest']]),
        (S, r'the largest page has a median of (\d+) and a 5th to 95th percentile of (\d+) to (\d+)', [v['largest_model'], *v['largest_5_95']]),
        (S, r'at least 10, 20 or 40 handles is (0\.\d+), (0\.\d+) and (0\.\d+) and the largest page has (\d+) handles; with a uniform choice among all existing pages the figures are (0\.\d+), (0\.\d+) and (0\.\d+), and (\d+) handles',
         [*v['null_pages'], v['null_max_pages'], *v['null_all'], v['null_max_all']]),
        (S, r'about 630 pages \((\d+) on average over 200 runs\), against (\d+) in the record', [v['model_pages'], v['task_pages']]),
        (S, r'Over the ([\d,]+) arrivals this probability does not depend on how many pages of the same task were already active: it is (0\.\d+) when one or two pages of the task had been edited in the previous 24~h, (0\.\d+) for three to five, (0\.\d+) for six to eleven, (0\.\d+) for twelve to twenty-four and (0\.\d+) beyond, and between (0\.\d+) and (0\.\d+) on every day',
         [v['decisions'], *v['by_active'][1:], v['by_day_min'], v['by_day_max']]),
        (S, r'(\d+) decisions in which an agent created a page (\d+)\\% of the time', [v['by_active_n0'], str(v['by_active'][0])]),
        (S, r'created more often than its later ones, (0\.\d+) against (0\.\d+)', [v['c_first'], v['c_later']]),
        (S, r'Of the ([\d,]+) arrivals, ([\d,]+) were taken with a page of the handle.s own task visible in the last 100 lines, and (\d+)\\% of them created a page, while the (\d+) taken with none visible created one (\d+)\\%',
         [v['decisions'], v['n_vis'], v['c_vis'], v['n_novis'], v['c_novis']]),
        (S, r'Of the ([\d,]+) arrivals after a handle.s first, (\d+)\\% fell on a page of the handle.s own family', [v['n_later'], v['own_later']]),
        (S, r'over ([\d,]+) candidate page-moments', [v['n_attach']]),
        (S, r'the rate is (\d\.\d), (\d\.\d), (\d\.\d) and (\d\.\d) at \$k\$ of 2, 3', v['kernel'][1:5]),
        # SI: S3
        (S, r'lists the (\d+) pieces carried by at least 15 handles. Thirteen are months', [v['n_pieces']]),
        (S, r'no piece between (\d\.\d) and (\d\.\d)', [v['gap_below'], v['gap_above']]),
        (S, r'The (\d+) pieces above the cut', [v['n_specific']]),
        (S, r'The (\d+) pieces below it are generic', [v['n_generic']]),
        (S, r'the earlier task-mates on the page of the first edit carry \$(0\.\d+) \\pm (0\.\d+)\$, the task-mates in the feed \$(0\.\d+) \\pm (0\.\d+)\$ and the task-mates out of view \$(0\.\d+) \\pm (0\.\d+)\$',
         [v['n1'][0], None, v['n1'][1], None, v['n1'][2], None]),
        (S, r'the contrast between them is between (0\.\d+) and (0\.\d+)', list(v['win_contrast'])),
        (S, r'grows from (0\.\d+) to (0\.\d+) at twelve hours', [v['win_6'][2], v['win_12'][2]]),
        (S, r'A newcomer saw on average (\d+) task-mates and (\d+) handles of other tasks in the feed, and (\d+)\\% of the newcomers saw no task-mate',
         [v['seen']['own'], v['seen']['other'], str(v['seen']['none_own'])]),
        (S, r'the coefficients of Table~\\ref\{tab:reg\} are (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) \(([\d,]+) records\), and with the (\d+) task-specific pieces alone (0\.\d+), (0\.\d+), (0\.\d+) and \$(-?0\.\d+)\$',
         [*v['pieces']['all']['b'], v['pieces']['all']['n'], v['n_specific'], *v['pieces']['task-specific']['b']]),
        (S, r'16 features, ([\d,]+) records; standard errors clustered by handle; \$R\^2\$ of (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+)',
         [v['n_names'], *v['names_r2']]),
        (S, r'same skeleton as a task-mate in view (\d+\.\d)\\% of the time, as a task-mate out of view (\d+\.\d)\\% of the time, and as a visible handle of another task (\d+\.\d)\\% of the time',
         [str(x) for x in v['skeleton'][:3]]) if v['skeleton'] else None,
        # SI: S4
        (S, r'lists the (\d+) pairs considered and why (\d+) were dropped', [v['candidates'], v['dropped']]),
        (S, r'the line passes inside the 95\\% interval of (\d+) of the (\d+) bins', list(v['each_inside'])),
        (S, r'(\w+) candidates had fewer than 100 uses', [v['under_100']]),
        (S, r'For (\w+) the minority form never occurs', [v['never']]),
        (S, r'and the pooled slope between (0\.\d+) and (0\.\d+)[.;]', [*v['th_slope']]),
        (S, r'The coefficient of the page lies between (0\.\d+) and (0\.\d+) under every rule, the number of conventions kept between (\d+) and (\d+)', list(v['th_page']) + list(v['th_kept'])),
        (S, r'median width is (0\.\d+) for the kept conventions and (0\.\d+) for the dropped ones', [v['width_kept'], v['width_dropped']]),
        (S, r'between (\d+) and ([\d,]+) uses, on between (\d+) and (\d+) pages, and prior strengths between (0\.\d+) and (0\.\d+)', [*v['uses_range'], *v['pages_range'], *v['s_range']]),
        (S, r'six hours before each first use: (0\.\d+), (0\.\d+) and (0\.\d+) for the three classes, against (0\.\d+), (0\.\d+) and (0\.\d+) at the time of the use',
         [v['variants'][c]['shifted'] for c in ('coined', 'semantic', 'habit')] + [v['variants'][c]['first'] for c in ('coined', 'semantic', 'habit')]),
        (S, r'With all uses in place of first uses the slopes are (0\.\d+), (0\.\d+) and (0\.\d+)', [v['variants'][c]['all'] for c in ('coined', 'semantic', 'habit')]),
        (S, r'In the (\d+) first uses where the page majority and the family majority lie on opposite sides of one half, the handle follows the page (\d+)\\% of the time \(95\\% interval (\d+) to (\d+)\\%\): (\d+)\\% for the coined names, (\d+)\\% for the near-synonyms and (\d+)\\% for the habits',
         [v['fam_conflict_n'], str(v['fam_conflict'][0]), str(v['fam_conflict'][1]), str(v['fam_conflict'][2]), *[str(v_[0]) for v_ in (load_cache('si_forms.pkl')['family']['conflict'][c] for c in ('coined', 'semantic', 'habit'))]]),
        (S, r'on the ([\d,]+) first uses that have all three, with standard errors clustered by handle, the page carries \$(0\.\d+) \\pm (0\.\d+)\$, the family.s other pages \$(0\.\d+) \\pm (0\.\d+)\$ and the feed \$(0\.\d+) \\pm (0\.\d+)\$, and the contrast page minus family is (0\.\d+) \(95\\% interval (0\.\d+) to (0\.\d+)',
         [v['fam_n'], v['fam_joint'][0], v['fam_se'][0], v['fam_joint'][1], v['fam_se'][1], v['fam_joint'][2], v['fam_se'][2], v['fam_contrast']['est'], v['fam_contrast']['lo'], v['fam_contrast']['hi']]),
        (S, r'(\d+) strata, the slope of the response to the page share is \$(0\.\d+) \\pm (0\.\d+)\$ against (0\.\d+) without them', [None, v['fam_fe'][1], v['fam_fe'][2], v['fam_fe'][0]]),
        (S, r'in a joint fit on ([\d,]+) records the earlier same-task authors of the page carry \$(0\.\d+) \\pm (0\.\d+)\$, the same-task handles of the previous six hours on other pages \$(0\.\d+) \\pm (0\.\d+)\$ and the feed \$(0\.\d+) \\pm (0\.\d+)\$; the contrast between the first two, (0\.\d+) \(95\\% interval \$(-?0\.\d+)\$ to (0\.\d+)\).*?follows the page (\d+)\\% of the time \(n = (\d+)\)',
         [v['names_fam_n'], v['names_fam'][0], None, v['names_fam'][1], None, v['names_fam'][2], None, v['names_contrast']['est'], v['names_contrast']['lo'], v['names_contrast']['hi'], str(v['names_conflict'][0]), v['names_conflict'][3]]),
        # SI: S5
        (S, r'In (\d+) of the ([\d,]+) appends the chosen page belongs to another task', list(v['page_outside'])),
        (S, r'which leaves ([\d,]+) candidates of ([\d,]+) appends, each with exactly one chosen candidate', [v['n_cand'], v['n_dec']]),
        (S, r'from 9\.93 and 8\.29 to (\d\.\d+) and (\d\.\d+)', v['clogit'][:2]),
        (S, r'An own-task page that is visible is chosen (\d\.\d)\\% of the time \(n = ([\d,]+)\), and one that has scrolled out of the last 100 lines (\d\.\d)\\% of the time \(n = ([\d,]+)\)',
         [str(v['own_visible'][0]), v['own_visible'][1], str(v['own_visible'][2]), v['own_visible'][3]] if v['own_visible'] else [None] * 4),
        (S, r'to \$(-?0\.\d+) \\pm 0\.\d+\$ on the page, \$(0\.\d+) \\pm 0\.\d+\$ in the feed and \$(-0\.\d+) \\pm 0\.\d+\$ out of view', v['n2'][:3]),
        (S, r'The feed against the exposure out of view holds, \$?(0\.\d+)\$? against \$?(0\.\d+)\$? in the main specification', [v['c_page'][2], v['c_page'][3]]),
        (S, r'survives feature by block intercepts as well, \$?(0\.\d+)\$? in the fourth column', [v['c3'][1]]),
        (S, r'The share among other tasks is \$(0\.\d+) \\pm 0\.\d+\$ in the main specification and \$(0\.\d+) \\pm 0\.\d+\$ with feature by task intercepts', [v['n1'][3], v['n2'][3]]),
        (S, r'It gives \$(0\.\d+) \\pm 0\.\d+\$ for the page, \$(0\.\d+) \\pm 0\.\d+\$ for the same-task uses in the feed, \$(-?0\.\d+) \\pm 0\.\d+\$ for those out of view and \$(-?0\.\d+) \\pm', v['fx']),
        (S, r'the page minus out of view is (0\.\d+) \(95\\% interval (0\.\d+) to (0\.\d+)\), and the feed minus out of view (0\.\d+) \(\$(-?0\.\d+)\$ to (0\.\d+)',
         [v['nx_contrast'][0]['est'], v['nx_contrast'][0]['lo'], v['nx_contrast'][0]['hi'], v['nx_contrast'][1]['est'], v['nx_contrast'][1]['lo'], v['nx_contrast'][1]['hi']]),
        (S, r'the noise of such a share lowers every coefficient \(the page of forms to (0\.\d+)\); with three instances the page of forms rises to (0\.\d+)',
         [v['sens']['at least 1 instance']['forms']['b'][0], v['sens']['at least 3 instances']['forms']['b'][0]]),
        (S, r'lowers the page coefficient of forms to (0\.\d+) and raises the exposure out of view to (0\.\d+)',
         [v['sens']['task of a use from its page']['forms']['b'][0], v['sens']['task of a use from its page']['forms']['b'][2]]),
        (S, r'log-odds higher by \$?(\d\.\d)\$? than a page of another task, a factor of (\d+)', [v['logodds'], round(float(np.exp(v['logodds'])))]),
        (S, r'([\d,]+) first uses of the (\d+) conventions; standard errors clustered by handle; \$R\^2\$ of (0\.\d+), (0\.\d+) and (0\.\d+)', [v['forms_n'], v['kept'], *v['forms_r2']]),
        # SI: S6
        (S, r'than the fitted rule predicts from what is in view, in (\d+) of the (\d+) conventions', [v['habit_more'], v['n_conv']]),
        (S, r'\$s\$ is larger for (\d+) of the (\d+) conventions', [v['first_larger'], v['n_conv']]),
        (S, r'is written in (\d+)\\% of the first uses of its convention and in (\d+)\\% of the later ones', [v['sec_first'], v['sec_later']]),
        (S, r'which ranges from (\d+) for .{0,40}? to (\d+) for .{0,40}? and is a median (\d+)\\% of the uses', [*v['neff']]),
        (S, r'but lies above it, at (\d\.\d) at \$s=(0\.\d+)\$ and (\d\.\d) at \$s=(0\.\d+)\$ against (\d\.\d) and (\d\.\d)', v['first_bins']),
        (S, r'puts one prior at zero, for (\d+) of the (\d+) conventions at \$s=(0\.\d+)\$', [v['rec_floored'][0], v['rec_floored'][1], ('rel', v['rec_floored'][2], 0.05)]),
        (S, r'and for none from \$s=(0\.\d+)\$', [('rel', v['rec_none_from'], 0.05)]),
        (S, r'by a median factor of (\d\.\d) at \$s=(0\.\d+)\$', [v['rec_factor'][0], ('rel', v['rec_factor'][1], 0.05)]),
        (S, r'the model lies between (0\.\d+) and (\d\.\d+) times \$1/s\$ \(.*? and (0\.\d+) from the lowest bin\)', [v['rec_ratio'][0], v['rec_ratio'][1], v['rec_ratio'][2]]),
        (S, r'a value that (\d+)\\% of the refitted model conventions', [v['rec_below']]),
        (S, r'twice the standard deviation of the final share across 50 runs is (\d\.\d) at \$s=0\$, where every run ends at one form or the other, (0\.\d+) at 0\.02, (0\.\d+) at 0\.1, (0\.\d+) at 0\.3',
         [v['sd2'][0.0], v['sd2'][0.02], v['sd2'][0.1], v['sd2'][0.3]]),
        (S, r'with a standard deviation of (0\.\d+), against (0\.\d+) from run to run at a fixed direction', list(v['spread_form_vol'])),
        (S, r'the co-occurrence ratio is (\d\.\d) with a rate of 0\.5, (\d\.\d) with 0\.23 and (\d\.\d) with 0\.1, and the volatility (\d\.\d), (\d\.\d) and (\d\.\d)', [*v['usage_gap_02'], *v['usage_vol_02']]),
        (S, r'\(co-occurrence (\d\.\d) against (\d\.\d), volatility (\d\.\d) against (\d\.\d)\)', v['l30'][:4]),
        (S, r'the co-occurrence ratio falls to (\d\.\d), the volatility to (\d\.\d)', v['half'][:2]),
        (S, r'a geometric mean of (\d\.\d) in the model against (\d\.\d) observed, and is noisy convention by convention, with a correlation of (0\.\d+) between the logarithms and (\d+) of the (\d+)',
         [v['pc_gap'][1], v['pc_gap'][2], v['pc_gap'][0], v['inside_gap'][0], v['inside_gap'][1]]),
        (S, r'on first uses (\d\.\d) and (\d\.\d), with a correlation of (0\.\d+) and (\d+) of (\d+) observed values inside', [v['pc_vol'][2], v['pc_vol'][1], v['pc_vol'][0], v['inside_vol'][0], v['inside_vol'][1]]),
        # SI: S7
        (S, r'gpt-oss-20b answered all (\d+) trials, and (\d+) replies could not be parsed; GPT-6\.1 Sol answered (\d+)',
         [v['lab_copy']['trials'], v['lab_copy']['models']['gpt-oss-20b']['errors'], v['lab_copy']['models']['gpt-6.1-sol']['replies']]),
        (S, r'Of the (\d+) gpt-oss-20b agents that checked, (\w+) answered the padded value, and (\d+) answered right without checking; GPT-6\.1 Sol answered the padded value after checking (\d+) times out of (\d+)',
         [*v['lab_oss'], *v['lab_sol']]),
        (S, r'higher than on unanimous pages by (0\.\d+) \(95\\% bootstrap interval (0\.\d+) to (0\.\d+)\) with 5 notes, (0\.\d+) \((0\.\d+) to (0\.\d+)\) with 10, (0\.\d+) \((0\.\d+) to (0\.\d+)\) with 20 and (0\.\d+) \((0\.\d+) to (0\.\d+)\) with 40',
         [x for nn in (5, 10, 20, 40) for x in (v['lab_mixed'][nn]['diff'], v['lab_mixed'][nn]['lo'], v['lab_mixed'][nn]['hi'])]),
        (S, r'with 20 notes it ranges from (0\.\d+) for the Slovak Republic to (0\.\d+) for Poland', list(v['lab_country20'])),
        (S, r'agents 11 to 30 answer wrong (\d+)\\%, (\d+)\\%, (\d+)\\% and (\d+)\\% of the time for 1, 7, 19 and 32 starting notes, and the last agent answers wrong in (\d+), (\d+), (\d+) and (\d+) of the 20 chains',
         [str(v['lab_chains'][k]['wrong_late']) for k in (1, 7, 19, 32)] + [round(v['lab_chains'][k]['final_wrong'] * v['lab_chains'][k]['chains']) for k in (1, 7, 19, 32)]),
        # Extended Data
        (E, r'\((\d+) blocks, holding ([\d,]+) of the ([\d,]+) task edits of the period; (\d+) to ([\d,]+) edits and (\d+) to (\d+) pages',
         [v['ed1a']['n_blocks'], v['ed1a']['edits_in'], v['ed1a']['edits_period'], *v['ed1a']['edits'], *v['ed1a']['pages']]),
        (E, r'between (\d+)\\% and (\d+)\\% of the uses of a convention are in a group spanning', [str(x) for x in v['pw_movable']]),
        (E, r'97\.5th percentile of the time null for (\d+) of the (\d+) conventions and of the task and time null for (\d+)', [v['pw_above'][0], v['pw_above'][2], v['pw_above'][1]]),
        (E, r'The median excess over the null is (0\.\d+) for the time null and (0\.\d+) for the task and time null', list(v['pw_excess'])),
        (E, r'(\d+) of 16, (\d+) of 15 and (\d+) of 16 conventions, with correlations between observed values and model medians of (0\.\d+) \(of the logarithms\), (0\.\d+) and (0\.\d+)',
         [v['inside_gap'][0], v['inside_vol'][0], v['inside_final'][0], v['inside_gap'][2], v['inside_vol'][2], v['inside_final'][2]]),
        (E, r'the bins lie above the law, at (\d\.\d) at \$s=(0\.\d+)\$ and (\d\.\d) at \$s=(0\.\d+)\$ against (\d\.\d) and (\d\.\d)', v['first_bins']),
        (E, r'The refitted model lies between (0\.\d+) and (\d\.\d+) times \$1/s\$', [v['rec_ratio'][0], v['rec_ratio'][1]]),
        (E, r'\((\d+) of 100 conventions at \$s=(0\.\d+)\$, none from \$s=(0\.\d+)\$\) and overestimates \$s\$ \(median factor (\d\.\d) at \$s=(0\.\d+)\$\)',
         [v['rec_floored'][0], ('rel', v['rec_floored'][2], 0.05), ('rel', v['rec_none_from'], 0.05), v['rec_factor'][0], ('rel', v['rec_factor'][1], 0.05)]),
        (E, r'names 9,728 newcomer--feature records; forms ([\d,]+) first uses', [v['forms_n']]),
        (E, r'forms, ([\d,]+) first uses with all three shares; names, ([\d,]+) records', [v['fam_n'], v['names_fam_n']]),
        # main text: the revised sentences
        (M, r'In (\d+) of the ([\d,]+) appends the chosen page belongs to another task', list(v['page_outside'])),
        (M, r'leaving ([\d,]+)\.', [v['n_dec']]),
        (M, r'The page regression uses ([\d,]+) candidate--decision observations from ([\d,]+) decisions', [v['n_cand'], v['n_dec']]),
        (M, r'a handle.s first use of a convention follows its page (\d+)\\% of the time', [str(v['fam_conflict'][0])]),
        (M, r'the model stays between (0\.\d+) and (\d\.\d+) times the law', [v['rec_ratio'][0], v['rec_ratio'][1]]),
    ]


def ds_revs_swarm(v):
    return range(v['revs_all'] - v['edits'])


def main():
    v = expected_values()
    texts = {f: open(os.path.join(ROOT, 'paper', f)).read() for f in ('main.tex', 'si.tex', 'extended_data.tex')}
    failed = 0
    for chk in checks(v):
        if chk is None:
            continue
        f, pattern, values = chk
        m = re.search(pattern, texts[f], flags=re.S)
        if not m:
            failed += 1
            print(f'MISSING  {f}: no match for /{pattern[:90]}/')
            continue
        bad = []
        for g, val in zip(m.groups(), values):
            if val is None or g is None:
                continue
            try:
                ok = agree(g, val)
            except ValueError:
                ok = False
            if not ok:
                shown = float(val) * 100 if isinstance(val, str) else (val[1] if isinstance(val, tuple) else val)
                bad.append(f'"{g}" vs {shown:.4g}')
        if bad:
            failed += 1
            print(f'DRIFT    {f}: {", ".join(bad)}   in /{pattern[:70]}/')
    total = sum(1 for c in checks(v) if c is not None)
    print(f'\n{total - failed} of {total} checks pass')
    return failed


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
