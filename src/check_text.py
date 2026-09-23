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
    v['excluded'] = len(ds.swarm); v['excluded_18'] = sum(1 for h in ds.swarm if ds.birth[h].strftime('%d %b') == '18 Jun')
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
    # identification
    p, n, f = ident['pages'], ident['names'], ident['forms']
    v['n_records'], v['n_first'], v['n_cand'], v['n_dec'] = n['n'], f['n'], p['n'], p['n_decisions']
    v['own_vis_ratio'] = p['b1'][0] / p['b1'][1]; v['r2'] = (f['r2_1'], n['r2_1'], p['r2_1']); v['fx'] = list(f['bx'])
    v['out_full'], v['out_pop'] = ident_full['names']['b1'][2], n['b1'][2]
    for w, r in ident['windows'].items():
        v[f'win_{w}'] = list(r['same']['b']); v[f'win_other_{w}'] = list(r['other']['b'])
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
    return v


def checks(v):
    """(file, pattern, values): every group of the pattern must agree with the value at the same position."""
    M, S = 'main.tex', 'si.tex'
    win = lambda w: v[f'win_{w}']
    return [
        # main text: the record and the population
        (M, r'leaving ([\d{,}]+) edits under ([\d{,}]+) handles', [v['edits'] + sum(1 for r in ds_revs_swarm(v)), v['handles_all']]),
        (M, r'([\d{,}]+) of the ([\d{,}]+) handles, and ([\d{,}]+) of the ([\d{,}]+) edits, of which ([\d{,}]+) are on the (\d+) task pages of (\d+) task families',
         [v['handles'], v['handles_all'], v['edits'], v['revs_all'], v['task_edits'], v['task_pages'], v['families']]),
        (M, r'(\d+) handles wrote on one family only, (\d+) on two, and (\d+) on three or more', [v['one_family'], v['two_families'], v['three_families']]),
        (M, r'Of the ([\d{,}]+) excluded handles, (\d+) were born on 18 June', [v['excluded'], v['excluded_18']]),
        (M, r'edits on (\d+) task pages belonging to (\d+) tasks', [v['task_pages'], v['families']]),
        (M, r'The agents wrote on (\d+) task pages', [v['task_pages']]),
        # where to write
        (M, r'(\w+) appends in (\w+) went to a page still in the last hundred lines', [round(10 * float(v['in_window'])), 10]),
        (M, r'\$c=(0\.\d+)\$', [v['c']]),
        (M, r'chosen (\w+) times less readily than a page of one', [round(v['own_vis_ratio'])]),
        # how to sign
        (M, r'runs from (0\.\d+) to (0\.\d+) across the 16 name features, against (0\.\d+) to (0\.\d+) across the conventions',
         [*v['s_features'], *v['s_range']]),
        (M, r'pieces carried by at least (\d+) handles', [15]),
        (M, r'Of the remaining (\d+) pieces, (\d+) belong to a single task', [v['n_pieces'] - v['n_months'], v['n_specific']]),
        (M, r'and (\d+) are found across tasks', [v['n_generic']]),
        (M, r'These (\d+) pieces and the date stamp are the (\d+) name features', [v['n_generic'], v['n_features']]),
        # how to write
        (M, r'We start from (\d+) candidate conventions and keep the (\d+) that have', [v['candidates'], v['kept']]),
        (M, r'we took (\d+) conventions in which two forms', [v['kept']]),
        (M, r'still writes \\texttt\{R4\} one time in (\w+)', [('rel', 1 / v['mu_Rn'], 0.1)]),
        (M, r'still writes \\texttt\{CONFIRMED\} one time in (\w+)', [('rel', 1 / v['mu_CONFIRMED'], 0.1)]),
        (M, r'with probability \$?(0\.\d+)\$?, the mean share', [v['usage']]),
        # the model
        (M, r'The model has ([\d{,}]+) identical agents', [v['agents']]),
        (M, r'over the (\d\.\d) days that hold the central', [v['days']]),
        (M, r'percentile over (\d+) runs', [200]),
        (M, r'percentiles over (\d+) runs and the five values', [md.RUNS]),
        # SI: S1
        (S, r'median of (\d\.\d)~h among the (\d+) handles', [v['span_h'], v['n_span']]),
        (S, r'(\d+)\\% of them span more than 6~h and (\d\.\d)\\% more than a day', [v['span_6h'], v['span_1d']]),
        (S, r'median of (\d) edits, (\d+)\\% made only one, and the most active made (\d+)', [v['edits_median'], v['edits_one'], v['edits_max']]),
        (S, r'median of (\d\.\d)~h, but stayed readable for a median of (\d+) days', [v['page_span_h'], v['readable_d']]),
        (S, r'of the ([\d,]+) pages touched by the population, ([\d,]+) were eventually deleted', [v['pages_touched'], v['pages_deleted']]),
        (S, r'(\d+)\\% were still readable one day after their creation and (\d+)\\% one week', [v['readable_1d'], v['readable_1w']]),
        (S, r'gathered (\d+) handles, and (\d+) of the ([\d,]+) handles worked on at least one of the seven largest families', [v['largest_family'], v['top7_handles'], v['handles']]),
        # SI: S3
        (S, r'rises from (\d+)\\% at 15 lines to (\d+)\\% at 30, (\d+)\\% at 100 and (\d+)\\% at 300, and the fitted slope is (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+)',
         [v['inside_15'], v['inside_30'], v['inside_100'], v['inside_300'], v['slope_15'], v['slope_30'], v['slope_100'], v['slope_300']]),
        (S, r'is (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 30 lines, (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 100 \(medians over 10 runs; the 200 runs of Fig.~4a give (0\.\d+) for 20 handles\), (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 300, and (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) with 1,000, against (0\.\d+), (0\.\d+), (0\.\d+) and (0\.\d+) observed',
         [*v['mw_30'], *v['mw_100'], v['p20_model'], *v['mw_300'], *v['mw_1000'], *v['p_tail']]),
        (S, r'the record has (\d+) handles on its largest page, and the median largest page of the four models has (\d+), (\d+), (\d+) and (\d+)',
         [v['largest'], v['mw_max_30'], v['mw_max_100'], v['mw_max_300'], v['mw_max_1000']]),
        (S, r'over 200 runs with 100 lines its median is (\d+) and its 5th to 95th percentile is (\d+) to (\d+)', [v['largest_model'], *v['largest_5_95']]),
        (S, r'at least 10, 20 or 40 handles is (0\.\d+), (0\.\d+) and (0\.\d+) and the largest page has (\d+) handles; with a uniform choice among all existing pages the figures are (0\.\d+), (0\.\d+) and (0\.\d+), and (\d+) handles',
         [*v['null_pages'], v['null_max_pages'], *v['null_all'], v['null_max_all']]),
        (S, r'about 630 pages \((\d+) on average over 200 runs\), against (\d+) in the record', [v['model_pages'], v['task_pages']]),
        (S, r'over the ([\d,]+) decisions, it is (0\.\d+) when one or two pages of the task had been edited in the previous 24~h, (0\.\d+) for three to five, (0\.\d+) for six to eleven, (0\.\d+) for twelve to twenty-four and (0\.\d+) beyond, and it is between (0\.\d+) and (0\.\d+) on every day',
         [v['decisions'], *v['by_active'][1:], v['by_day_min'], v['by_day_max']]),
        (S, r'(\d+) decisions in which an agent created a page (\d+)\\% of the time', [v['by_active_n0'], str(v['by_active'][0])]),
        (S, r'created more often than its later ones, (0\.\d+) against (0\.\d+)', [v['c_first'], v['c_later']]),
        (S, r'of the ([\d,]+) decisions, ([\d,]+) were taken with a page of the handle.s own task visible in the last 100 lines, and (\d+)\\% of them created a page, while the (\d+) taken with none visible created one (\d+)\\%',
         [v['decisions'], v['n_vis'], v['c_vis'], v['n_novis'], v['c_novis']]),
        (S, r'Of the ([\d,]+) handles, (\d+) wrote on one task family only, (\d+) on two and (\d+) on three or more, and of the ([\d,]+) landings after a handle.s first, (\d+)\\% fell on a page of the handle.s own family',
         [v['handles'], v['one_family'], v['two_families'], v['three_families'], v['n_later'], v['own_later']]),
        (S, r'over ([\d,]+) candidate page-moments', [v['n_attach']]),
        (S, r'the rate is (\d\.\d), (\d\.\d), (\d\.\d) and (\d\.\d) at \$k\$ of 2, 3', v['kernel'][1:5]),
        # SI: S4
        (S, r'lists the (\d+) name pieces carried by at least 15 handles. Thirteen are months', [v['n_pieces']]),
        (S, r'no piece between (\d\.\d) and (\d\.\d)', [v['gap_below'], v['gap_above']]),
        (S, r'The sixteen pieces above it', [None]) if v['n_specific'] == 16 else (S, r'The (\w+) pieces above it', [v['n_specific']]),
        (S, r'the authors of the page carry \$(0\.\d+) \\pm (0\.\d+)\$, the task-mates in the feed \$(0\.\d+) \\pm (0\.\d+)\$ and the task-mates out of view \$(0\.\d+) \\pm (0\.\d+)\$',
         [win(6)[0], None, win(6)[1], None, win(6)[2], None]),
        (S, r'with three hours the three coefficients are (0\.\d+), (0\.\d+) and (0\.\d+), and with one and a half hours (0\.\d+), (0\.\d+) and (0\.\d+)', [*win(3), *win(1.5)]),
        (S, r'the two shorter windows, \$(-?0\.\d+)\$ and \$(-?0\.\d+)\$, and (0\.\d+) at six hours', [v['win_other_1.5'][0], v['win_other_3'][0], v['win_other_6'][0]]),
        (S, r'same skeleton as a task-mate in view (\d+\.\d)\\% of the time, as a task-mate out of view (\d+\.\d)\\% of the time, and as a visible handle of another task (\d+\.\d)\\% of the time',
         [str(x) for x in v['skeleton'][:3]]) if v['skeleton'] else None,
        # SI: S5
        (S, r'lists the (\d+) pairs of competing forms considered and why (\d+) were dropped', [v['candidates'], v['dropped']]),
        (S, r'(\w+) candidates had fewer than 100 uses', [v['under_100']]),
        (S, r'For (\w+) the minority form never occurs', [v['never']]),
        (S, r'median width is (0\.\d+) for the kept conventions and (0\.\d+) for the dropped ones', [v['width_kept'], v['width_dropped']]),
        (S, r'between (\d+) and ([\d,]+) uses, on between (\d+) and (\d+) pages, and prior strengths between (0\.\d+) and (0\.\d+)', [*v['uses_range'], *v['pages_range'], *v['s_range']]),
        (S, r'the handle follows the page (\d+)\\% of the time \(95\\% interval (\d+) to (\d+)\\%\), (\d+)\\% for the coined names, (\d+)\\% for the near-synonyms and (\d+)\\% for the habits',
         [str(v['fam_conflict'][0]), str(v['fam_conflict'][1]), str(v['fam_conflict'][2]), *[str(v_[0]) for v_ in (load_cache('si_forms.pkl')['family']['conflict'][c] for c in ('coined', 'semantic', 'habit'))]]),
        (S, r'on the ([\d,]+) first uses that have all three, the page carries a coefficient of \$(0\.\d+) \\pm', [v['fam_n'], v['fam_joint'][0]]),
        (S, r'within convention x family x day|(\d+) strata, the slope of the response to the page share is \$(0\.\d+) \\pm (0\.\d+)\$ against (0\.\d+) without them', [None, v['fam_fe'][1], v['fam_fe'][2], v['fam_fe'][0]]),
        (S, r'the authors of the page carry \$(0\.\d+) \\pm (0\.\d+)\$ against \$(0\.\d+) \\pm (0\.\d+)\$ for the task-mates of the previous six hours who were not among them and \$(0\.\d+) \\pm (0\.\d+)\$ for the feed, and when the page.s authors and the task-mates disagree the newcomer follows the page (\d+)\\% of the time \(n = (\d+)\)',
         [v['names_fam'][0], None, v['names_fam'][1], None, v['names_fam'][2], None, str(v['names_conflict'][0]), v['names_conflict'][3]]),
        # SI: S6, S7
        (S, r'At \$s = 0\.02\$ the final share ranges from (0\.\d+) to (\d\.\d+), at \$s = 0\.1\$ from (0\.\d+) to (0\.\d+), at \$s = 0\.3\$ from (0\.\d+) to (0\.\d+), and at \$s\$ of 0\.6 and 1 from (0\.\d+) to (0\.\d+)',
         [*v['final_0.02'], *v['final_0.1'], *v['final_0.3'], *v['final_06_1']]),
        (S, r'it is (\d\.\d) at \$s = 0\$, where every run ends at one form or the other, (0\.\d+) at 0\.02, (0\.\d+) at 0\.1, (0\.\d+) at 0\.3', [v['sd2'][0.0], v['sd2'][0.02], v['sd2'][0.1], v['sd2'][0.3]]),
        (S, r'at \$s = 0\.2\$ it is (\d\.\d+), (\d\.\d+), (\d\.\d+), (\d\.\d+) and (\d\.\d+) times sampling noise for the five directions, and at 0\.6 it is between (\d\.\d+) and (\d\.\d+)',
         [*v['dir_vol_02'], min(v['dir_vol_06']), max(v['dir_vol_06'])]),
        (S, r'with a standard deviation of (0\.\d+), against (0\.\d+) from run to run at a fixed direction, and for a name feature (0\.\d+) against (0\.\d+)', [*v['spread_form_vol'], *v['spread_name_vol']]),
        (S, r'co-occurrence ratio is (\d\.\d) and (\d\.\d) for the two most one-sided priors, (\d\.\d) and (\d\.\d) for the intermediate ones and (\d\.\d) for the symmetric one, and at \$s = 0\.4\$ the five values are (\d\.\d), (\d\.\d), (\d\.\d), (\d\.\d) and (\d\.\d)',
         [v['dir_gap_02'][0], v['dir_gap_02'][4], v['dir_gap_02'][1], v['dir_gap_02'][3], v['dir_gap_02'][2], *v['dir_gap_04']]),
        (S, r'the dependence is weaker, (0\.\d+) across directions against (0\.\d+) across runs', list(v['spread_name_gap'])),
        (S, r'the co-occurrence ratio is (\d\.\d) with a rate of 0\.5, (\d\.\d) with 0\.23 and (\d\.\d) with 0\.1, and at \$s = 0\.4\$ it is (\d\.\d), (\d\.\d) and (\d\.\d). The volatility hardly depends on it, (\d\.\d), (\d\.\d) and (\d\.\d)',
         [*v['usage_gap_02'], *v['usage_gap_04'], *v['usage_vol_02']]),
        (S, r'the co-occurrence ratio is (\d\.\d) against (\d\.\d), the volatility (\d\.\d) against (\d\.\d) and the distance from the prior (0\.\d+) against (0\.\d+)', v['l30']),
        (S, r'the co-occurrence ratio falls to (\d\.\d) and the volatility to (\d\.\d), and the final share no longer depends on the run, with a spread of (0\.\d+) at \$s = 0\.02\$ against (0\.\d+)', v['half']),
        (S, r'a geometric mean of (\d\.\d) in the model against (\d\.\d) observed, and is noisy convention by convention, with a correlation of (0\.\d+)', [v['pc_gap'][1], v['pc_gap'][2], v['pc_gap'][0]]),
        (S, r'with a correlation of (0\.\d+), and is close in level, (\d\.\d) in the model against (\d\.\d) on average', [v['pc_vol'][0], v['pc_vol'][1], v['pc_vol'][2]]),
        # SI: S2 and S8
        (S, r'0\.(\d+) for the task-mates out of view and 0\.\d+ for other tasks, but the coefficient of the task-mates out of view is larger than in the population, where it is (0\.\d+)', [None, v['out_pop']]),
        (S, r'It gives \$(0\.\d+) \\pm 0\.\d+\$ for the page, \$(0\.\d+) \\pm 0\.\d+\$ for the same-task uses visible in the feed, \$(-?0\.\d+) \\pm 0\.\d+\$ for the same-task uses out of view and \$(-?0\.\d+) \\pm', v['fx']),
        (S, r'with slopes of (0\.\d+), (0\.\d+) and (0\.\d+) for the three classes against (0\.\d+), (0\.\d+) and (0\.\d+) at the time of the use',
         [v['variants'][c]['shifted'] for c in ('coined', 'semantic', 'habit')] + [v['variants'][c]['first'] for c in ('coined', 'semantic', 'habit')]),
    ]


def ds_revs_swarm(v):
    return range(v['revs_all'] - v['edits'])


def main():
    v = expected_values()
    texts = {f: open(os.path.join(ROOT, 'paper', f)).read() for f in ('main.tex', 'si.tex')}
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
