"""Where to write: how agents pick a page. Writes cache/pages.pkl.

1. The response curve (Fig. 3a). Every time a handle writes on a task page that
   is new to it, each distinct page in the last 100 lines of the feed is a
   candidate; we record the share of those lines it holds and whether it was the
   page chosen. The same with windows of 15, 30 and 300 lines (Fig. S3a).
2. Visibility against audience: the same candidates with the number of handles
   already on them (Fig. S4b), and the attachment kernel over all task pages
   edited in the previous 24 h (Fig. S4a).
3. The creation rate c and what it depends on (Fig. S3c), and the inputs the
   model takes from the record.
"""
import collections

import numpy as np

from common import FEED_LINES, Dataset, Report, T, save_cache, wilson

WINDOWS = (15, 30, 100, 300)
DAY = 24 * 3600


def landing_records(ds, feed_lines=FEED_LINES):
    """Candidate pages for every decision to append to a task page new to the writer."""
    ts_all = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    pid_all = [r['page_id'] for r in ds.revs_all]
    seen = collections.defaultdict(set)
    response, control = [], []
    n_create = n_append = n_in_window = 0
    handles_on = collections.defaultdict(int)
    for r in ds.revs:
        handle, page = r['label'], r['page_id']
        if not ds.is_task(r) or page in seen[handle]:
            continue
        seen[handle].add(page)
        if r['diff_base_reason'] == 'page_created':
            n_create += 1
            handles_on[page] += 1
            continue
        n_append += 1
        k = np.searchsorted(ts_all, T(r['time']).timestamp())
        window = collections.Counter(pid_all[max(0, k - feed_lines):k])
        if page in window:
            n_in_window += 1
            for cand, lines in window.items():
                control.append((lines, handles_on[cand], int(cand == page)))
        for cand, lines in window.items():
            response.append((lines / feed_lines, int(cand == page)))
        handles_on[page] += 1
    return (np.array(response), np.array(control),
            dict(n_create=n_create, n_append=n_append, n_in_window=n_in_window))


def model_inputs(ds):
    """What the model takes from the record: the number of agents, their
    arrival times, how many distinct task pages a handle writes on, how long it
    stays, and the creation probability c (creations per handle-page pair)."""
    task_revs = [r for r in ds.revs if ds.is_task(r)]
    by_page = collections.defaultdict(set)
    pages_of, edits_of = collections.defaultdict(set), collections.Counter()
    n_create = 0
    for r in task_revs:
        by_page[r['page_id']].add(r['label'])
        pages_of[r['label']].add(r['page_id'])
        edits_of[r['label']] += 1
        n_create += r['diff_base_reason'] == 'page_created'
    pages_per = np.array([len(pages_of[h]) for h in ds.handles])
    edits_per = np.array([edits_of[h] for h in ds.handles])
    spans = [(T(v[-1]['time']) - T(v[0]['time'])).total_seconds() / 3600
             for v in ds.by_label.values() if len(v) >= 2]
    birth = np.array([ds.birth[h].timestamp() for h in ds.handles])
    return dict(n_agents=len(ds.handles), handles_per_page=np.array([len(v) for v in by_page.values()]),
                pages_per=pages_per, edits_per=edits_per, c=n_create / pages_per.sum(),
                n_pages=len(by_page), n_edits=len(task_revs), n_create=n_create,
                span_h=float(np.median(spans)), arrival_window=tuple(np.percentile(birth, [5, 95])))


def creation_records(ds):
    """For every decision to write on a task page new to the handle: whether it
    created one, how many pages of the same family had been edited in the
    previous 24 h, whether a page of that family was in the last 100 feed lines,
    whether this was the handle's first task page, and the day."""
    task = [r for r in ds.revs if ds.is_task(r)]
    ts = np.array([T(r['time']).timestamp() for r in task])
    pos = {id(r): k for k, r in enumerate(ds.revs_all)}
    fam_all = [ds.family(r) for r in ds.revs_all]
    seen, first_done, out = collections.defaultdict(set), set(), []
    for i, r in enumerate(task):
        handle, page, fam = r['label'], r['page_id'], ds.family(r)
        if page in seen[handle]:
            continue
        seen[handle].add(page)
        j = np.searchsorted(ts, ts[i] - DAY)
        active = {task[k]['page_id'] for k in range(j, i) if ds.family(task[k]) == fam}
        k = pos[id(r)]
        visible = fam in fam_all[max(0, k - FEED_LINES):k]
        out.append(dict(active=len(active), created=int(r['diff_base_reason'] == 'page_created'),
                        first=int(handle not in first_done), visible=int(visible), day=r['time'][:10],
                        own=int(fam == ds.first_family[handle])))
        first_done.add(handle)
    return out


def attachment_records(ds):
    """For every append to a task page new to the handle: every task page
    edited in the previous 24 h is a candidate; its number of handles so far
    and whether it was chosen."""
    task = [r for r in ds.revs if ds.is_task(r)]
    ts = np.array([T(r['time']).timestamp() for r in task])
    handles_on = collections.defaultdict(set)
    seen, out = collections.defaultdict(set), []
    for i, r in enumerate(task):
        handle, page = r['label'], r['page_id']
        new = page not in seen[handle]
        seen[handle].add(page)
        if new and r['diff_base_reason'] != 'page_created':
            j = np.searchsorted(ts, ts[i] - DAY)
            for c in {task[k]['page_id'] for k in range(j, i)}:
                out.append((len(handles_on[c]), int(c == page)))
        handles_on[page].add(handle)
    return np.array(out)


def cross_table(control, line_bins, handle_bins, nmin=100):
    """P(chosen) for every cell of lines held x handles already there."""
    table = np.full((len(line_bins), len(handle_bins)), np.nan)
    for a, (l0, l1) in enumerate(line_bins):
        for b, (h0, h1) in enumerate(handle_bins):
            m = ((control[:, 0] >= l0) & (control[:, 0] < l1)
                 & (control[:, 1] >= h0) & (control[:, 1] < h1))
            if m.sum() >= nmin:
                table[a, b] = control[m, 2].mean()
    return table


def main():
    say = Report('Where to write')
    ds = Dataset()
    response, control, counts = landing_records(ds)
    inputs = model_inputs(ds)
    total = counts['n_create'] + counts['n_append']
    say(f'decisions on a task page new to the writer: {total}; creations {counts["n_create"]}, appends {counts["n_append"]}')
    say(f'appends landing on a page inside the last {FEED_LINES} feed lines: '
        f'{counts["n_in_window"] / counts["n_append"]:.2f}')
    say(f'response of P(write there) to the page share of the feed: slope '
        f'{np.polyfit(response[:, 0], response[:, 1], 1)[0]:.2f} ({len(response)} candidates)')
    say(f'task pages {inputs["n_pages"]}, task edits {inputs["n_edits"]}, creations {inputs["n_create"]}, '
        f'handle-page pairs {inputs["pages_per"].sum()}: c = {inputs["c"]:.3f}; creations per task edit '
        f'{inputs["n_create"] / inputs["n_edits"]:.3f}')
    say(f'distinct task pages per handle: mean {inputs["pages_per"].mean():.2f}; task edits per handle: mean '
        f'{inputs["edits_per"].mean():.2f}; median activity span {inputs["span_h"]:.2f} h; '
        f'central 90% of arrivals over {(inputs["arrival_window"][1] - inputs["arrival_window"][0]) / 86400:.1f} days')
    hpp = inputs['handles_per_page']
    say(f'handles per task page: P(>=5,10,20,40) = '
        f'{[round(float(np.mean(hpp >= k)), 3) for k in (5, 10, 20, 40)]}, largest {hpp.max()}')

    windows = {}
    for w in WINDOWS:
        r, _, c = landing_records(ds, feed_lines=w)
        windows[w] = dict(response=r, slope=np.polyfit(r[:, 0], r[:, 1], 1)[0],
                          inside=c['n_in_window'] / c['n_append'])
        say(f'window {w:4} lines: appends inside {windows[w]["inside"]:.2f}, slope {windows[w]["slope"]:.2f}')

    creation = creation_records(ds)
    cr = {k: np.array([d[k] for d in creation]) for k in ('active', 'created', 'first', 'visible', 'own')}
    say(f'creation probability over {len(creation)} decisions: {cr["created"].mean():.2f}; '
        f'first task page {cr["created"][cr["first"] == 1].mean():.2f}, later {cr["created"][cr["first"] == 0].mean():.2f}')
    active_bins = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 25), (25, 1000)]
    by_active = []
    for a, b in active_bins:
        m = (cr['active'] >= a) & (cr['active'] < b)
        if m.sum() >= 30:
            by_active.append((a, b, int(m.sum()), *wilson(cr['created'][m].sum(), m.sum())))
    say('  by pages of the task active in the previous 24 h: '
        + ', '.join(f'{a}-{b - 1}: {p:.2f} (n={n})' for a, b, n, p, _, _ in by_active))
    by_day = collections.defaultdict(list)
    for d in creation:
        by_day[d['day']].append(d['created'])
    by_day = {d: np.mean(x) for d, x in sorted(by_day.items()) if len(x) >= 30}
    say('  by day (days with at least 30 decisions): ' + ', '.join(f'{d[5:]}: {p:.2f}' for d, p in by_day.items())
        + f'; range {min(by_day.values()):.2f}-{max(by_day.values()):.2f}')
    v = cr['visible'] == 1
    say(f'  a page of the same task in the last {FEED_LINES} lines: {v.sum()} decisions, created '
        f'{cr["created"][v].mean():.2f}; none: {(~v).sum()} decisions, created {cr["created"][~v].mean():.2f}')
    later = cr['first'] == 0
    say(f'  landings after a handle\'s first: {later.sum()}, on a page of the handle\'s own family '
        f'{cr["own"][later].mean():.2f}')
    fams = collections.Counter()
    for h in ds.handles:
        fams[len({ds.family(r) for r in ds.by_label[h] if ds.is_task(r)})] += 1
    say(f'  handles by number of task families written on: one {fams[1]}, two {fams[2]}, '
        f'three or more {sum(n for k, n in fams.items() if k >= 3)}')

    attach = attachment_records(ds)
    k_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 14), (14, 22), (22, 35), (35, 1000)]
    kernel = []
    for a, b in k_bins:
        m = (attach[:, 0] >= a) & (attach[:, 0] < b)
        if m.sum() >= 150 and attach[m, 1].sum() >= 10:
            p, lo, hi = wilson(attach[m, 1].sum(), m.sum())
            kernel.append((attach[m, 0].mean(), p, lo, hi, m.sum()))
    kernel = np.array(kernel)
    ref = kernel[1, 1] if kernel[0, 0] < 0.5 else kernel[0, 1]
    say(f'attachment kernel over {len(attach)} candidate page-moments, A(k)/A(1): '
        f'{[round(v / ref, 1) for v in kernel[:, 1]]} at k = {[round(v, 1) for v in kernel[:, 0]]}')
    line_bins = [(1, 2), (2, 3), (3, 5), (5, 9), (9, 15), (15, 100)]
    handle_bins = [(1, 2), (2, 4), (4, 7), (7, 12), (12, 25), (25, 1000)]
    table = cross_table(control, line_bins, handle_bins)
    say('cross-table P(chosen), rows = lines held, columns = handles there:')
    for (a, b), row in zip(line_bins, table):
        say(f'  lines {a:2}-{b - 1:2}: ' + ' '.join(f'{x:6.3f}' if not np.isnan(x) else '     -' for x in row))
    say(f'  from 1 line to 15 or more, at fixed handles: x{np.nanmean(table[-1] / table[0]):.1f} on average')

    save_cache('pages.pkl', dict(response=response, control=control, counts=counts, inputs=inputs,
                                 windows=windows, creation=creation, by_active=by_active, by_day=by_day,
                                 kernel=kernel, ref=ref, table=table, line_bins=line_bins,
                                 handle_bins=handle_bins, n_attach=len(attach)))
    say.write()


if __name__ == '__main__':
    main()
