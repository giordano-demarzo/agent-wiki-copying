"""Figure 2: a collective forms within a day.
(a) attention concentrates: task pages written on and effective number of pages per 6-hour block;
(b) how names are dressed over time: share of new handles with each kind of piece, blocks of 100;
(c) the patchwork: agreement of two uses on the same page and on different pages, with the time-shuffled null.
"""
import collections
import datetime

import matplotlib.dates as mdates
import numpy as np

import names as nm
from common import Dataset, Report, T, load_cache
from style import AMBER, CLASS_COLOR, INDIGO, INK, SLATE, panel, plt, save, use

H = 6
BLOCK = 100


def main():
    use()
    say = Report('Figure 2')
    ds = Dataset()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[0.95, 1.1, 1.0]))
    plt.subplots_adjust(wspace=0.8)

    # (a) attention concentration per 6-hour block
    ax = axes[0]
    panel(ax, 'a')
    task = [r for r in ds.revs if ds.is_task(r)]
    times = np.array([T(r['time']) for r in task])
    t0, t1 = datetime.datetime(2026, 6, 15), datetime.datetime(2026, 6, 23)
    edges = [t0 + datetime.timedelta(hours=H * i) for i in range(int((t1 - t0).total_seconds() // 3600 // H) + 1)]
    xs, active, effective = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = [r for r, t in zip(task, times) if a <= t < b]
        if len(m) < 30:
            continue
        c = collections.Counter(r['page_id'] for r in m)
        p = np.array(list(c.values()), float) / len(m)
        xs.append(a + (b - a) / 2)
        active.append(len(c))
        effective.append(1 / np.sum(p ** 2))
    ax.plot(xs, active, 'o-', color=INK, ms=2.6, label='pages written on')
    ax.plot(xs, effective, 's-', color=AMBER, ms=2.6, label='effective number of pages')
    ax.set_yscale('log')
    ax.set_ylim(1, 300)
    ax.set_ylabel('task pages per 6-hour block')
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[16, 18, 20, 22]))
    ax.set_xlim(datetime.datetime(2026, 6, 15, 12), datetime.datetime(2026, 6, 22, 12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d Jun'))
    ax.tick_params(axis='x', labelsize=5.6)
    ax.legend(frameon=False, fontsize=5.6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.6)
    top = np.argsort(active)[-2:]
    say(f'(a) {len(xs)} blocks of {H} h with at least 30 task edits; median pages written on {np.median(active):.0f}, '
        f'median effective number {np.median(effective):.1f}; at the two busiest blocks {sorted(active[i] for i in top)} '
        f'pages, effective {sorted(round(effective[i]) for i in top)}')

    # (b) how names are dressed, over time
    ax = axes[1]
    panel(ax, 'b')
    d = load_cache('names.pkl')
    generic = [p['name'] for p in d['pieces'] if p['cls'] == 'generic']
    specific = [p['name'] for p in d['pieces'] if p['cls'] == 'task-specific']
    kinds = {'affiliation stamp': nm.indicators(ds, nm.AFFILIATION),
             'date stamp': d['features']['date'],
             'role word': nm.indicators(ds, [p for p in generic if p not in nm.AFFILIATION]),
             'task-specific word': nm.indicators(ds, specific)}
    color = {'affiliation stamp': AMBER, 'date stamp': SLATE, 'role word': INDIGO, 'task-specific word': INK}
    n = len(ds.handles) // BLOCK * BLOCK
    x = np.arange(n // BLOCK)
    days = [ds.birth[ds.handles[i + BLOCK // 2]].strftime('%d Jun') for i in range(0, n, BLOCK)]
    for t, v in kinds.items():
        y = v[:n].reshape(-1, BLOCK).mean(1)
        ax.plot(x, y, '-o', color=color[t], lw=1.1, ms=2.4, label=t)
        say(f'(b) {t}: overall {v.mean():.2f}; blocks {" ".join(f"{b:.2f}" for b in y)}')
    ticks = [i for i in range(len(days)) if i == 0 or days[i] != days[i - 1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([days[i] for i in ticks], fontsize=5.6, rotation=40, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('share of new handles with the feature')
    ax.set_xlabel('new handles in order of birth, blocks of 100', fontsize=6.6)
    ax.legend(frameon=False, fontsize=5.6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6,
              columnspacing=1.2)

    # (c) the patchwork
    ax = axes[2]
    panel(ax, 'c')
    rows = sorted(load_cache('forms.pkl')['patchwork'], key=lambda r: r['same'] - r['diff'])
    for j, r in enumerate(rows):
        col = CLASS_COLOR[r['cls']][0]
        ax.plot([r['diff'], r['same']], [j, j], '-', color=col, lw=1.0, alpha=0.6)
        ax.plot(r['same'], j, 'o', color=col, ms=4)
        ax.plot(r['diff'], j, 'o', color=col, ms=3, mfc='white', mew=1)
        ax.plot(r['shuffled'], j, '|', color=SLATE, ms=5, mew=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['name'] for r in rows], fontsize=5.0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('P(two uses take the same form)')
    ax.plot([], [], 'o', color=SLATE, ms=4, label='same page')
    ax.plot([], [], 'o', color=SLATE, ms=3, mfc='white', mew=1, label='different pages')
    ax.plot([], [], '|', color=SLATE, ms=5, label='same page, time-shuffled')
    for col, lab in CLASS_COLOR.values():
        ax.plot([], [], 's', color=col, ms=3.5, label=lab)
    ax.legend(frameon=False, fontsize=5.4, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.2,
              columnspacing=1.0)
    save(fig, 'fig2_collective')
    say.write()


if __name__ == '__main__':
    main()
