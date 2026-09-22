"""Figure 3: one rule for three choices.
(a) where to write: P(a candidate page is chosen) against its share of the last 100 feed lines;
(b) how to sign: P(a name carries a feature) against its share among the task-mates in view and out of
    view, each adjusted for the other (component plus residual of a joint fit with feature fixed effects);
(c) how to write: P(a first use takes a form) against the share of that form on the page, by class.
"""
import numpy as np

from common import Report, fit, load_cache, wilson
from style import CLASS_COLOR, GREY, INK, SLATE, panel, plt, save, use

PAGE_BINS = [(0, 0.015), (0.015, 0.033), (0.033, 0.065), (0.065, 0.13), (0.13, 0.5)]
NAME_EDGES = (0, 0.1, 0.25, 0.45, 0.65, 0.85, 1.01)


def binned_forms(records, nb=5, nmin=10):
    """First-use records (share on the page, took A), in nb equal bins, Wilson intervals."""
    v = np.asarray(records)
    edges = np.linspace(0, 1, nb + 1)
    xs, ps, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (v[:, 0] >= a) & ((v[:, 0] <= b) if b >= 1 else (v[:, 0] < b))
        if m.sum() >= nmin:
            p, l, h = wilson(v[m, 1].sum(), m.sum())
            xs.append(v[m, 0].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    return np.array(xs), np.array(ps), np.array(lo), np.array(hi)


def main():
    use()
    say = Report('Figure 3')
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    plt.subplots_adjust(wspace=0.55)
    for ax, t in zip(axes, ('Where to write', 'How to sign', 'How to write')):
        ax.set_title(t, fontsize=7.4, pad=12, color=SLATE)

    ax = axes[0]
    panel(ax, 'a')
    response = load_cache('pages.pkl')['response']
    xs, ps, lo, hi = [], [], [], []
    for a, b in PAGE_BINS:
        m = (response[:, 0] >= a) & (response[:, 0] < b)
        if m.sum() >= 20:
            p, l, h = wilson(response[m, 1].sum(), m.sum())
            xs.append(response[m, 0].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    ps, lo, hi = map(np.array, (ps, lo, hi))
    ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt='o', color=INK, ms=3.4, capsize=0, label='empirical')
    ax.plot([0, 0.22], [0, 0.22], color=GREY, lw=0.6, ls='--', label='proportional')
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.22)
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.set_xlabel('page’s share of the feed')
    ax.set_ylabel('P(write there)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')
    say('(a) ' + ', '.join(f'share {x:.3f}: {p:.3f}' for x, p in zip(xs, ps)))

    ax = axes[1]
    panel(ax, 'b')
    r = load_cache('names.pkl')['records']
    m = ~np.isnan(r['view']) & ~np.isnan(r['out'])
    y, xv, xo, feat = r['y'][m], r['view'][m], r['out'][m], r['feature'][m]
    b, se, _ = fit([xv, xo], y, (feat,), cluster=r['handle'][m])
    levels = np.unique(feat)
    X = np.column_stack([np.ones(len(y)), xv, xo] + [(feat == lev).astype(float) for lev in levels[1:]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    for x, coef, col, mk, lab in ((xv, beta[1], INK, 'o', 'task-mates in view'),
                                  (xo, beta[2], SLATE, 's', 'task-mates out of view')):
        adjusted = y.mean() + coef * (x - x.mean()) + resid
        xs, ps, es = [], [], []
        for a, bb in zip(NAME_EDGES[:-1], NAME_EDGES[1:]):
            sel = (x >= a) & (x < bb)
            if sel.sum() >= 25:
                xs.append(x[sel].mean())
                ps.append(adjusted[sel].mean())
                es.append(1.96 * adjusted[sel].std(ddof=1) / np.sqrt(sel.sum()))
        ax.errorbar(xs, ps, yerr=es, fmt=mk, color=col, ms=3.4, capsize=0, label=lab,
                    mfc='white' if mk == 's' else col)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('share of task-mates with the feature')
    ax.set_ylabel('P(name has the feature)')
    ax.legend(frameon=False, fontsize=5.6, loc='upper left', handlelength=1.6)
    say(f'(b) joint fit (n={m.sum()}): in view {b[0]:.2f}, out of view {b[1]:.2f}')

    ax = axes[2]
    panel(ax, 'c')
    by_class = load_cache('forms.pkl')['by_class']
    for cls, (color, label) in CLASS_COLOR.items():
        xs, ps, lo, hi = binned_forms(by_class[cls])
        ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt='o', color=color, ms=3.2, capsize=0, label=label)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('share of the form on the page')
    ax.set_ylabel('P(first use takes the form)')
    ax.legend(frameon=False, fontsize=5.6, loc='upper left')
    save(fig, 'fig3_mechanism')
    say.write()


if __name__ == '__main__':
    main()
