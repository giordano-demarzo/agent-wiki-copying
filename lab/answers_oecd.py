"""Do agents copy answers? The OECD Equity task as a convention with consequences.

For five questions of the OECD Equity task (a share for the Czech Republic,
Hungary, Poland, the Slovak Republic and Slovenia) the record carries two
competing values: a 'padded' one with a trailing zero (9.70, 9.90, 16.40, 14.60,
23.10) and a 'raw' one with two real decimals (9.69, 9.91, 16.38, 14.59, 23.13).
Treated as a convention with forms A = padded and B = raw, the answer goes
through the paper's own machinery: the uses in time order, the page-first
exposure, the copy rule fitted by maximum likelihood, the agreement on the same
page against different pages, and the time-shuffled null.

Run from agent-wiki-copying/src. Writes lab/results/answers_oecd.txt and
lab/plots/answers_oecd.png.
"""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
import conventions as cv  # noqa: E402
from analysis_forms import agreement, by_page, shuffled_agreement  # noqa: E402
from common import T, Dataset, block_of, wilson  # noqa: E402
from style import AMBER, GREY, INK, MODEL, panel, plt, use  # noqa: E402

PADDED = ['9.70', '9.90', '16.40', '14.60', '23.10']
RAW = ['9.69', '9.91', '16.38', '14.59', '23.13']
NAME = 'padded / raw (OECD answers)'


def num(values):
    return cv.rx(r'(?<![\d.])(' + '|'.join(v.replace('.', r'\.') for v in values) + r')(?![\d])', 0)


cv.CONV[NAME] = ('answer', num(PADDED), num(RAW))


def main():
    use()
    L = []
    say = lambda s='': (print(s), L.append(s))
    ds = Dataset()
    idx, form = cv.sequence(ds.revs, NAME)
    a = (form == 0).astype(int)                       # 1 = padded
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    pages = np.array([page[i] for i in idx])
    times = np.array([T(ds.revs[i]['time']).timestamp() for i in idx], float)
    say(f'{NAME}: {len(idx)} uses on {len(set(pages))} pages by {len({label[i] for i in idx})} handles; '
        f'padded in {a.mean():.2f}')

    # each question separately: which value
    for p, r in zip(PADDED, RAW):
        cp, cr = cv.rx(r'(?<![\d.])' + p.replace('.', r'\.') + r'(?![\d])', 0), cv.rx(r'(?<![\d.])' + r.replace('.', r'\.') + r'(?![\d])', 0)
        n_p = sum(cp(ds.revs[i]['added'] or '') > 0 for i in idx)
        n_r = sum(cr(ds.revs[i]['added'] or '') > 0 for i in idx)
        say(f'  {p} / {r}: edits mentioning padded {n_p}, raw {n_r}')

    # the copy rule
    share, share_page = cv.exposure(idx, form, label, page)
    seen, first = set(), []
    for k, i in enumerate(idx):
        if label[i] not in seen:
            seen.add(label[i])
            first.append(k)
    first = np.array(first)
    mu = cv.fit_mu(share, a.astype(float))
    s_hat, s_lo, s_hi = cv.profile_interval(share, a.astype(float))
    say(f'copy rule on every use with >=3 in view (n={np.isfinite(share).sum()}): mu_padded {mu[0]:.3f}, '
        f'mu_raw {mu[1]:.3f}, s {mu.sum():.2f} (profile {s_hat:.2f}, 95% {s_lo:.2f}-{s_hi:.2f})')
    x, y = share_page[first], a[first]
    m = np.isfinite(x)
    slope = np.polyfit(x[m], y[m], 1) if m.sum() > 5 else [np.nan, np.nan]
    say(f'first uses with >=3 on the page: n={m.sum()}, OLS slope {slope[0]:.2f}, intercept {slope[1]:.2f}')
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    xb, yb, eb = [], [], []
    for lo, hi in bins:
        sel = m & (x >= lo) & (x < hi)
        if sel.sum() >= 3:
            p_, l_, h_ = wilson(y[sel].sum(), sel.sum())
            xb.append(x[sel].mean()); yb.append(p_); eb.append((p_ - l_, h_ - p_))
            say(f'  page share padded {lo:.1f}-{hi:.1f}: n={sel.sum()}, P(padded) {p_:.2f}')
    # every use, page-first exposure
    mm = np.isfinite(share)
    sl_all = np.polyfit(share[mm], a[mm], 1)
    say(f'every use, page-first exposure: n={mm.sum()}, slope {sl_all[0]:.2f}')

    # the patchwork
    same, diff = agreement(by_page(pages, a))
    blocks = np.array([block_of(t) for t in times])
    null = shuffled_agreement(pages, a.copy(), blocks, np.random.default_rng(0))
    say(f'agreement: same page {same:.2f}, different pages {diff:.2f}, same page after shuffling within 3-h blocks {null:.2f}')

    # over time
    days = np.array([ds.revs[i]['time'][:10] for i in idx])
    for d in sorted(set(days)):
        sel = days == d
        say(f'  {d}: {sel.sum():4d} uses, padded {a[sel].mean():.2f}')

    # figure: response and timeline
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4))
    plt.subplots_adjust(wspace=0.45)
    ax = axes[0]
    panel(ax, 'a')
    ax.errorbar(xb, yb, yerr=np.array(eb).T, fmt='o', color=INK, ms=4, capsize=2)
    gx = np.linspace(0, 1, 2)
    ax.plot(gx, mu[0] + (1 - mu.sum()) * gx, '-', color=MODEL, lw=1.2, label=f'copy rule, $s$ = {mu.sum():.2f}')
    ax.plot([0, 1], [0, 1], '--', color=GREY, lw=0.7)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel='share padded on the page',
           ylabel='P(first use is padded)')
    ax.legend(frameon=False, fontsize=5.6, loc='upper left')
    ax = axes[1]
    panel(ax, 'b')
    order = np.argsort(times)
    k = 40
    roll = np.convolve(a[order], np.ones(k) / k, mode='valid')
    tt = (times[order][k - 1:] - times[order][0]) / 86400
    ax.plot(tt, roll, '-', color=AMBER, lw=1.2)
    ax.set(ylim=(-0.03, 1.03), xlabel='days since first use', ylabel=f'share padded (last {k} uses)')
    out = ROOT.parent / 'figures' / 'answers_oecd.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    (ROOT.parent / 'results' / 'answers_oecd.txt').write_text('\n'.join(L) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
