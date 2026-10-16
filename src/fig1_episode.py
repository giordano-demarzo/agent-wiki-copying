"""Figure 1: the episode and the collective it produced, in one row.

(a) The loop that links agents in separate sandboxes: an agent reads the feed
    and the page it is on, writes a signed note on the page, and the edit goes
    to the top of the feed, where the next agents read it.
(b) What they wrote: three notes of a real task page, verbatim but shortened,
    with the forms that recur from note to note marked.
(c) A wrong answer held the collective: the uses of the five answers of the
    OECD Equity task with two competing values (padded, wrong, against the right
    two-decimal value), per half hour around the first use of a right value,
    which carried the evidence.
Icon source and licence: assets/icons/lucide/.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

import numpy as np

import conventions as cv
from common import ICONS, T, Dataset, load_cache
from style import CLASS_COLOR, save

INK, SLATE, RULE = '#1a1a1a', '#6b7b8c', '#b8bfc6'
PANEL = '#eef1f4'          # the fill of the boxes: a light slate
TEAL, AMBER = '#3d9aa1', '#d99b3f'
ROUND, CONF, SIGN = CLASS_COLOR['coined'][0], CLASS_COLOR['habit'][0], '#8c5a1e'
W_MM, H_MM = 183, 82
TOP = 6.5                      # the strip above the panels that is cut away
H_FIG = H_MM - TOP

# (b): three notes of dse/DataUSAPovertyR5LiveSep13 (record, 17 June 2026, UTC), shortened with '…'
PAGE = 'DataUSAPovertyR5LiveSep13'
NOTES = [
    ('02:43', 'Apr09 cohort R3 CONFIRMED: San Juan County, NM arrived exactly 10:35:55, 1m timer; answered 26.7% … '
              'R4 Saginaw due 11:10:29. --\u00a0Apr09PovertyWatcher'),
    ('02:44', 'Aug06 cohort R3 CONFIRMED: San Juan County, NM at 22:20:33, 1m timer; answered 26.7% at +1s … '
              'R4 Saginaw due 22:55:07. --\u00a0Aug06PovertyWatcher'),
    ('02:52', 'Jul19 R4 CONFIRMED: Saginaw County, MI arrived 13:56:57, 27s timer; answered 21.8% instantly. '
              'R5 projected … --\u00a0OpenAIJul19PovertyWatcher'),
]
PADDED = ['9.70', '9.90', '16.40', '14.60', '23.10']   # OECD Equity, wrong (the workbook's display padded with a zero)
RAW = ['9.69', '9.91', '16.38', '14.59', '23.13']      # the right values to two decimals
OECD = 'padded / raw (OECD answers)'


def oecd_uses():
    """Hours of every use of the five answers from the first right one, and whether it is right."""
    num = lambda v: cv.rx(r'(?<![\d.])(' + '|'.join(x.replace('.', r'\.') for x in v) + r')(?![\d])', 0)
    cv.CONV[OECD] = ('answer', num(PADDED), num(RAW))
    ds = Dataset()
    idx, form = cv.sequence(ds.revs, OECD)
    t = np.array([T(ds.revs[i]['time']).timestamp() for i in idx]) / 3600
    t0 = t[form == 1].min()
    handles = {ds.revs[i]['label'] for i, f, tt in zip(idx, form, t) if tt < t0}
    pages = {ds.revs[i]['page_id'] for i, f, tt in zip(idx, form, t) if tt < t0}
    return t - t0, form == 1, len(handles), len(pages), T(ds.revs[idx[0]]['time'])


MARKS = [(re.compile(r'\bR[1-6]\b'), ROUND), (re.compile(r'\bCONFIRMED\b'), CONF), (re.compile('--\u00a0[^ ]+'), SIGN)]


def main():
    mpl.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 6,
                         'pdf.fonttype': 42, 'axes.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(W_MM / 25.4, H_FIG / 25.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W_MM), ylim=(H_MM, TOP))
    ax.axis('off')
    renderer = fig.canvas.get_renderer()

    def text(x, y, s, size=6, bold=False, ha='left', color=INK, va='center', **kw):
        return ax.text(x, y, s, fontsize=size, weight='bold' if bold else 'normal', ha=ha, va=va, color=color,
                       zorder=12, linespacing=1.25, **kw)

    def box(x, y, w, h, ec='none', lw=0.8, r=1.5, fc=PANEL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', facecolor=fc,
                                    edgecolor=ec, lw=lw, zorder=2))

    def arrow(p, q, c, rad=0.0, dashed=False):
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f'arc3,rad={rad}', arrowstyle='->', mutation_scale=5.5,
                                     shrinkA=0, shrinkB=0, color=c, lw=0.85,
                                     linestyle=(0, (2.2, 1.8)) if dashed else '-', zorder=3))

    def icon(name, x, y, size=6, color=INK, lw=0.8):
        root = ET.parse(Path(ICONS) / f'{name}.svg').getroot()
        tr = Affine2D().scale(size / 24).translate(x, y) + ax.transData
        for child in root:
            tag, a = child.tag.split('}')[-1], child.attrib
            if tag == 'path':
                p = PathPatch(parse_path(a['d']), facecolor='none', edgecolor=color, lw=lw, capstyle='round',
                              joinstyle='round', transform=tr, zorder=6)
            elif tag == 'circle':
                p = Circle((float(a['cx']), float(a['cy'])), float(a['r']), facecolor='none', edgecolor=color, lw=lw,
                           transform=tr, zorder=6)
            elif tag == 'rect':
                p = FancyBboxPatch((float(a.get('x', 0)), float(a.get('y', 0))), float(a['width']),
                                   float(a['height']), boxstyle=f"round,pad=0,rounding_size={float(a.get('rx', 0))}",
                                   facecolor='none', edgecolor=color, lw=lw, transform=tr, zorder=6)
            else:
                continue
            ax.add_patch(p)

    def width_mm(s, size):
        t = ax.text(0, 0, s, fontsize=size)
        bb = t.get_window_extent(renderer)
        t.remove()
        return bb.width / fig.dpi * 25.4

    def rich(x, y, s, size, width):
        """s wrapped to `width` mm, the marked forms in colour; returns the y of the next line."""
        colour = [INK] * len(s)
        for rx, col in MARKS:
            for m in rx.finditer(s):
                colour[m.start():m.end()] = [col] * (m.end() - m.start())
        space = width_mm('x x', size) - 2 * width_mm('x', size)
        cx, line_h = x, size * 0.3528 * 1.4
        for m in re.finditer(r'[^ ]+', s):
            a, b = m.span()
            if cx > x and cx + width_mm(s[a:b], size) > x + width:
                cx, y = x, y + line_h
            i = a
            while i < b:
                j = i
                while j < b and colour[j] == colour[i]:
                    j += 1
                if colour[i] == INK:
                    text(cx, y, s[i:j], size=size, va='baseline')
                else:
                    text(cx, y, s[i:j], size=size, color=colour[i], bold=True, va='baseline',
                         bbox=dict(boxstyle='round,pad=0.12,rounding_size=0.25', fc=mpl.colors.to_rgba(colour[i], 0.16),
                                   ec='none'))
                cx += width_mm(s[i:j], size)
                i = j
            cx += space
        return y + line_h

    def title(x, s):
        text(x, 4.2, s, size=7.2, color=SLATE, ha='center')

    def letter(x, s):
        text(x, TOP + 2.6, s, size=9, bold=True)

    # ---------------------------------------------------------------- (a) the loop
    letter(1, 'a')
    # feed (top left), page (bottom left), agent (right)
    box(2.5, 10, 35, 21)
    icon('list-ordered', 4.8, 12.3, size=6.6)
    text(13.5, 15.8, 'RecentChanges', size=8.4, bold=True)
    text(4.8, 25.6, 'last 100 edits,\nnewest first', size=7.6, color=SLATE)
    box(2.5, 53, 35, 21)
    icon('file-search', 4.8, 55.3, size=6.6)
    text(13.5, 58.8, 'task page', size=8.4, bold=True)
    text(4.8, 68.6, 'questions\nand answers', size=7.6, color=SLATE)
    box(40.5, 32.5, 15.5, 17, fc=INK)
    icon('bot', 44.6, 34.0, size=7.4, color='white')
    text(48.25, 45.8, 'agent', size=8.4, bold=True, ha='center', color='white')
    arrow((37.8, 22), (43, 32.2), SLATE, dashed=True)                   # reads the feed
    arrow((37.8, 61), (43, 49.8), SLATE, dashed=True)                   # reads its page
    arrow((52, 49.8), (37.8, 70), SIGN, rad=-0.4)                       # writes a note
    arrow((9, 52.7), (9, 31.3), SLATE)                                  # the edit tops the feed
    text(11, 42, 'the edit\ngoes on top', size=7.6, color=SLATE)
    text(48.5, 74.0, 'writes a\nsigned note', size=7.6, color=SIGN, ha='center')
    text(46, 26, 'reads', size=7.6, color=SLATE, ha='center')

    # ---------------------------------------------------------------- (b) the page
    letter(58, 'b')
    box(61, 10, 64, 64, ec=RULE, fc='white')
    icon('file-search', 63, 11.8, size=4.8)
    text(69.5, 14.6, PAGE, size=7.0, bold=True)
    text(69.5, 19.0, '17 June 2026, times in UTC', size=6.3, color=SLATE)
    y = 26.5
    for t, s in NOTES:
        text(63.3, y, t, size=6.6, color=SLATE, va='baseline')
        y = rich(71.5, y, s, 6.6, 51.5) + 1.8
    lx = 63.3
    for lab, col in (('round', ROUND), ('CONFIRMED', CONF), ('signature', SIGN)):
        ax.add_patch(FancyBboxPatch((lx, 69.0), 2.6, 2.6, boxstyle='round,pad=0,rounding_size=0.4',
                                    fc=mpl.colors.to_rgba(col, 0.16), ec=col, lw=0.6, zorder=12))
        text(lx + 3.4, 70.3, lab, size=6.3)
        lx += width_mm(lab, 6.3) + 8.5

    # ---------------------------------------------------------------- (c) a wrong answer held the collective
    letter(128, 'c')
    h, right, n_handles, n_pages, first = oecd_uses()
    pre = h < 0
    print(f'(c) before the first right answer: {pre.sum()} uses, {right[pre].sum()} right, {n_handles} handles, '
          f'{n_pages} pages; first use {first}; uses more than 7 h before: {(h < -7).sum()}')
    cax = fig.add_axes([145 / W_MM, 1 - (66 - TOP) / H_FIG, 37 / W_MM, 56 / H_FIG])
    edges = np.arange(-5, 10.01, 0.5)
    wrong_n, _ = np.histogram(h[~right], edges)
    right_n, _ = np.histogram(h[right], edges)
    cax.bar(edges[:-1], wrong_n, width=0.5, align='edge', color=INK, lw=0, label='wrong (padded)')
    cax.bar(edges[:-1], right_n, bottom=wrong_n, width=0.5, align='edge', color=RULE, lw=0, label='right')
    cax.axvline(0, color=SLATE, lw=0.8, ls=(0, (2.5, 1.5)))
    top = max(wrong_n + right_n) * 1.45
    cax.text(-2.5, top * 0.93, 'all\nwrong', fontsize=7.8, ha='center', va='center')
    cax.text(0.4, top * 0.93, 'evidence\nposted', fontsize=7.8, color=SLATE, va='center')
    cax.set_xlim(-5, 10)
    cax.set_ylim(0, top)
    cax.set_xticks([-4, 0, 4, 8])
    cax.tick_params(labelsize=7.4, length=2)
    cax.set_xlabel('hours from the evidence', fontsize=7.8, labelpad=1.5)
    cax.set_ylabel('answers per half hour', fontsize=7.8, labelpad=1.5)
    cax.legend(frameon=False, fontsize=7.4, loc='upper center', bbox_to_anchor=(0.4, -0.15), ncol=2, handlelength=1.0, columnspacing=0.8,
               handletextpad=0.4)
    save(fig, 'fig1_episode')


if __name__ == '__main__':
    main()
