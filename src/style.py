"""One palette and one set of rcParams for every figure.

Every colour has one role across the paper:
    black          the record
    teal           the copy model and its theory
    browns         names (dark, mid and light for the record, the binned means and the model)
    indigo, moss, rose   the three classes of conventions: coined names, near-synonyms, habits of style
    vermilion      gpt-oss-20b in the laboratory
    amber          GPT-6.1 Sol in the laboratory
    slate, grey    secondary series, text and references
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

INK = '#1a1a1a'
TEAL = '#3d9aa1'
AMBER = '#d99b3f'
SLATE = '#6b7b8c'
ROSE = '#c1698e'
INDIGO = '#4c5fa8'
MOSS = '#5f9e6e'
GREY = '#b8bfc6'

DATA, MODEL, GREY_REF = INK, TEAL, GREY
CLASS_COLOR = {'coined': (INDIGO, 'coined names'),
               'semantic': (MOSS, 'near-synonyms'),
               'habit': (ROSE, 'habits of style')}
NAME_BROWN = '#8c5a1e'   # names: binned empirical name features
NAME_DARK = '#4f3210'    # names: a darker shade for a third series
NAME_LIGHT = '#c49a6c'   # names: the copy model
OSS = '#c8573e'          # gpt-oss-20b (vermilion)
SOL = AMBER              # GPT-6.1 Sol

RC = {'font.size': 7.5, 'axes.titlesize': 6.4, 'axes.labelsize': 7.4,
      'legend.fontsize': 6, 'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6,
      'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'pdf.fonttype': 42,
      'font.family': 'sans-serif', 'axes.spines.top': False,
      'axes.spines.right': False}


def use():
    plt.rcParams.update(RC)


def panel(ax, letter, x=-0.3):
    ax.text(x, 1.05, letter, transform=ax.transAxes, fontweight='bold',
            fontsize=9, va='bottom')


def save(fig, name):
    """The PDF goes to paper/figs/, where the manuscript includes it; a PNG
    preview goes to figures/."""
    import os
    from common import FIGS, PREVIEW
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(PREVIEW, f'{name}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote paper/figs/{name}.pdf')
