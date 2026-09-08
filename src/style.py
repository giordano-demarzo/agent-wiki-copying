"""One palette and one set of rcParams for every figure.

The data are always black, the model is always teal, a second empirical
series is amber, references are light grey, and the three convention classes
are indigo, moss and rose.
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
               'habit': (ROSE, 'typographic habits')}

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
    import os
    from common import FIGURES
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURES, f'{name}.{ext}'), dpi=250,
                    bbox_inches='tight')
    print(f'wrote figures/{name}.pdf and .png')
