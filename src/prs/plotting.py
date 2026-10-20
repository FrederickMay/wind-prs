"""Shared matplotlib style so all notebook figures look like one set.

Colours come from a colourblind-validated categorical palette, used in a
fixed order (blue, orange, aqua). Charts use at most three series colours;
GREY is for reference lines and context.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
GREY = "#8a8985"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "axes.edgecolor": "#d6d5d0", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 1.8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "font.family": ["Segoe UI", "DejaVu Sans"],
    })


def label_end(ax, x, y, text, color):
    """Direct label at the end of a line (text in ink, marker in series colour)."""
    ax.plot([x], [y], "o", ms=4, color=color)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def fig(*args, **kw):
    use_style()
    return plt.subplots(*args, constrained_layout=True, **kw)
