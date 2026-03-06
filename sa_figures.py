import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Science Advances (inches) ---
SA_SINGLE_COL = 3.55   # Single column width
SA_DOUBLE_COL = 7.25   # Double column width
SA_MAX_HEIGHT = 7.8    # Maximum allowed figure height


def set_sa_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Myriad Pro", "Myriad", "Helvetica", "Arial", "DejaVu Sans"],

        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "axes.linewidth": 0.3,
        "lines.linewidth": 0.3,

        # Major ticks (Science Advances style)
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,

        # Minor ticks (smaller and lighter)
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.2,
        "ytick.minor.width": 0.2,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "savefig.dpi": 300,
        "figure.dpi": 100,
        "figure.autolayout": False,
    })


def sa_figure(ncols=1, height=3.0):
    """
    Create a figure formatted according to Science Advances guidelines.

    Parameters
    ----------
    ncols : int
        1 -> single column (3.55 in)
        2 -> double column (7.25 in)
    height : float
        Figure height in inches (preferably <= SA_MAX_HEIGHT)
    """
    if ncols == 1:
        width = SA_SINGLE_COL
    elif ncols == 2:
        width = SA_DOUBLE_COL
    else:
        raise ValueError("ncols must be 1 or 2.")

    height = min(height, SA_MAX_HEIGHT)
    fig = plt.figure(figsize=(width, height))
    return fig


def add_panel_label(ax, label, x=-0.12, y=1.02):
    """
    Add a panel label (A, B, C, ...) in Science Advances style,
    positioned outside the axes (similar to the previous label_axes function).
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,   # Important to prevent the text from being clipped
    )


def save_sa(fig, filename, dpi=300):
    """
    Save the figure as a vector PDF with appropriate formatting.
    """
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")

