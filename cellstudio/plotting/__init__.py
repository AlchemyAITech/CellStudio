
import matplotlib.pyplot as plt
import seaborn as sns

def init_plot_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.titlesize": 14,
        "figure.figsize": (8, 6), "savefig.dpi": 300, "savefig.bbox": "tight"
    })

from .base import BasePlotter  # noqa: F401
from .registry import PlotterRegistry, PlotterCollection  # noqa: F401
from . import classification  # noqa: F401
from . import curves  # noqa: F401
