
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("t-SNE")
class TSNEPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
    def plot(self, save_dir: str, features=None, **kwargs):
        if features is None: return
        init_plot_style()
        try:
            tsne = TSNE(n_components=2, random_state=42)
            embedded = tsne.fit_transform(features)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=embedded[:,0], y=embedded[:,1], hue=self.y_true, palette=['blue', 'red'], alpha=0.7)
            plt.title('t-SNE Feature Map Visualization')
            plt.savefig(os.path.join(save_dir, "tsne_map.png")); plt.close()
        except: pass
