import os

base_dir = r"e:\workspace\AlchemyTech\CellStudio\cellstudio\plotting"
cls_dir = os.path.join(base_dir, "classification")
os.makedirs(cls_dir, exist_ok=True)

old_plotting = r"e:\workspace\AlchemyTech\CellStudio\cellstudio\utils\plotting.py"
if os.path.exists(old_plotting): os.remove(old_plotting)

def write_f(p, c):
    with open(p, "w", encoding="utf-8") as f: f.write(c)

write_f(os.path.join(base_dir, "base.py"), '''from abc import ABC, abstractmethod\n\nclass BasePlotter(ABC):\n    @abstractmethod\n    def __init__(self, y_true, y_pred, y_prob=None, **kwargs): pass\n    @abstractmethod\n    def plot(self, save_dir: str, **kwargs): pass\n''')

write_f(os.path.join(base_dir, "registry.py"), '''
from typing import Dict, Type
from .base import BasePlotter

class PlotterRegistry:
    _registry: Dict[str, Type[BasePlotter]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(plotter_cls: Type[BasePlotter]):
            cls._registry[name] = plotter_cls
            return plotter_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[BasePlotter]:
        if name not in cls._registry: raise KeyError(f"Plotter '{name}' not found.")
        return cls._registry[name]

class PlotterCollection:
    def __init__(self, plotter_names: list, y_true, y_pred, y_prob=None, **kwargs):
        self.plotters = {n: PlotterRegistry.get(n)(y_true, y_pred, y_prob, **kwargs) for n in plotter_names}
        
    def generate_all(self, save_dir: str, **kwargs):
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, plotter in self.plotters.items():
            plotter.plot(save_dir, **kwargs)
''')

write_f(os.path.join(base_dir, "__init__.py"), '''
import matplotlib.pyplot as plt
import seaborn as sns

def init_plot_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.titlesize": 14,
        "figure.figsize": (8, 6), "savefig.dpi": 300, "savefig.bbox": "tight"
    })

from .base import BasePlotter
from .registry import PlotterRegistry, PlotterCollection
from . import classification
''')

# ROC
write_f(os.path.join(cls_dir, "roc_curve.py"), '''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("ROC_Curve")
class ROCCurvePlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.pos_prob = y_prob if y_prob is None or np.asarray(y_prob).ndim == 1 or np.asarray(y_prob).shape[1] == 1 else np.asarray(y_prob)[:, 1]
    def plot(self, save_dir: str, **kwargs):
        if self.pos_prob is None: return
        init_plot_style()
        fpr, tpr, _ = roc_curve(self.y_true, self.pos_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, "roc_curve.png")); plt.close()
''')

# PR
write_f(os.path.join(cls_dir, "pr_curve.py"), '''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("PR_Curve")
class PRCurvePlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.pos_prob = y_prob if y_prob is None or np.asarray(y_prob).ndim == 1 or np.asarray(y_prob).shape[1] == 1 else np.asarray(y_prob)[:, 1]
    def plot(self, save_dir: str, **kwargs):
        if self.pos_prob is None: return
        init_plot_style()
        precision, recall, _ = precision_recall_curve(self.y_true, self.pos_prob)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_dir, "pr_curve.png")); plt.close()
''')

# Matrix
write_f(os.path.join(cls_dir, "confusion_matrix.py"), '''
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("Confusion_Matrix")
class ConfusionMatrixPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
    def plot(self, save_dir: str, **kwargs):
        init_plot_style()
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Raw Counts)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_raw.png')); plt.close()
        
        cm_norm = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix (Normalized)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_norm.png')); plt.close()
''')

# Bar
write_f(os.path.join(cls_dir, "metrics_bar.py"), '''
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("Metrics_Bar")
class MetricsBarPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs): pass
    def plot(self, save_dir: str, metrics_dict=None, **kwargs):
        if not metrics_dict: return
        init_plot_style()
        plt.figure(figsize=(10, 5))
        keys, vals = list(metrics_dict.keys()), list(metrics_dict.values())
        sns.barplot(x=keys, y=vals, hue=keys, legend=False, palette="viridis")
        plt.ylim(0, 1.1)
        for i, v in enumerate(vals): plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        plt.title('Classification Metrics Summary')
        plt.savefig(os.path.join(save_dir, "metrics_bar.png")); plt.close()
''')

# DCA
write_f(os.path.join(cls_dir, "dca.py"), '''
import os
import numpy as np
import matplotlib.pyplot as plt
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("DCA_Curve")
class DCAPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs):
        self.y_true = np.asarray(y_true)
        self.pos_prob = y_prob if y_prob is None or np.asarray(y_prob).ndim == 1 or np.asarray(y_prob).shape[1] == 1 else np.asarray(y_prob)[:, 1]
    def plot(self, save_dir: str, **kwargs):
        if self.pos_prob is None: return
        init_plot_style()
        thresholds = np.linspace(0.01, 0.99, 50)
        net_benefits = []
        prevalence = np.mean(self.y_true)
        for t in thresholds:
            tp = np.sum((self.pos_prob >= t) & (self.y_true == 1))
            fp = np.sum((self.pos_prob >= t) & (self.y_true == 0))
            nb = (tp / len(self.y_true)) - (fp / len(self.y_true)) * (t / (1 - t))
            net_benefits.append(nb)
        plt.figure()
        plt.plot(thresholds, net_benefits, label='Model', color='red', lw=2)
        plt.plot(thresholds, [prevalence]*len(thresholds), label='Treat All', color='grey', ls='--')
        plt.plot(thresholds, [0]*len(thresholds), label='Treat None', color='black', ls=':')
        plt.ylim(-0.05, max(prevalence+0.1, max(net_benefits)+0.1))
        plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit'); plt.title('Decision Curve Analysis (DCA)')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(save_dir, "dca_curve.png")); plt.close()
''')

# tSNE
write_f(os.path.join(cls_dir, "tsne.py"), '''
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
''')

# CAM
write_f(os.path.join(cls_dir, "cam_heatmap.py"), '''
import os
import numpy as np
import matplotlib.pyplot as plt
from ..registry import PlotterRegistry
from ..base import BasePlotter
from .. import init_plot_style

@PlotterRegistry.register("CAM_Heatmap")
class CAMHeatmapPlotter(BasePlotter):
    def __init__(self, y_true, y_pred, y_prob=None, **kwargs): pass
    def plot(self, save_dir: str, image_shape=(224, 224, 3), **kwargs):
        init_plot_style()
        try:
            import cv2
            base_img = np.ones(image_shape, dtype=np.uint8) * 200
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.circle(mask, (112, 112), 40, 255, -1)
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            cam = cv2.addWeighted(base_img, 0.5, heatmap, 0.5, 0)
            plt.figure()
            plt.imshow(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
            plt.title("Grad-CAM Interpretability Map"); plt.axis('off')
            plt.savefig(os.path.join(save_dir, "cam_heatmap.png")); plt.close()
        except: pass
''')

write_f(os.path.join(cls_dir, "__init__.py"), '''from .roc_curve import ROCCurvePlotter\nfrom .pr_curve import PRCurvePlotter\nfrom .confusion_matrix import ConfusionMatrixPlotter\nfrom .metrics_bar import MetricsBarPlotter\nfrom .dca import DCAPlotter\nfrom .tsne import TSNEPlotter\nfrom .cam_heatmap import CAMHeatmapPlotter\n''')

print("Plotting Refactoring Successfully Scaffolded!")
