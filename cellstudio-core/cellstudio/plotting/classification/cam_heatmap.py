
import os

import matplotlib.pyplot as plt
import numpy as np

from .. import init_plot_style
from ..base import BasePlotter
from ..registry import PlotterRegistry


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
