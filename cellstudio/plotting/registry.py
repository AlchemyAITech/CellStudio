"""Plotter registry and collection utilities.

Provides ``PLOTTER_REGISTRY`` for registering visualization classes and
``PlotterCollection`` for batch-generating plots.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Type

from ..core.registry import Registry
from .base import BasePlotter

logger = logging.getLogger(__name__)

PLOTTER_REGISTRY = Registry('plotter')

# Backward-compatible alias.
PlotterRegistry = PLOTTER_REGISTRY


class PlotterCollection:
    """Batch executor for multiple registered plotters.

    Args:
        plotter_names: List of registered plotter names to instantiate.
        y_true: Ground-truth data forwarded to each plotter constructor.
        y_pred: Prediction data forwarded to each plotter constructor.
        y_prob: Optional probability data.
        **kwargs: Extra keyword arguments for plotter constructors.
    """

    def __init__(
        self,
        plotter_names: list[str],
        y_true,
        y_pred,
        y_prob=None,
        **kwargs,
    ) -> None:
        self.plotters: Dict[str, BasePlotter] = {
            n: PLOTTER_REGISTRY.get(n)(y_true, y_pred, y_prob, **kwargs)
            for n in plotter_names
        }

    def generate_all(self, save_dir: str, **kwargs) -> None:
        """Generate all registered plots and save to *save_dir*.

        Args:
            save_dir: Directory to write output figures to.
            **kwargs: Forwarded to each plotter's ``plot()`` method.
        """
        os.makedirs(save_dir, exist_ok=True)
        for name, plotter in self.plotters.items():
            try:
                plotter.plot(save_dir, **kwargs)
            except Exception:
                logger.warning("Plotter '%s' failed.", name, exc_info=True)
