from typing import Dict, List, Any, Optional
from ..metrics.registry import MetricRegistry
from ..plotting.registry import PlotterRegistry

class Evaluator:
    """
    The Zenith Evaluation Broker.
    Orchestrates decoupled metrics and plotters simultaneously based on the runner's validation outputs.
    Ensures that metric calculation and reporting are seamlessly decoupled from the runtime iteration loop.
    """
    def __init__(self, metrics_cfg: List[Dict], plotters_cfg: List[Dict] = None):
        if not metrics_cfg:
            metrics_cfg = []
        if not plotters_cfg:
            plotters_cfg = []
            
        self.metrics = [MetricRegistry.build(cfg) for cfg in metrics_cfg]
        self.plotters = [PlotterRegistry.build(cfg) for cfg in plotters_cfg]
        self._predictions = []
        self._data_samples = []

    def process(self, data_batch: dict, outputs: Any):
        """Accumulates batches of ground truth and predictions across an epoch."""
        self._data_samples.append(data_batch)
        self._predictions.append(outputs)

    def evaluate(self, work_dir: str) -> Dict[str, float]:
        """
        Calculates all registered metrics and issues plotting commands.
        Returns a flat dictionary of computed scalar metrics for the Logger/Checkpoint hooks to use.
        """
        metrics_result = {}
        import torch
        import numpy as np
        y_true = []
        y_pred = []
        y_prob = None
        
        # Determine the format of accumulated predictions
        # Classification: _predictions = [{'probs': [N, C], 'preds': [N], 'gt_labels': [N]}, ...]
        if self._predictions and isinstance(self._predictions[0], dict) and 'gt_labels' in self._predictions[0]:
            y_true = torch.cat([p['gt_labels'] for p in self._predictions]).numpy()
            y_pred = torch.cat([p['preds'] for p in self._predictions]).numpy()
            y_prob = torch.cat([p['probs'] for p in self._predictions]).numpy()
        else:
            # Fallback for detection/segmentation arrays
            y_true, y_pred, y_prob = self._data_samples, self._predictions, None

        # 1. Compute Base Metrics safely
        for metric in self.metrics:
            try:
                # Metrics expect matching signatures: (preds, targets)
                metric_val = metric.compute(y_true=y_true, y_pred=y_pred, y_prob=y_prob) 
            except TypeError:
                metric_val = metric.compute(self._predictions, self._data_samples)
                
            if isinstance(metric_val, dict):
                metrics_result.update(metric_val)
            elif metric_val is not None:
                metrics_result[metric.__class__.__name__] = metric_val
                
        # 2. Render Advanced Visualizations
        for plotter in self.plotters:
            try:
                try:
                    plotter.plot(save_dir=work_dir, y_true=y_true, y_pred=y_pred, y_prob=y_prob) 
                except TypeError:
                    plotter.plot(save_dir=work_dir, predictions=self._predictions, data_samples=self._data_samples)
            except Exception as e:
                import traceback
                print(f"[Evaluator] Plotter {plotter.__class__.__name__} encountered a non-fatal plotting exception:")
                traceback.print_exc()
            
        # 3. Memory Cleanup post-epoch & Save Predictions
        import os
        import pickle
        pred_file = os.path.join(work_dir, 'predictions.pkl')
        try:
            with open(pred_file, 'wb') as f:
                pickle.dump({'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}, f)
        except Exception as e:
            print(f"[Evaluator] Failed to save predictions to {pred_file}: {e}")
            
        self._predictions.clear()
        self._data_samples.clear()
        
        return metrics_result
