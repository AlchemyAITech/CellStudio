from typing import Dict, List, Any
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
            y_true, y_pred, y_prob = [batch.get('data_samples', []) for batch in self._data_samples], self._predictions, None

        # 1. Compute Base Metrics safely
        for metric in self.metrics:
            # Metrics expect matching signatures: (preds, targets)
            metric_val = metric.compute(y_true=y_true, y_pred=y_pred, y_prob=y_prob) 
                
            if isinstance(metric_val, dict):
                metrics_result.update(metric_val)
            elif metric_val is not None:
                metrics_result[metric.__class__.__name__] = metric_val
                
        # 2. Plotting is deferred: Model results are written to prediction files.
        # User requested to only plot Best and Last epochs, which is handled
        # externally via plot_results.py to avoid epoch-loop matplotlib bottlenecks.
            
        # 3. Memory Cleanup post-epoch & Save Predictions
        import os
        
        # Determine task type heuristically to avoid picking up dense mask blobs
        # Determine task type heuristically to avoid picking up dense mask blobs
        if self._predictions and not isinstance(self._predictions[0], list):
            import pickle
            pred_file = os.path.join(work_dir, 'predictions.pkl')
            try:
                # Disable massively dense pickles for Instance Segmentation (List of infer results)
                if len(self._predictions) > 0 and 'CellStudioInferResult' not in str(type(self._predictions[0])):
                    with open(pred_file, 'wb') as f:
                        pickle.dump({'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}, f)
            except Exception:
                pass
            
        self._predictions.clear()
        self._data_samples.clear()
        
        return metrics_result
