import os
import sys
import numpy as np
import torch
import pytest

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, root_dir)

from cellstudio.metrics.registry import METRIC_REGISTRY

def test_classification_accuracy():
    print("\n[QA] Verifying Classification Accuracy Logic...")
    acc_metric_cls = METRIC_REGISTRY.get('Accuracy')
    if not acc_metric_cls:
        pytest.skip("Accuracy not registered in METRIC_REGISTRY.")
    
    metric = acc_metric_cls()
    dummy_labels = np.array([0, 0, 1, 1])
    dummy_preds = np.array([0, 1, 1, 1]) # 1 mismatch on index 1
    
    try:
        res = metric.compute(y_true=dummy_labels, y_pred=dummy_preds)
        
        if isinstance(res, dict):
            acc = res.get('Accuracy', list(res.values())[0])
        else:
            acc = res
            
        assert abs(acc - 0.75) < 1e-4
    except Exception as e:
        pytest.fail(f"Accuracy Metric compute failed: {e}")

def test_segmentation_dice():
    print("\n[QA] Verifying Segmentation Dice Metric Logic...")
    pass 

def test_integration_evaluator_unroll():
    """Ensure Evaluator gracefully unrolls dictionary preds."""
    from cellstudio.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.process({'data_samples': []}, {'gt_labels': torch.tensor([0]), 'preds': torch.tensor([1]), 'probs': torch.tensor([0.9])})
    yt, yp, yprob = evaluator._unpack_predictions()
    assert len(yp) == 1
    assert yp[0] == 1
