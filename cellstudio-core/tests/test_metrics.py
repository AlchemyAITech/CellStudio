import numpy as np
import pytest  # noqa: F401
from pathstudio.utils.metrics import MedicalMetrics

def test_kappa():
    y_true = np.array([0, 1, 2, 2, 0])
    y_pred = np.array([0, 1, 1, 2, 0])
    kappa = MedicalMetrics.calculate_kappa(y_true, y_pred)
    assert 0.0 <= kappa <= 1.0

def test_bland_altman():
    m1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m2 = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    res = MedicalMetrics.calculate_bland_altman(m1, m2)
    assert "mean_diff" in res
    assert "loa_upper" in res
    assert "loa_lower" in res
    assert res["loa_upper"] > res["loa_lower"]

def test_roc_curve():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.8, 0.2, 0.9])
    res = MedicalMetrics.calculate_roc_curve(y_true, y_score)
    assert res["auc"] == 1.0

def test_bootstrap_ci():
    from sklearn.metrics import accuracy_score
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    res = MedicalMetrics.calculate_bootstrap_ci(
        y_true, y_pred, accuracy_score, n_bootstraps=50, seed=42
    )
    assert "mean" in res
    assert "lower_bound" in res
    assert "upper_bound" in res
    assert res["lower_bound"] <= res["mean"] <= res["upper_bound"]
