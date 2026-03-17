import numpy as np
from typing import List, Dict, Any, Callable

class MedicalMetrics:
    """
    Advanced Medical Statistical Metrics for Pathology AI Validation.
    Provides parameterized metrics calculation to be emitted to the frontend UI.
    """

    @staticmethod
    def calculate_kappa(y_true: np.ndarray, y_pred: np.ndarray, weights: str = None) -> float:
        """
        Calculates Cohen's Kappa or Weighted Kappa for consistency validation.
        Args:
            y_true: Array of ground truth labels (ints)
            y_pred: Array of predicted labels (ints)
            weights: None, 'linear', or 'quadratic' for weighted kappa
        Returns:
            Cohen's Kappa score.
        """
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(y_true, y_pred, weights=weights)

    @staticmethod
    def calculate_bland_altman(measure1: np.ndarray, measure2: np.ndarray) -> Dict[str, Any]:
        """
        Generates data required for a Bland-Altman plot, typically used to measure 
        agreement between two continuous clinical measurements.
        Args:
            measure1: First set of measurements.
            measure2: Second set of measurements.
        Returns:
            Dictionary containing means, differences, mean_diff, std_diff, and Limits of Agreement (LoA).
        """
        measure1 = np.asarray(measure1)
        measure2 = np.asarray(measure2)
        
        means = np.mean([measure1, measure2], axis=0)
        diffs = measure1 - measure2
        
        md = float(np.mean(diffs))
        sd = float(np.std(diffs, axis=0))
        
        loa_upper = md + 1.96 * sd
        loa_lower = md - 1.96 * sd
        
        return {
            "means": means.tolist(),
            "diffs": diffs.tolist(),
            "mean_diff": md,
            "std_diff": sd,
            "loa_upper": loa_upper,
            "loa_lower": loa_lower
        }

    @staticmethod
    def calculate_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
        """
        Calculates ROC curve points and AUC.
        Args:
            y_true: True binary labels (0 or 1).
            y_score: Target scores, can either be probability estimates or confidence values.
        Returns:
            Dictionary containing False Positive Rates, True Positive Rates, Thresholds, and the AUC.
        """
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc)
        }

    @staticmethod
    def calculate_bootstrap_ci(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metric_func: Callable, 
        n_bootstraps: int = 1000, 
        alpha: float = 0.05,
        seed: int = 42,
        **metric_kwargs
    ) -> Dict[str, float]:
        """
        Calculates Bootstrapped Confidence Intervals (CI) for a given metric.
        Args:
            y_true: Array of ground truth labels
            y_pred: Array of predicted labels or scores
            metric_func: Validation function (e.g., from sklearn.metrics or custom) that takes (y_true, y_pred, **kwargs)
            n_bootstraps: Number of bootstrap iterations.
            alpha: Significance level (default 0.05 gives 95% CI).
            seed: Random seed for reproducibility.
            metric_kwargs: Additional arguments to pass to metric_func.
        Returns:
            Dictionary with mean of bootstraps, lower_bound, and upper_bound.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)
        bootstrapped_scores = []
        
        rng = np.random.RandomState(seed)
        
        # Name matching for some specific sklearn function edge case handling
        func_name = getattr(metric_func, '__name__', '')
        
        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, n)
            
            # Skip if AUC logic needs two classes but we only sampled one
            if "roc_auc" in func_name and len(np.unique(y_true[indices])) < 2:
                continue
                
            try:
                score = metric_func(y_true[indices], y_pred[indices], **metric_kwargs)
                bootstrapped_scores.append(score)
            except Exception:
                # Catch metric specific edge cases where sampling failed criteria
                continue
            
        if not bootstrapped_scores:
            raise ValueError("All bootstrap iterations failed. Check your data and metric function assumptions.")
            
        bootstrapped_scores = np.array(bootstrapped_scores)
        mean_score = float(np.mean(bootstrapped_scores))
        
        # Percentiles
        lower_bound = float(np.percentile(bootstrapped_scores, (alpha / 2.0) * 100))
        upper_bound = float(np.percentile(bootstrapped_scores, (1 - alpha / 2.0) * 100))
        
        return {
            "mean": mean_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "alpha": alpha,
            "n_successful_bootstraps": len(bootstrapped_scores)
        }
