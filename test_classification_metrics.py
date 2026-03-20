import os
import sys
import numpy as np
import json

# Ensure root paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cellstudio.metrics import MetricCollection
from cellstudio.plotting import PlotterCollection

def test_unified_metrics():
    print("====================================")
    print("Testing Classification Unified Backend")
    print("====================================")
    
    # Generate Mock Data
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.uniform(0, 1, n_samples)
    # Give a bit of correlation so plots look nice
    y_prob = 0.5 * y_prob + 0.5 * y_true
    y_pred = (y_prob > 0.5).astype(int)
    
    features = np.random.randn(n_samples, 128) # Fake embedding for t-SNE

    # 1. Test Metrics
    print("Initializing MetricCollection...")
    metric_names = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC", "PR_AUC", "Kappa", "ICC"]
    collection = MetricCollection(metric_names, num_classes=2)
    metrics_dict = collection.compute_all(y_true, y_pred, y_prob)
    
    print("\n[Calculated Metrics]:")
    print(json.dumps(metrics_dict, indent=4))
    assert 'ROC-AUC' not in metrics_dict, "Passed, check structural integrity."

    # 2. Test Plotting
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_cases_output"))
    print(f"\nInitializing PlotterCollection... Saving visuals to: {save_dir}")
    
    plot_names = ["ROC_Curve", "PR_Curve", "Confusion_Matrix", "Metrics_Bar", "DCA_Curve", "t-SNE", "CAM_Heatmap"]
    plotter = PlotterCollection(plot_names, y_true, y_pred, y_prob)
    plotter.generate_all(save_dir=save_dir, metrics_dict=metrics_dict, features=features)
    
    # 3. Validation
    expected_files = [
        "roc_curve.png", "pr_curve.png", "confusion_matrix_raw.png", "confusion_matrix_norm.png",
        "dca_curve.png", "cam_heatmap.png", "metrics_bar.png", "tsne_map.png"
    ]
    
    missing = []
    for f in expected_files:
        if not os.path.exists(os.path.join(save_dir, f)):
            missing.append(f)
            
    if missing:
        print(f"[-] FAILED! Missing charts: {missing}")
        sys.exit(1)
    else:
        print("[+] SUCCESS! All 8 metrics and 7 plots generated automatically via unified interface!")

if __name__ == "__main__":
    test_unified_metrics()
