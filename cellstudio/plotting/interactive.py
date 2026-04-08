import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from sklearn.metrics import accuracy_score, f1_score


def show_interactive_threshold_gui(y_true, y_prob):
    """
    Launches a dynamic matplotlib GUI popup with a threshold slider.
    As the slider is dragged, the threshold changes, and accuracy & F1 score recompute real-time.
    Supports native panning and zooming.
    
    Args:
        y_true: Array of ground truth labels (0 or 1).
        y_prob: Array of prediction probabilities for the positive class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    import matplotlib
    # Try to switch to an interactive backend if we are running locally
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        pass # Fallback to default TkAgg or whatever is available

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    
    threshold_init = 0.5
    y_pred = (y_prob >= threshold_init).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    ax.set_title(f"Dynamic Metric Evaluation (Threshold={threshold_init:.2f})\\nAccuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Histogram distribution for visual intuition
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.5, label="Negative Class", color="royalblue")
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.5, label="Positive Class", color="crimson")
    
    line = ax.axvline(x=threshold_init, color='black', linestyle='--', linewidth=2, label="Current Threshold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend()
    
    # Slider UI
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=threshold_init)
    
    def update(val):
        t = slider.val
        yp = (y_prob >= t).astype(int)
        a = accuracy_score(y_true, yp)
        f = f1_score(y_true, yp, zero_division=0)
        ax.set_title(f"Dynamic Metric Evaluation (Threshold={t:.2f})\\nAccuracy: {a:.4f} | F1: {f:.4f}")
        line.set_xdata(t)
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    print("[GUI] Launching interactive plotting window. Waiting for user interaction to close...")
    plt.show(block=True)
