import os
import sys
import pickle

# Link CellStudio Python path explicitly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellstudio.plotting.registry import PlotterRegistry
import cellstudio.plotting.curves  # noqa: F401

def process_file(pkl_file, work_dir, prefix):
    if not os.path.exists(pkl_file): return
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    plotters = [
        PlotterRegistry.build({'type': 'ROCPlotter'}),
        PlotterRegistry.build({'type': 'ConfusionMatrixPlotter'})
    ]
    
    for plotter in plotters:
        plotter.plot(save_dir=work_dir, y_true=data['y_true'], y_pred=data['y_pred'], y_prob=data['y_prob'])
        
    # Rename default filenames to include prefix
    roc = os.path.join(work_dir, 'roc_curve.png')
    if os.path.exists(roc): os.replace(roc, os.path.join(work_dir, f'{prefix}roc_curve.png'))
    
    cm = os.path.join(work_dir, 'confusion_matrix.png')
    if os.path.exists(cm): os.replace(cm, os.path.join(work_dir, f'{prefix}confusion_matrix.png'))

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_results.py work_dir")
        sys.exit(1)
        
    work_dir = sys.argv[1]
    print(f"Generating decoupled Plot artifacts for {work_dir}...")
    
    process_file(os.path.join(work_dir, 'best_predictions.pkl'), work_dir, 'best_')
    process_file(os.path.join(work_dir, 'predictions.pkl'), work_dir, 'last_')

if __name__ == '__main__':
    main()
