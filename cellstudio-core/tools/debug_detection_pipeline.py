"""
Debug Detection Pipeline — Diagnoses why mAP@50 is zero/low.
Runs ONE batch through the full forward_test() path and prints diagnostics.

Usage:
    python tools/debug_detection_pipeline.py <config_path>
"""
import argparse, sys, os, numpy as np, torch, logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellstudio.engine.config.config import Config
import cellstudio.tasks.object_detection
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.mmdet_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.metrics
import cellstudio.plotting  # noqa: F401

LOG_LINES = []

def log(msg=""):
    LOG_LINES.append(msg)

def sep(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")

def analyze_bbox(name, bboxes):
    bboxes = np.array(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
    if len(bboxes) == 0:
        log(f"  {name}: EMPTY (0 boxes)")
        return
    log(f"  {name}: {len(bboxes)} boxes, shape={bboxes.shape}")
    log(f"    x range: [{bboxes[:,0].min():.1f}, {bboxes[:,2].max():.1f}]")
    log(f"    y range: [{bboxes[:,1].min():.1f}, {bboxes[:,3].max():.1f}]")
    w = bboxes[:,2] - bboxes[:,0]
    h = bboxes[:,3] - bboxes[:,1]
    log(f"    w: min={w.min():.1f} max={w.max():.1f} mean={w.mean():.1f}")
    log(f"    h: min={h.min():.1f} max={h.max():.1f} mean={h.mean():.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    # Suppress all noisy output
    logging.disable(logging.CRITICAL)
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    cfg = Config.fromfile(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    sep(f"CONFIG: {os.path.basename(args.config)}")
    log(f"  Model: {cfg.get('model',{}).get('type','?')}")
    log(f"  img_size: {cfg.get('img_size','N/A')}")
    
    # Build model
    from cellstudio.models.builder import MODEL_REGISTRY
    model = MODEL_REGISTRY.build(cfg.get('model'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Build val dataset
    from cellstudio.datasets.mido import MIDODataset
    from cellstudio.datasets.collate import pseudo_collate
    from torch.utils.data import DataLoader
    
    val_cfg = cfg.get('val_dataloader')
    ds_cfg = val_cfg.get('dataset').copy()
    ds_cfg.pop('type', None)
    val_dataset = MIDODataset(**ds_cfg)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0, shuffle=False, collate_fn=pseudo_collate)
    log(f"  Val size: {len(val_dataset)} images")
    
    # Raw dataset item
    sep("RAW DATASET ITEM")
    raw = val_dataset[0]
    log(f"  Keys: {list(raw.keys())}")
    if 'imgs' in raw:
        log(f"  imgs shape: {raw['imgs'].shape}")
    if 'data_samples' in raw:
        ds = raw['data_samples']
        log(f"  data_samples type: {type(ds).__name__}")
        if isinstance(ds, dict):
            log(f"  data_samples keys: {list(ds.keys())}")
            if 'gt_bboxes' in ds:
                analyze_bbox("GT bboxes (raw)", ds['gt_bboxes'])
            if 'gt_labels' in ds:
                log(f"  GT labels: {ds['gt_labels'][:10]}... (unique: {np.unique(ds['gt_labels'])})")
    
    # Collated batch
    sep("COLLATED BATCH")
    batch = next(iter(val_loader))
    log(f"  Batch keys: {list(batch.keys())}")
    if 'imgs' in batch:
        log(f"  imgs: shape={batch['imgs'].shape} dtype={batch['imgs'].dtype}")
        log(f"  imgs range: [{batch['imgs'].min():.2f}, {batch['imgs'].max():.2f}]")
    if 'data_samples' in batch:
        ds_list = batch['data_samples']
        log(f"  data_samples: list of {len(ds_list)}")
        log(f"  ds[0] type: {type(ds_list[0]).__name__}")
        if isinstance(ds_list[0], dict):
            log(f"  ds[0] keys: {list(ds_list[0].keys())}")
            if 'gt_bboxes' in ds_list[0]:
                analyze_bbox("GT bboxes (sample 0)", ds_list[0]['gt_bboxes'])
    
    # Forward test
    sep("FORWARD_TEST OUTPUT")
    model.eval()
    with torch.no_grad():
        outputs = model.forward_test(batch['imgs'], batch.get('data_samples'))
    
    log(f"  Output type: {type(outputs).__name__}, len: {len(outputs) if isinstance(outputs, list) else 'N/A'}")
    if isinstance(outputs, list):
        for i, res in enumerate(outputs):
            log(f"\n  --- Prediction image {i} ---")
            log(f"    Type: {type(res).__name__}")
            if hasattr(res, 'bboxes'):
                bb = res.bboxes.numpy() if isinstance(res.bboxes, torch.Tensor) else np.array(res.bboxes)
                analyze_bbox("Pred bboxes", bb)
            else:
                log(f"    NO 'bboxes' attribute! attrs: {[a for a in dir(res) if not a.startswith('_')]}")
            if hasattr(res, 'scores'):
                sc = res.scores.numpy() if isinstance(res.scores, torch.Tensor) else np.array(res.scores)
                if len(sc) > 0:
                    log(f"    Scores: min={sc.min():.4f} max={sc.max():.4f} mean={sc.mean():.4f}")
                    log(f"    >0.5:{(sc>0.5).sum()} >0.1:{(sc>0.1).sum()} >0.01:{(sc>0.01).sum()} total:{len(sc)}")
                else:
                    log(f"    Scores: EMPTY")
            if hasattr(res, 'labels'):
                lb = res.labels.numpy() if isinstance(res.labels, torch.Tensor) else np.array(res.labels)
                if len(lb) > 0:
                    log(f"    Labels: unique={np.unique(lb)} count={len(lb)}")
    
    # Manual mAP
    sep("MANUAL mAP COMPUTATION")
    from cellstudio.metrics.detection.core import DetMatchCache, bbox_iou
    
    y_true = [batch.get('data_samples', [])]
    y_pred = [outputs]
    
    log(f"  y_true: list[{len(y_true)}], y_true[0] len={len(y_true[0])}")
    log(f"  y_pred: list[{len(y_pred)}], y_pred[0] len={len(y_pred[0])}")
    
    try:
        DetMatchCache._last_id = None  # Force recompute
        result = DetMatchCache.get(y_true, y_pred, None, 0.5)
        log(f"\n  mAP50:      {result['mAP50']:.4f}")
        log(f"  Precision:  {result['Precision']:.4f}")
        log(f"  Recall:     {result['Recall']:.4f}")
        log(f"  F1:         {result['F1']:.4f}")
        tc = result.get('true_counts', [])
        pc = result.get('pred_counts', [])
        log(f"  GT counts:   {tc[:5]}")
        log(f"  Pred counts: {pc[:5]}")
    except Exception as e:
        import traceback
        log(f"\n  ERROR: {e}")
        log(traceback.format_exc())
    
    # Direct IoU
    sep("DIRECT IoU TEST")
    try:
        ds0 = batch['data_samples'][0]
        gt = np.array(ds0['gt_bboxes']) if isinstance(ds0, dict) and 'gt_bboxes' in ds0 else np.array([])
        if hasattr(outputs[0], 'bboxes'):
            pd = outputs[0].bboxes.numpy() if isinstance(outputs[0].bboxes, torch.Tensor) else np.array(outputs[0].bboxes)
        else:
            pd = np.array([])
        log(f"  GT: {len(gt)} boxes, Pred: {len(pd)} boxes")
        if len(gt) > 0 and len(pd) > 0:
            best_ious = []
            for pi in range(min(10, len(pd))):
                iou_row = bbox_iou(pd[pi:pi+1], gt)
                best_ious.append(float(iou_row.max()) if len(iou_row) > 0 else 0.0)
            log(f"  Best IoU per pred (first 10): {[f'{x:.4f}' for x in best_ious]}")
            log(f"  Max IoU overall: {max(best_ious):.4f}")
        else:
            log(f"  Cannot compute IoU — one side is empty")
    except Exception as e:
        log(f"  IoU Error: {e}")
    
    sep("DEBUG COMPLETE")
    
    # Restore stderr and write log file
    sys.stderr = _stderr
    log_path = os.path.join('work_dirs', f'debug_{config_name}.log')
    os.makedirs('work_dirs', exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG_LINES))
    print(f"Debug log saved to: {log_path}")

if __name__ == '__main__':
    main()
