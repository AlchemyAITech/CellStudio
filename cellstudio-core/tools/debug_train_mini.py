"""
Debug Mini-Training Script — Trains on small subset with detailed per-epoch diagnostics.
Uses TRAIN split for both training AND evaluation to test if the model can overfit.

Usage: python tools/debug_train_mini.py <config_path> [--epochs 5] [--samples 10]
Output: work_dirs/debug_mini_<config_name>/debug_report.log + training_curves.png
"""
import argparse, sys, os, numpy as np, torch, json, warnings
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
os.environ['ULTRALYTICS_SILENT'] = '1'

from cellstudio.engine.config.config import Config
import cellstudio.tasks.object_detection
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.mmdet_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.metrics
import cellstudio.plotting  # noqa: F401

from cellstudio.models.builder import MODEL_REGISTRY
from cellstudio.datasets.mido import MIDODataset
from cellstudio.metrics.detection.core import DetMatchCache

LOG = []
def log(msg=""):
    LOG.append(str(msg))


def make_batch(dataset, indices):
    """Manually collate dataset items into a batch dict."""
    items = [dataset[i] for i in indices]
    imgs = torch.stack([it['imgs'] for it in items])
    data_samples = [it['data_samples'] for it in items]
    return {'imgs': imgs, 'data_samples': data_samples}


def compute_grad_stats(model):
    grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads.append(p.grad.data.abs().mean().item())
    if not grads:
        return 0.0, 0.0, 0.0
    return float(np.mean(grads)), float(np.min(grads)), float(np.max(grads))


def compute_param_change(model, init_params):
    changes = []
    for name, p in model.named_parameters():
        if name in init_params:
            changes.append((p.data - init_params[name]).abs().mean().item())
    return float(np.mean(changes)) if changes else 0.0


def run_eval(model, dataset, indices, device):
    """Run evaluation and return mAP metrics."""
    model.eval()
    all_gt, all_pred, all_scores = [], [], []
    
    with torch.no_grad():
        for i in indices:
            batch = make_batch(dataset, [i])
            imgs = batch['imgs']
            ds = batch['data_samples'][0]
            gt = np.array(ds.get('gt_bboxes', np.zeros((0,4)))) if isinstance(ds, dict) else np.zeros((0,4))
            
            try:
                outputs = model.forward_test(imgs, batch.get('data_samples'))
                res = outputs[0]
                pb = res.bboxes.numpy() if isinstance(res.bboxes, torch.Tensor) else np.array(res.bboxes)
                ps = res.scores.numpy() if isinstance(res.scores, torch.Tensor) else np.array(res.scores)
            except Exception:
                # Fallback: try direct eval mode prediction for YOLO models
                try:
                    from ultralytics.utils.nms import non_max_suppression
                    model.eval()
                    raw = model.model(imgs.to(device))
                    if isinstance(raw, (tuple, list)):
                        raw = raw[0]
                    if isinstance(raw, dict):
                        # YOLO11 train mode dict - extract one2many
                        om = raw.get('one2many', raw)
                        if isinstance(om, dict) and 'boxes' in om:
                            raw = torch.cat([om['boxes'], om['scores']], dim=1)
                        else:
                            # Can't parse - return empty
                            all_gt.append(gt); all_pred.append(np.zeros((0,4))); all_scores.append(np.array([]))
                            continue
                    dets = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.6)
                    det = dets[0]
                    if det is not None and len(det):
                        pb = det[:, :4].cpu().numpy()
                        ps = det[:, 4].cpu().numpy()
                    else:
                        pb = np.zeros((0, 4))
                        ps = np.array([])
                except Exception:
                    pb = np.zeros((0, 4))
                    ps = np.array([])
            
            all_gt.append(gt)
            all_pred.append(pb)
            all_scores.append(ps)
    
    model.train()
    
    if not all_gt:
        return {'mAP50': 0, 'Precision': 0, 'Recall': 0, 'F1': 0,
                'total_gt_boxes': 0, 'total_pred_boxes': 0,
                'score_min': 0, 'score_max': 0, 'score_mean': 0,
                'scores_gt_05': 0, 'scores_gt_01': 0}
    
    DetMatchCache._last_id = None
    result = DetMatchCache._compute(all_gt, all_pred, all_scores, 0.5)
    result['total_gt_boxes'] = sum(len(g) for g in all_gt)
    result['total_pred_boxes'] = sum(len(p) for p in all_pred)
    
    all_sc = np.concatenate(all_scores) if result['total_pred_boxes'] > 0 else np.array([])
    if len(all_sc) > 0:
        result['score_min'] = float(all_sc.min())
        result['score_max'] = float(all_sc.max())
        result['score_mean'] = float(all_sc.mean())
        result['scores_gt_05'] = int((all_sc > 0.5).sum())
        result['scores_gt_01'] = int((all_sc > 0.1).sum())
    else:
        result['score_min'] = result['score_max'] = result['score_mean'] = 0.0
        result['scores_gt_05'] = result['scores_gt_01'] = 0
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join('work_dirs', f'debug_mini_{config_name}')
    os.makedirs(out_dir, exist_ok=True)
    
    log(f"{'='*60}")
    log(f"  DEBUG MINI-TRAINING: {config_name}")
    log(f"  Epochs: {args.epochs}, Samples: {args.samples}")
    log(f"{'='*60}")
    
    # Build model
    model = MODEL_REGISTRY.build(cfg.get('model'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device); model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n  Total params:  {total_params:,}")
    log(f"  Trainable:     {trainable:,}")
    log(f"  Frozen:        {total_params - trainable:,}")
    
    # Build dataset (train split only)
    ds_cfg = cfg.get('train_dataloader').get('dataset').copy()
    ds_type = ds_cfg.pop('type', 'MIDODataset')
    if ds_type == 'TileMIDODataset':
        from cellstudio.datasets.tile_mido import TileMIDODataset
        dataset = TileMIDODataset(**ds_cfg)
    else:
        dataset = MIDODataset(**ds_cfg)
    n = min(args.samples, len(dataset))
    indices = list(range(n))
    log(f"  Dataset total: {len(dataset)}, using: {n}")
    
    # Test one item
    log(f"\n  --- Sample item 0 ---")
    item = dataset[0]
    log(f"  keys: {list(item.keys())}")
    if 'imgs' in item:
        log(f"  imgs: {item['imgs'].shape} range=[{item['imgs'].min():.2f}, {item['imgs'].max():.2f}]")
    if 'data_samples' in item:
        ds = item['data_samples']
        if isinstance(ds, dict):
            log(f"  ds keys: {list(ds.keys())}")
            gt = ds.get('gt_bboxes', np.zeros((0,4)))
            log(f"  gt_bboxes: {len(gt)} boxes")
            if len(gt) > 0:
                gt = np.array(gt)
                w = gt[:,2] - gt[:,0]; h = gt[:,3] - gt[:,1]
                log(f"    w: [{w.min():.1f}, {w.max():.1f}] h: [{h.min():.1f}, {h.max():.1f}]")
    
    # Setup optimizer
    ocfg = cfg.get('optim_wrapper', {}).get('optimizer', {}).copy()
    opt_type = ocfg.pop('type', 'AdamW')
    optimizer = getattr(torch.optim, opt_type)(model.parameters(), **ocfg)
    log(f"  Optimizer: {opt_type}, lr={ocfg.get('lr', 'N/A')}")
    
    init_params = {name: p.data.clone() for name, p in model.named_parameters()}
    history = defaultdict(list)
    
    for epoch in range(args.epochs):
        log(f"\n{'='*60}")
        log(f"  EPOCH {epoch+1}/{args.epochs}")
        log(f"{'='*60}")
        
        model.train()
        epoch_losses = defaultdict(list)
        np.random.shuffle(indices)
        
        for bi, idx in enumerate(indices):
            optimizer.zero_grad()
            batch = make_batch(dataset, [idx])
            
            try:
                outputs = model.forward_train(batch['imgs'], batch.get('data_samples'))
            except Exception as e:
                log(f"  !!! forward_train CRASHED on item {idx}: {e}")
                import traceback; log(traceback.format_exc())
                continue
            
            loss = outputs.get('loss')
            if loss is None:
                log(f"  !!! 'loss' missing! keys: {list(outputs.keys())}")
                continue
            
            try:
                loss.backward()
            except Exception as e:
                log(f"  !!! backward CRASHED: {e}")
                continue
            
            g_mean, g_min, g_max = compute_grad_stats(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()
            
            for k, v in outputs.items():
                val = v.item() if isinstance(v, torch.Tensor) else (float(v) if isinstance(v, (int, float)) else None)
                if val is not None:
                    epoch_losses[k].append(val)
            
            if bi == 0:
                log(f"\n  Batch 0 details:")
                for k, v in outputs.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    log(f"    {k}: {val}")
                log(f"  grad: mean={g_mean:.6f} min={g_min:.6f} max={g_max:.6f}")
                log(f"  loss.requires_grad: {loss.requires_grad}")
                log(f"  loss.grad_fn: {loss.grad_fn}")
        
        log(f"\n  Epoch {epoch+1} averages:")
        for k, vals in epoch_losses.items():
            avg = np.mean(vals)
            history[k].append(avg)
            log(f"    {k}: {avg:.6f}")
        
        pd = compute_param_change(model, init_params)
        history['param_delta'].append(pd)
        log(f"    param_delta: {pd:.8f}")
        
        lr = optimizer.param_groups[0]['lr']
        history['lr'].append(lr)
        log(f"    lr: {lr:.6f}")
        
        # Eval on same train data
        log(f"\n  Eval (train set):")
        try:
            ev = run_eval(model, dataset, indices, device)
            log(f"    mAP@50:    {ev['mAP50']:.4f}")
            log(f"    Precision: {ev['Precision']:.4f}")
            log(f"    Recall:    {ev['Recall']:.4f}")
            log(f"    F1:        {ev['F1']:.4f}")
            log(f"    GT boxes:  {ev['total_gt_boxes']}")
            log(f"    Pred boxes:{ev['total_pred_boxes']}")
            log(f"    Scores:    [{ev['score_min']:.4f}, {ev['score_max']:.4f}] mean={ev['score_mean']:.4f}")
            log(f"    >0.5: {ev['scores_gt_05']} >0.1: {ev['scores_gt_01']}")
            
            history['mAP50'].append(ev['mAP50'])
            history['Precision'].append(ev['Precision'])
            history['Recall'].append(ev['Recall'])
            history['pred_count'].append(ev['total_pred_boxes'])
            history['score_max'].append(ev['score_max'])
        except Exception as e:
            log(f"    Eval FAILED: {e}")
            import traceback; log(traceback.format_exc())
            for k in ['mAP50','Precision','Recall']: history[k].append(0)
    
    # Plots
    log(f"\n{'='*60}")
    log(f"  GENERATING PLOTS")
    log(f"{'='*60}")
    
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs = list(range(1, args.epochs + 1))
        
        pk, pn = [], []
        if 'loss' in history: pk.append('loss'); pn.append('Total Loss')
        for k in sorted(history.keys()):
            if ('_loss' in k or 'loss_' in k) and k != 'loss' and k not in pk:
                pk.append(k); pn.append(k)
        for k,n in [('mAP50','mAP@50'),('Precision','Precision'),('Recall','Recall'),
                     ('param_delta','Param Delta'),('pred_count','Pred Count'),('score_max','Max Score')]:
            if k in history and k not in pk: pk.append(k); pn.append(n)
        
        npl = len(pk)
        cols = min(3, npl)
        rows = max(1, (npl + cols - 1) // cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        fig.suptitle(f'Debug Mini-Train: {config_name}', fontsize=14, fontweight='bold')
        axes = np.array(axes).flatten() if npl > 1 else [axes]
        
        for ai, (key, name) in enumerate(zip(pk, pn)):
            if ai >= len(axes): break
            ax = axes[ai]
            vals = history[key]
            ax.plot(epochs[:len(vals)], vals, 'b-o', ms=4)
            ax.set_title(name); ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
            if vals: ax.annotate(f'{vals[-1]:.4f}', xy=(len(vals), vals[-1]), fontsize=8, color='red')
        
        for ai in range(npl, len(axes)): axes[ai].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        log(f"  Plot saved: {out_dir}/training_curves.png")
    except Exception as e:
        log(f"  Plot failed: {e}")
    
    # Save
    with open(os.path.join(out_dir, 'debug_report.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))
    with open(os.path.join(out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    
    print(f"Done. Report: {out_dir}/debug_report.log")
    print(f"Plot:   {out_dir}/training_curves.png")


if __name__ == '__main__':
    main()
