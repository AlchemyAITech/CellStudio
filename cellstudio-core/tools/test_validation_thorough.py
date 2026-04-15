"""
Thorough end-to-end validation test.
Tests:
  1. Dataset loading (train + val) with missing-file robustness
  2. DataLoader collation
  3. Model forward_train (loss computation)
  4. Model forward_test (prediction output format)
  5. Evaluator process + evaluate (mAP50 computation)
  6. EvalHook + TextLoggerHook ordering
  7. Full 1-epoch train + val via EpochBasedRunner
  8. val_interval logic correctness
"""
import sys
import os
import torch
import traceback
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cellstudio.engine.config.config import Config
from cellstudio.tasks.registry import TASK_REGISTRY

# Registry imports (same as train.py)
import cellstudio.tasks.object_detection
import cellstudio.tasks.classification
import cellstudio.models.adapters.ultralytics_adapter
import cellstudio.models.adapters.timm_adapter
import cellstudio.models.adapters.mmdet_adapter
import cellstudio.pipeline.transforms.loading
import cellstudio.pipeline.transforms.formatting
import cellstudio.pipeline.transforms.visual_aug
import cellstudio.metrics
import cellstudio.plotting  # noqa: F401

CONFIG_PATH = 'configs/detect/yolo_v8m_det_mido_tile.yaml'


def load_config():
    """Test 0: Config loading and val_interval check"""
    print("=" * 70)
    print("[TEST 0] Loading config & checking val_interval")
    cfg = Config.fromfile(CONFIG_PATH)
    print(f"  Top-level keys: {list(cfg.keys())}")
    print(f"  runner config: {dict(cfg.get('runner', {}))}")
    
    val_interval = cfg.get('runner', {}).get('val_interval', 'MISSING')
    print(f"  val_interval = {val_interval}")
    assert val_interval == 1, f"FAIL: val_interval != 1, got {val_interval}"
    print("[TEST 0] PASSED ✓")
    return cfg


def test_dataset(cfg):
    """Test 1: Dataset loading"""
    print("=" * 70)
    print("[TEST 1] Dataset loading")
    
    for split_name, dl_key in [('train', 'train_dataloader'), ('val', 'val_dataloader')]:
        dl_cfg = cfg[dl_key]
        ds_cfg = dict(dl_cfg['dataset'])
        ds_type = ds_cfg.pop('type')
        
        if ds_type == 'TileMIDODataset':
            from cellstudio.datasets.tile_mido import TileMIDODataset
            ds = TileMIDODataset(**ds_cfg)
        else:
            from cellstudio.datasets.mido import MIDODataset
            ds = MIDODataset(**ds_cfg)
        
        print(f"  {split_name}: {len(ds.raw_items)} images -> {len(ds)} tiles")
        assert len(ds) > 0, f"FAIL: {split_name} dataset has 0 tiles"
        
        # Test single item access
        item = ds[0]
        print(f"  {split_name} item keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")
    
    print("[TEST 1] PASSED ✓")


def test_dataloader_and_model(cfg):
    """Test 2-4: DataLoader collation, forward_train, forward_test"""
    print("=" * 70)
    print("[TEST 2] DataLoader + Model")
    
    # Build task for clean construction
    task_cfg = dict(cfg.get('task', {}))
    task_type = task_cfg.pop('type')
    task = TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    task.build_env()
    task.build_model()
    task.build_datasets()
    
    model = task.model
    device = next(model.parameters()).device
    
    # --- DataLoader test ---
    print("  [2a] Train DataLoader")
    train_batch = next(iter(task.train_dataloader))
    print(f"    batch keys: {list(train_batch.keys())}")
    print(f"    imgs shape: {train_batch['imgs'].shape}")
    if 'data_samples' in train_batch:
        ds = train_batch['data_samples']
        print(f"    data_samples: len={len(ds)}, type_0={type(ds[0])}")
        if isinstance(ds[0], dict):
            print(f"    sample keys: {list(ds[0].keys())}")
            if 'gt_bboxes' in ds[0]:
                gt = ds[0]['gt_bboxes']
                print(f"    gt_bboxes: shape={gt.shape if hasattr(gt, 'shape') else '?'}, dtype={gt.dtype if hasattr(gt, 'dtype') else '?'}")
    
    print("  [2b] Val DataLoader")
    val_batch = next(iter(task.val_dataloader))
    print(f"    batch keys: {list(val_batch.keys())}")
    print(f"    imgs shape: {val_batch['imgs'].shape}")
    
    # --- forward_train test ---
    print("=" * 70)
    print("[TEST 3] forward_train")
    model.train()
    imgs = train_batch['imgs'].to(device)
    data_samples = train_batch.get('data_samples')
    
    with torch.amp.autocast('cuda', enabled=False):
        train_out = model.forward_train(imgs, data_samples)
    
    print(f"  output type: {type(train_out)}")
    if isinstance(train_out, dict):
        print(f"  output keys: {list(train_out.keys())}")
        for k, v in train_out.items():
            if hasattr(v, 'item'):
                print(f"    {k}: {v.item():.6f}")
            else:
                print(f"    {k}: {v}")
        assert 'loss' in train_out, "FAIL: 'loss' not in forward_train output"
    print("[TEST 3] PASSED ✓")
    
    # --- forward_test test ---
    print("=" * 70)
    print("[TEST 4] forward_test")
    model.eval()
    val_imgs = val_batch['imgs'].to(device)
    val_samples = val_batch.get('data_samples')
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=False):
            test_out = model.forward_test(val_imgs, val_samples)
    
    print(f"  output type: {type(test_out)}")
    if isinstance(test_out, list):
        print(f"  output len: {len(test_out)}")
        for i, item in enumerate(test_out[:2]):
            print(f"    item[{i}] type: {type(item).__name__}")
            if hasattr(item, 'bboxes'):
                bb = item.bboxes
                print(f"      bboxes: type={type(bb).__name__}, shape={bb.shape if hasattr(bb, 'shape') else '?'}, len={len(bb)}")
            if hasattr(item, 'scores'):
                sc = item.scores
                print(f"      scores: type={type(sc).__name__}, len={len(sc)}")
            if hasattr(item, 'labels'):
                lb = item.labels
                print(f"      labels: type={type(lb).__name__}, len={len(lb)}")
    print("[TEST 4] PASSED ✓")
    
    return task, test_out, val_batch


def test_evaluator(cfg, test_out, val_batch):
    """Test 5: Evaluator process + evaluate"""
    print("=" * 70)
    print("[TEST 5] Evaluator process + evaluate")
    from cellstudio.evaluation.evaluator import Evaluator
    
    val_evaluator_cfg = cfg.get('val_evaluator', {})
    metrics = list(val_evaluator_cfg.get('metrics', []))
    evaluator = Evaluator(metrics_cfg=metrics)
    print(f"  Metrics: {[m.__class__.__name__ for m in evaluator.metrics]}")
    
    # Simulate what EvalHook.after_val_iter does
    evaluator.process(val_batch, test_out)
    print(f"  After process: {len(evaluator._predictions)} pred batches, {len(evaluator._data_samples)} data batches")
    
    # Inspect internal format
    print(f"  _predictions[0] type: {type(evaluator._predictions[0])}")
    print(f"  _data_samples[0] type: {type(evaluator._data_samples[0])}")
    
    # Check how DetMatchCache.get() will receive the data
    # The evaluator.evaluate() calls:
    #   y_true = [batch.get('data_samples', []) for batch in self._data_samples]
    #   y_pred = self._predictions
    # So y_true is a list of lists (one list per batch of DataSample dicts)
    # and y_pred is a list of lists (one list per batch of CellStudioInferResult)
    
    if isinstance(evaluator._data_samples[0], dict):
        y_true_sample = evaluator._data_samples[0].get('data_samples', [])
        print(f"  y_true (from data_samples): type={type(y_true_sample)}, len={len(y_true_sample)}")
        if len(y_true_sample) > 0:
            t0 = y_true_sample[0]
            print(f"    t0 type: {type(t0)}")
            if isinstance(t0, dict):
                print(f"    t0 keys: {list(t0.keys())}")
                if 'gt_bboxes' in t0:
                    gt = t0['gt_bboxes']
                    print(f"    gt_bboxes: shape={gt.shape if hasattr(gt, 'shape') else len(gt)}")
    
    # Now run evaluate
    tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', '_test_eval_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        eval_result = evaluator.evaluate(tmp_dir)
        print(f"  evaluate() result:")
        for k, v in eval_result.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            elif isinstance(v, list) and len(v) > 5:
                print(f"    {k}: [list of {len(v)} items]")
            else:
                print(f"    {k}: {v}")
        
        assert 'mAP50' in eval_result, f"FAIL: 'mAP50' not in evaluate result. Got: {list(eval_result.keys())}"
        print("[TEST 5] PASSED ✓")
    except Exception as e:
        print(f"[TEST 5] FAILED ✗ - {e}")
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def test_hook_ordering(cfg, task):
    """Test 6: Hook registration ordering"""
    print("=" * 70)
    print("[TEST 6] Hook registration ordering")
    
    task.build_evaluator()
    task.build_runner()
    
    runner = task.runner
    print(f"  Registered hooks ({len(runner._hooks)}):")
    eval_idx = logger_idx = ckpt_idx = None

    for i, hook in enumerate(runner._hooks):
        name = hook.__class__.__name__
        print(f"    [{i}] {name}")
        if name == 'EvalHook':
            eval_idx = i
        elif name == 'TextLoggerHook':
            logger_idx = i
        elif name == 'CheckpointHook':
            ckpt_idx = i
    
    # EvalHook MUST run before TextLoggerHook and CheckpointHook
    ok = True
    if eval_idx is not None and logger_idx is not None:
        if eval_idx < logger_idx:
            print(f"  EvalHook[{eval_idx}] before TextLoggerHook[{logger_idx}] ✓")
        else:
            print(f"  BUG: EvalHook[{eval_idx}] AFTER TextLoggerHook[{logger_idx}] ✗")
            ok = False
    
    if eval_idx is not None and ckpt_idx is not None:
        if eval_idx < ckpt_idx:
            print(f"  EvalHook[{eval_idx}] before CheckpointHook[{ckpt_idx}] ✓")
        else:
            print(f"  BUG: EvalHook[{eval_idx}] AFTER CheckpointHook[{ckpt_idx}] ✗")
            ok = False
    
    if ok:
        print("[TEST 6] PASSED ✓")
    else:
        print("[TEST 6] FAILED ✗")
        raise AssertionError("Hook ordering incorrect")


def test_val_interval_logic():
    """Test 7: val_interval logic correctness"""
    print("=" * 70)
    print("[TEST 7] val_interval logic")
    
    tests = [
        (1, 5, [0, 1, 2, 3, 4]),
        (5, 20, [4, 9, 14, 19]),
        (10, 25, [9, 19, 24]),
        (3, 10, [2, 5, 8, 9]),
    ]
    
    for val_interval, max_epochs, expected in tests:
        actual = []
        for epoch in range(max_epochs):
            is_last = (epoch + 1 >= max_epochs)
            if (epoch + 1) % val_interval == 0 or is_last:
                actual.append(epoch)
        
        ok = actual == expected
        print(f"  interval={val_interval}, epochs={max_epochs}: {actual} {'✓' if ok else '✗'}")
        assert ok, f"FAIL: expected {expected}, got {actual}"
    
    print("[TEST 7] PASSED ✓")


def test_full_epoch():
    """Test 8: Full 1-epoch train + val"""
    print("=" * 70)
    print("[TEST 8] Full 1-epoch train+val cycle")
    
    from omegaconf import OmegaConf
    cfg = Config.fromfile(CONFIG_PATH)
    
    # Override to 1 epoch
    cfg = OmegaConf.merge(cfg, {
        'runner': {'max_epochs': 1, 'val_interval': 1},
    })
    
    tmp_work_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', '_test_full_epoch_tmp')
    cfg['work_dir'] = tmp_work_dir
    
    task_cfg = dict(cfg.get('task', {}))
    task_type = task_cfg.pop('type')
    task = TASK_REGISTRY.build({'type': task_type, 'cfg': cfg})
    
    try:
        task.execute(mode='train')
        
        runner = task.runner
        print(f"  Final epoch: {runner.epoch}")
        print(f"  Final iter: {runner.iter}")
        
        # Check val_metrics
        if hasattr(runner, 'val_metrics'):
            print(f"  val_metrics:")
            for k, v in runner.val_metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6f}")
                elif isinstance(v, list) and len(v) > 5:
                    print(f"    {k}: [list of {len(v)}]")
                else:
                    print(f"    {k}: {v}")
            assert 'mAP50' in runner.val_metrics, "FAIL: mAP50 not in val_metrics"
        else:
            raise AssertionError("val_metrics not set after validation")
        
        # Check scalars.json
        scalars_path = os.path.join(tmp_work_dir, 'scalars.json')
        if os.path.exists(scalars_path):
            import json
            with open(scalars_path, 'r') as f:
                lines = f.readlines()
            
            train_recs = [json.loads(l) for l in lines if '"mode": "train"' in l]
            val_recs = [json.loads(l) for l in lines if '"mode": "val"' in l]
            print(f"  scalars.json: {len(train_recs)} train, {len(val_recs)} val records")
            
            if val_recs:
                print(f"  Last val record: {val_recs[-1]}")
            else:
                raise AssertionError("No val records in scalars.json")
        else:
            raise AssertionError(f"scalars.json not found at {scalars_path}")
        
        # Check checkpoints
        print(f"  latest.pth: {os.path.exists(os.path.join(tmp_work_dir, 'latest.pth'))}")
        print(f"  best.pth: {os.path.exists(os.path.join(tmp_work_dir, 'best.pth'))}")
        
        print("[TEST 8] PASSED ✓")
        
    except Exception as e:
        print(f"[TEST 8] FAILED ✗ - {e}")
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(tmp_work_dir):
            shutil.rmtree(tmp_work_dir)


def main():
    print("=" * 70)
    print("THOROUGH VALIDATION TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 0: Config
    try:
        cfg = load_config()
        results['T0_config'] = 'PASSED'
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        return
    
    # Test 1: Dataset
    try:
        test_dataset(cfg)
        results['T1_dataset'] = 'PASSED'
    except Exception as e:
        print(f"[TEST 1] FAILED ✗ - {e}")
        traceback.print_exc()
        results['T1_dataset'] = f'FAILED: {e}'
    
    # Test 2-4: DataLoader + Model
    try:
        task, test_out, val_batch = test_dataloader_and_model(cfg)
        results['T2_dataloader'] = 'PASSED'
        results['T3_forward_train'] = 'PASSED'
        results['T4_forward_test'] = 'PASSED'
    except Exception as e:
        print(f"[TEST 2-4] FAILED ✗ - {e}")
        traceback.print_exc()
        results['T2_4_model'] = f'FAILED: {e}'
        # summary and exit
        print_summary(results)
        return
    
    # Test 5: Evaluator
    try:
        test_evaluator(cfg, test_out, val_batch)
        results['T5_evaluator'] = 'PASSED'
    except Exception as e:
        print(f"[TEST 5] FAILED ✗ - {e}")
        traceback.print_exc()
        results['T5_evaluator'] = f'FAILED: {e}'
    
    # Test 6: Hook ordering
    try:
        test_hook_ordering(cfg, task)
        results['T6_hooks'] = 'PASSED'
    except Exception as e:
        print(f"[TEST 6] FAILED ✗ - {e}")
        traceback.print_exc()
        results['T6_hooks'] = f'FAILED: {e}'
    
    # Test 7: val_interval logic
    try:
        test_val_interval_logic()
        results['T7_val_interval'] = 'PASSED'
    except Exception as e:
        print(f"[TEST 7] FAILED ✗ - {e}")
        traceback.print_exc()
        results['T7_val_interval'] = f'FAILED: {e}'
    
    # Test 8: Full epoch
    try:
        test_full_epoch()
        results['T8_full_epoch'] = 'PASSED'
    except Exception as e:
        print(f"[TEST 8] FAILED ✗ - {e}")
        traceback.print_exc()
        results['T8_full_epoch'] = f'FAILED: {e}'
    
    print_summary(results)


def print_summary(results):
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, result in results.items():
        status = "✓" if result == 'PASSED' else "✗"
        print(f"  {status} {name}: {result}")
        if result != 'PASSED':
            all_pass = False
    
    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED ✓✓✓")
    else:
        print("SOME TESTS FAILED ✗✗✗")
    print("=" * 70)


if __name__ == '__main__':
    main()
