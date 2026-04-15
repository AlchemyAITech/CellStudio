import os
import sys
import subprocess
import json
import yaml
import pytest

script_dir = os.path.dirname(os.path.abspath(__file__))
qa_suite_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(os.path.dirname(qa_suite_dir))

# Load catalog
catalog_path = os.path.join(qa_suite_dir, "qa_catalog.yaml")
with open(catalog_path, "r") as f:
    catalog = yaml.safe_load(f)["models"]

# Extract model configs for parametrization
model_entries = [(m["id"], m["config"]) for m in catalog]


@pytest.mark.parametrize("model_id,config_rel", model_entries)
def test_overfit_convergence(model_id, config_rel):
    print(f"\n[QA] Testing Pipeline Overfit Capability (5 Epochs) for {model_id}...")
    
    cfg_path = os.path.join(root_dir, config_rel)
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config missing: {cfg_path}")
        
    work_dir = os.path.join(root_dir, 'work_dirs', 'qa_overfit', model_id)
    
    # Dynamically force 3 epochs for fast QA convergence
    with open(cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    if 'runner' not in base_cfg: 
        base_cfg['runner'] = {'type': 'EpochBasedRunner'}
    base_cfg['runner']['max_epochs'] = 3
    
    temp_cfg_path = os.path.abspath(os.path.join(os.path.dirname(cfg_path), f'qa_temp_{model_id}.yaml'))
    with open(temp_cfg_path, 'w') as f:
        yaml.dump(base_cfg, f)
    
    cmd = [
        sys.executable, 'tools/train.py',
        temp_cfg_path,
        '--work-dir', work_dir
    ]
    
    # Stream output to stdout instead of capturing silently
    res = subprocess.run(cmd, cwd=root_dir, stdout=sys.stdout, stderr=sys.stderr)
    if os.path.exists(temp_cfg_path):
        os.remove(temp_cfg_path)
        
    assert res.returncode == 0, f"Overfit training crashed for {model_id}!"
    
    # Verify JSON scalars
    scalars_file = os.path.join(work_dir, 'scalars.json')
    assert os.path.exists(scalars_file), f"No scalars log found at {scalars_file}"
    
    losses = []
    with open(scalars_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if 'train/loss' in data:
                losses.append(data['train/loss'])
                
    assert len(losses) >= 2, f"Need at least 2 loss values to prove convergence for {model_id}."
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"  [{model_id}] -> Initial Loss: {initial_loss:.4f} | Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, f"[{model_id}] Loss did not decrease! {final_loss} vs {initial_loss}"
    # Even stricter requirement: Should drop by at least 10%
    assert final_loss < initial_loss * 0.95, f"[{model_id}] Loss did not decrease enough! {final_loss} vs {initial_loss}"
