import yaml  # noqa: F401
import glob
import os  # noqa: F401

configs = glob.glob('e:/workspace/AlchemyTech/CellStudio/configs/segmentation/*.yaml')

for cfg_path in configs:
    with open(cfg_path, 'r') as f:
        content = f.read()
        
    if '_base_' not in content:
        # Prepend the root runtime configuration
        base_injection = "_base_:\n  - ../_base_/default_runtime.yaml\n\n"
        content = base_injection + content
        
        with open(cfg_path, 'w') as f:
            f.write(content)
        print(f"Injected Zenith Runtime Hooks into {cfg_path}")
    else:
        print(f"Skipping {cfg_path}, already has hooks.")
