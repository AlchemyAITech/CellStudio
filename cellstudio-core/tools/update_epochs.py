import re
import glob

configs = glob.glob('e:/workspace/AlchemyTech/CellStudio/configs/segmentation/*.yaml')

for cfg_path in configs:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = re.sub(r'max_epochs:\s*\d+', 'max_epochs: 100', content)
    content = re.sub(r'T_max:\s*\d+', 'T_max: 100', content)
    
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Updated {cfg_path} to 100 epochs")
