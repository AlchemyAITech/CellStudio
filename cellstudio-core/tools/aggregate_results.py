import os
import re

models = [
    "timm_resnet18_mido", "timm_resnet50_mido", "timm_efficientnet_b3_mido", 
    "timm_mobilenetv3_mido", "yolo_v8m_cls_mido", "yolo_11m_cls_mido"
]

results = {}

for m in models:
    results[m] = {
        'Arch': m, 'Accuracy': 'N/A', 'F1': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A',
        'FLOPs': 'N/A', 'Params': 'N/A', 'FPS': 'N/A'
    }
    
    # Parse Train Log for max metrics
    train_log = f"work_dirs/{m}_train.log"
    if os.path.exists(train_log):
        with open(train_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Accuracy:' in line and 'F1_Score:' in line:
                    try:
                        acc = float(re.search(r'Accuracy:\s*([0-9.]+)', line).group(1))
                        f1 = float(re.search(r'F1_Score:\s*([0-9.]+)', line).group(1))
                        prec = float(re.search(r'Precision:\s*([0-9.]+)', line).group(1))
                        rec = float(re.search(r'Recall:\s*([0-9.]+)', line).group(1))
                        
                        if results[m]['Accuracy'] == 'N/A' or acc > results[m]['Accuracy']:
                            results[m]['Accuracy'] = acc
                            results[m]['F1'] = f1
                            results[m]['Precision'] = prec
                            results[m]['Recall'] = rec
                    except: pass
                    
    # Parse FLOPs Log
    flops_log = f"work_dirs/{m}_flops.log"
    if os.path.exists(flops_log):
        with open(flops_log, 'r', encoding='utf-8') as f:
            content = f.read()
            flop_match = re.search(r'FLOPs:\s*([A-Za-z0-9.]+)', content)
            param_match = re.search(r'Params:\s*([A-Za-z0-9.]+)', content)
            arch_match = re.search(r'Model:\s*([A-Za-z0-9._-]+)', content)
            if flop_match: results[m]['FLOPs'] = flop_match.group(1)
            if param_match: results[m]['Params'] = param_match.group(1)
            if arch_match: results[m]['Arch'] = arch_match.group(1)
            
    # Parse FPS Log
    fps_log = f"work_dirs/{m}_fps.log"
    if os.path.exists(fps_log):
        with open(fps_log, 'r', encoding='utf-8') as f:
            content = f.read()
            fps_match = re.search(r'Throughput\s*\(FPS\):\s*([0-9.]+)', content)
            if fps_match: results[m]['FPS'] = fps_match.group(1)

# Write Summary
lines = [
    "# Phase 5 MIDOG 六大主流模型对比实验基准测试摘要",
    "",
    "| Architecture | Acc | F1 | Precision | Recall | Params | FLOPs | GPU Inference FPS |",
    "|--------------|-----|----|-----------|--------|--------|-------|-------------------|"
]

for m in models:
    d = results[m]
    lines.append(f"| `{d['Arch']}` | {d['Accuracy']} | {d['F1']} | {d['Precision']} | {d['Recall']} | {d['Params']} | {d['FLOPs']} | {d['FPS']} |")

summary_path = "work_dirs/final_summary.md"
os.makedirs("work_dirs", exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
    
print(f"Summary generated at {summary_path}")
