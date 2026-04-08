import os
import json

models = [
    "yolo_v8m_det_mido_tile",
    "yolo_26m_det_mido_tile",
    "faster_rcnn_mido_tile",
    "detr_mido_tile",
    "fcos_mido_tile",
    "rtmdet_mido_tile"
]

print("| Model Architecture | Best mAP@50 | Recall | Precision | F1-Score | Best Epoch |")
print("|--------------------|-------------|--------|-----------|----------|------------|")

for m in models:
    scalar_file = os.path.join("work_dirs", m, "scalars.json")
    
    best_map = 0.0
    best_epoch = -1
    best_recall = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    
    if os.path.exists(scalar_file):
        with open(scalar_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'mAP50' in data:
                        map_val = data['mAP50']
                        if map_val > best_map:
                            best_map = map_val
                            best_epoch = data.get('epoch', -1)
                            best_recall = data.get('Recall', 0.0)
                            best_precision = data.get('Precision', 0.0)
                            best_f1 = data.get('F1', 0.0)
                except Exception:
                    pass
    
    # Optional fallback if no scalars.json but we have training.log
    if best_map == 0.0:
        log_file = os.path.join("work_dirs", m, "training.log")
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if 'mAP50:' in line:
                        try:
                            # Parse line like "... mAP50: 0.6553, ..."
                            import re
                            map_m = re.search(r'mAP50:\s*([0-9.]+)', line)
                            ep_m = re.search(r'Val Epoch \[(\d+)/\d+\]', line)
                            r_m = re.search(r'Recall:\s*([0-9.]+)', line)
                            p_m = re.search(r'Precision:\s*([0-9.]+)', line)
                            f1_m = re.search(r'F1:\s*([0-9.]+)', line)
                            
                            if map_m:
                                map_val = float(map_m.group(1))
                                if map_val > best_map:
                                    best_map = map_val
                                    best_epoch = int(ep_m.group(1)) if ep_m else best_epoch
                                    best_recall = float(r_m.group(1)) if r_m else best_recall
                                    best_precision = float(p_m.group(1)) if p_m else best_precision
                                    best_f1 = float(f1_m.group(1)) if f1_m else best_f1
                        except Exception:
                            pass
                            
    print(f"| {m.replace('_mido_tile', '').replace('_det', '')} | {best_map:.4f} | {best_recall:.4f} | {best_precision:.4f} | {best_f1:.4f} | {best_epoch} |")
