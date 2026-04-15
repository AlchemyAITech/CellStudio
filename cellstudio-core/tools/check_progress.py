import os, re, json

models = ['yolo_v8m_det_mido_tile','yolo_26m_det_mido_tile','faster_rcnn_mido_tile','detr_mido_tile','fcos_mido_tile','rtmdet_mido_tile']
out = []
for m in models:
    lf = os.path.join('work_dirs', m, 'training.log')
    if not os.path.exists(lf):
        out.append(f'{m}: NOT STARTED')
        continue
    lines = open(lf).readlines()
    max_ep, total_ep, last_loss = -1, -1, None
    for l in lines:
        match = re.search(r'Epoch \[(\d+)/(\d+)\]', l)
        if match:
            ep = int(match.group(1))
            te = int(match.group(2))
            if ep > max_ep: max_ep = ep; total_ep = te
        lm = re.search(r'loss: ([\d.]+)', l)
        if lm: last_loss = float(lm.group(1))
    # Check scalars for mAP
    sf = os.path.join('work_dirs', m, 'scalars.json')
    best_map = 0
    n_val = 0
    if os.path.exists(sf):
        for sl in open(sf):
            try:
                d = json.loads(sl.strip())
                v = d.get('val_mAP50', d.get('mAP50', 0))
                if v is not None and v > 0:
                    n_val += 1
                    if v > best_map: best_map = v
            except: pass
    out.append(f'{m}: epoch {max_ep}/{total_ep} | loss={last_loss} | best_mAP={best_map:.4f} | val_runs={n_val}')

with open('work_dirs/progress.txt', 'w') as f:
    f.write('\n'.join(out))
print('Done')
