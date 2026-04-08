import json, os

models = [
    'yolo_v8m_det_mido_tile',
    'yolo_26m_det_mido_tile',
    'faster_rcnn_mido_tile',
    'detr_mido_tile',
    'fcos_mido_tile',
    'rtmdet_mido_tile',
]

print(f"{'Model':<35} {'Loss':>8} {'mAP50':>8} {'Recall':>8} {'Preds':>8} {'ScoreMax':>8} {'ParamΔ':>10}")
print("-" * 95)

for m in models:
    p = os.path.join('work_dirs', f'debug_mini_{m}', 'history.json')
    if os.path.exists(p):
        d = json.load(open(p))
        loss = d.get('loss', [-1])[-1]
        mAP = d.get('mAP50', [0])[-1]
        rec = d.get('Recall', [0])[-1]
        pred = d.get('pred_count', [0])[-1]
        smax = d.get('score_max', [0])[-1]
        pd = d.get('param_delta', [0])[-1]
        print(f"{m:<35} {loss:8.4f} {mAP:8.4f} {rec:8.4f} {pred:8.0f} {smax:8.4f} {pd:10.6f}")
    else:
        print(f"{m:<35} NOT FOUND")
