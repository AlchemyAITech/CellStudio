$models = @("timm_resnet18_mido", "timm_resnet50_mido", "timm_efficientnet_b3_mido", "timm_mobilenetv3_mido", "yolo_v8m_cls_mido", "yolo_26m_cls_mido")
New-Item -ItemType Directory -Force -Path "work_dirs" | Out-Null

foreach ($m in $models) {
    Write-Host "================ TRAINING $m ================" -ForegroundColor Cyan
    python tools/train.py "configs/classify/$m.yaml" > "work_dirs/${m}_train.log" 2>&1
    
    Write-Host "================ BENCHMARKING FLOPs $m ================" -ForegroundColor Yellow
    python tools/benchmark_flops.py "configs/classify/$m.yaml" > "work_dirs/${m}_flops.log" 2>&1
    
    Write-Host "================ BENCHMARKING FPS $m ================" -ForegroundColor Green
    python tools/benchmark_fps.py "configs/classify/$m.yaml" > "work_dirs/${m}_fps.log" 2>&1
}

Write-Host "AGGREGATING ALL LOGS AND GENERATING SUMMARY..." -ForegroundColor Magenta
python tools/aggregate_results.py
Write-Host "PIPELINE COMPLETED! Please check work_dirs/final_summary.md"
