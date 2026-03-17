import argparse
import os
import cv2
import json
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Pure ONNXRuntime Deployment Tool for High-Performance Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to exported .onnx model.")
    parser.add_argument("--task", type=str, required=True, choices=['classification'], help="Task type. (YOLO detection/segmentation deployment SDK coming in v2)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialize Inferencer
    if args.task == "classification":
        from pathstudio.engine.inferencer import ClassificationONNXInferencer
        print(f"[DEPLOY] Loading {args.model} on {args.device}...")
        t_init = time.perf_counter()
        inferencer = ClassificationONNXInferencer(args.model, args.device)
        print(f"[DEPLOY] Model loaded in {(time.perf_counter() - t_init)*1000:.2f} ms")
    else:
        raise NotImplementedError(f"Task {args.task} specific inferencer not specified.")
        
    # 2. Read Image
    img = cv2.imread(args.source)
    if img is None:
        raise ValueError(f"Could not read image source at: {args.source}")
        
    # 3. Predict & Profile
    print(f"[DEPLOY] Starting inference pipeline...")
    result = inferencer.predict(img)
    
    print("\n[DEPLOY] Inference Result:")
    print(json.dumps(result, indent=2))
    
    # Report performance metrics
    latency = result['latency_ms']
    print("\n[PERFORMANCE REPORT]")
    print(f"  Pre-process:  {latency['preprocess']:7.3f} ms")
    print(f"  Inference:    {latency['inference']:7.3f} ms")
    print(f"  Post-process: {latency['postprocess']:7.3f} ms")
    print(f"  Total Latency:{latency['total']:7.3f} ms")
    print(f"  Max FPS:      {(1000.0 / latency['total']):7.1f} FPS")

if __name__ == "__main__":
    main()
