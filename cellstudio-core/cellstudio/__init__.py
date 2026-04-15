"""CellStudio — Deep learning framework for pathological cell analysis.

Subpackages:
    core        Universal Registry and shared utilities.
    engine      Runner, Hooks, and Config system.
    tasks       Task orchestrators (Classification, Detection, Segmentation).
    datasets    Dataset abstractions and MIDO format adapters.
    models      Model adapter interfaces.
    backends    Third-party backend isolation (Ultralytics, timm, Cellpose).
    metrics     Evaluation metrics.
    plotting    Visualization and plotting.
    evaluation  Evaluator orchestration.
    inference   Inference engine.
    pipeline    Data transform DAG (Compose + transform nodes).
    structures  Standard data structures (DataSample, InferResult).
"""

import os

# ---------------------------------------------------------------------------------
# 🚀 PRETRAINED WEIGHTS INTERCEPTOR (Isolating Global Downloads)
# ---------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_weights")
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# [Passiveness Control]
# Enforce isolation: Prevent system-wide caching from polluting the OS
os.environ['TORCH_HOME'] = os.path.join(PRETRAINED_DIR, 'torch_hub')
os.environ['HF_HOME'] = os.path.join(PRETRAINED_DIR, 'huggingface')
os.environ['MIM_CACHE_DIR'] = os.path.join(PRETRAINED_DIR, 'mim')
os.environ['YOLO_CONFIG_DIR'] = os.path.join(PRETRAINED_DIR, 'yolo_cfg')

__version__ = "0.2.0"
