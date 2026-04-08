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

__version__ = "0.2.0"
