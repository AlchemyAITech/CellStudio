"""CellStudio FastAPI inference and training service.

Endpoints:
    POST /predict/{model_id}  — Run inference on an uploaded image.
    POST /train               — Dispatch an async training job.
    POST /evaluate            — Dispatch an async evaluation job.
    GET  /status/{job_id}     — Poll job status and tail logs.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import uuid

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Ensure CellStudio is importable when running standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cellstudio.inference.inferencer import CellStudioInferencer  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CellStudio MLOps API",
    description="Train, evaluate, and infer with CellStudio models via REST.",
    version="2.1.0",
)

# In-memory stores — production should use Redis / DB
INFERENCERS: dict[str, CellStudioInferencer] = {}
JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    """Request body for train and evaluate endpoints."""

    config_yaml: str
    checkpoint_path: str | None = None


# ---------------------------------------------------------------------------
# Startup — pre-load models into GPU VRAM
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Load models declared in ``api/config.yaml`` into GPU memory."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        logger.warning("No api/config.yaml found; skipping model pre-load.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    models = cfg.get('models', {})
    logger.info("========== CellStudio API BOOT ==========")
    for model_id, m_cfg in models.items():
        logger.info("Warming cache: '%s' ...", model_id)
        abs_conf = os.path.join(os.path.dirname(__file__), m_cfg['config'])
        abs_ckpt = os.path.join(os.path.dirname(__file__), m_cfg['checkpoint'])
        try:
            inst = CellStudioInferencer(
                abs_conf, abs_ckpt, m_cfg.get('device', 'cuda'),
            )
            INFERENCERS[model_id] = inst
            logger.info("  [OK] '%s' ready.", model_id)
        except Exception:
            logger.exception("  [FAILED] '%s' init failed.", model_id)
    logger.info("==========================================")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@app.post("/predict/{model_id}")
async def predict(model_id: str, file: UploadFile = File(...)):
    """Run inference using a pre-loaded model.

    Args:
        model_id: Key matching a model in ``api/config.yaml``.
        file: Uploaded image file.

    Returns:
        JSON with ``status``, ``model``, and ``predictions`` fields.
    """
    if model_id not in INFERENCERS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not loaded. Available: {list(INFERENCERS)}",
        )
    try:
        ext = os.path.splitext(file.filename or '.png')[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = INFERENCERS[model_id](tmp_path)
        os.remove(tmp_path)
        return {"status": "success", "model": model_id, "predictions": result}
    except Exception as e:
        logger.exception("Prediction failed for model '%s'.", model_id)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.post("/train")
async def train(req: JobRequest):
    """Spawn an asynchronous training job via ``tools/cli.py train``.

    Args:
        req: Request body containing raw YAML config text.

    Returns:
        JSON with ``job_id`` and ``status``.
    """
    job_id = str(uuid.uuid4())
    log_file = os.path.join('work_dirs', f'job_{job_id}.log')
    tmp_cfg_path = os.path.join(tempfile.gettempdir(), f'cfg_{job_id}.yaml')

    with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
        f.write(req.config_yaml)

    cmd = [sys.executable, "-u", "tools/cli.py", "train", tmp_cfg_path]
    os.makedirs('work_dirs', exist_ok=True)
    with open(log_file, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)

    JOBS[job_id] = {'process': proc, 'log_file': log_file, 'type': 'train'}
    return {"status": "dispatched", "job_id": job_id, "config_file": tmp_cfg_path}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.post("/evaluate")
async def evaluate(req: JobRequest):
    """Spawn an asynchronous evaluation job.

    Args:
        req: Request body with YAML config and ``checkpoint_path``.

    Returns:
        JSON with ``job_id`` and ``status``.
    """
    if not req.checkpoint_path:
        raise HTTPException(
            status_code=400,
            detail="'checkpoint_path' is required for evaluation.",
        )

    job_id = str(uuid.uuid4())
    log_file = os.path.join('work_dirs', f'job_{job_id}.log')
    tmp_cfg_path = os.path.join(tempfile.gettempdir(), f'cfg_{job_id}.yaml')

    with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
        f.write(req.config_yaml)

    cmd = [
        sys.executable, "-u", "tools/cli.py",
        "eval", tmp_cfg_path, req.checkpoint_path,
    ]
    os.makedirs('work_dirs', exist_ok=True)
    with open(log_file, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)

    JOBS[job_id] = {'process': proc, 'log_file': log_file, 'type': 'eval'}
    return {"status": "dispatched", "job_id": job_id}


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll job status and tail the last 50 lines of output.

    Args:
        job_id: UUID returned by ``/train`` or ``/evaluate``.

    Returns:
        JSON with ``state`` (RUNNING / SUCCESS / FAILED), ``exit_code``,
        and ``tail_logs``.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Unknown job ID.")

    job = JOBS[job_id]
    proc = job['process']
    ret = proc.poll()

    logs = ""
    if os.path.exists(job['log_file']):
        with open(job['log_file'], 'r') as f:
            lines = f.readlines()
            logs = "".join(lines[-50:])

    if ret is None:
        state = "RUNNING"
    elif ret == 0:
        state = "SUCCESS"
    else:
        state = "FAILED"

    return {"job_id": job_id, "state": state, "exit_code": ret, "tail_logs": logs}


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    s_cfg = cfg.get('server', {})
    uvicorn.run(
        "main:app",
        host=s_cfg.get('host', '0.0.0.0'),
        port=s_cfg.get('port', 18080),
        reload=False,
    )
