"""
Edge Deployment FastAPI Server

Serves the ONNX INT8 model via a lightweight REST API.
Designed for resource-constrained environments:
- No GPU required
- Minimal memory footprint
- Fast cold start
- Single file deployment
"""

import numpy as np
import onnxruntime as rt
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

app = FastAPI(
    title="Edge Risk Classifier API",
    description=(
        "Lightweight ONNX INT8 risk classification for edge deployment. "
        "No GPU required. Designed for resource-constrained environments."
    ),
    version="1.0.0"
)

# Load model once at startup
MODEL_PATH = "models/risk_classifier_int8.onnx"
METADATA_PATH = "models/model_metadata.json"

_session = None
_metadata = None
_input_name = None
_output_name = None


def load_model():
    global _session, _metadata, _input_name, _output_name

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run train_and_export.py first."
        )

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.graph_optimization_level = (
        rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    _session = rt.InferenceSession(MODEL_PATH, sess_options=sess_options)
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name

    with open(METADATA_PATH) as f:
        _metadata = json.load(f)

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Classes: {_metadata['classes']}")
    print(f"Features: {_metadata['n_features']}")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ONNX INT8 RandomForest",
        "deployment": "edge",
        "n_features": _metadata["n_features"] if _metadata else 0,
        "classes": _metadata["classes"] if _metadata else [],
        "model_size_mb": _metadata["model_sizes_mb"]["onnx_int8"]
                         if _metadata else 0,
        "accuracy": _metadata["accuracy"] if _metadata else 0
    }


class PredictionRequest(BaseModel):
    features: List[float]
    return_probabilities: Optional[bool] = False


class BatchPredictionRequest(BaseModel):
    samples: List[List[float]]
    return_probabilities: Optional[bool] = False


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predict risk level for a single incident.
    Designed for real-time edge inference with minimal latency.
    """
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n_expected = _metadata["n_features"]
    if len(request.features) != n_expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {n_expected} features, got {len(request.features)}"
        )

    start = time.perf_counter()

    X = np.array(request.features, dtype=np.float32).reshape(1, -1)
    outputs = _session.run(None, {_input_name: X})
    pred_label_idx = int(outputs[0][0])
    pred_label = _metadata["classes"][pred_label_idx]

    latency_ms = (time.perf_counter() - start) * 1000

    result = {
        "risk_level": pred_label,
        "risk_index": pred_label_idx,
        "latency_ms": round(latency_ms, 3),
        "model": "ONNX INT8",
        "deployment": "edge"
    }

    # Probabilities if requested and available
    if request.return_probabilities and len(outputs) > 1:
        proba = outputs[1]
        if hasattr(proba, 'values'):
            proba_list = list(proba.values())
        else:
            proba_list = proba[0].tolist() if len(proba.shape) > 1 else proba.tolist()
        result["probabilities"] = {
            cls: round(float(p), 4)
            for cls, p in zip(_metadata["classes"], proba_list)
        }

    return result


@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple incidents.
    More efficient than calling /predict repeatedly.
    """
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n_expected = _metadata["n_features"]
    for i, sample in enumerate(request.samples):
        if len(sample) != n_expected:
            raise HTTPException(
                status_code=400,
                detail=f"Sample {i}: expected {n_expected} features, "
                       f"got {len(sample)}"
            )

    start = time.perf_counter()
    X = np.array(request.samples, dtype=np.float32)
    outputs = _session.run(None, {_input_name: X})
    preds = outputs[0].tolist()
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "predictions": [_metadata["classes"][int(p)] for p in preds],
        "n_samples": len(preds),
        "latency_ms": round(latency_ms, 3),
        "throughput_per_sec": round(len(preds) / (latency_ms / 1000), 0),
        "model": "ONNX INT8",
        "deployment": "edge"
    }


@app.get("/model-info")
def model_info():
    """Return full model metadata including benchmark results."""
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = dict(_metadata)

    benchmark_path = "benchmarks/results.json"
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            result["benchmark_results"] = json.load(f)

    return result


@app.get("/feature-names")
def feature_names():
    """Return expected feature names and order."""
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "features": _metadata["feature_names"],
        "n_features": _metadata["n_features"],
        "classes": _metadata["classes"]
    }


if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=8004, reload=False)