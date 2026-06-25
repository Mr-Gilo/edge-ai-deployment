# Edge AI Deployment with ONNX

Demonstrates training a machine learning model and deploying it
for resource-constrained edge environments using ONNX format
and INT8 quantisation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ONNX](https://img.shields.io/badge/ONNX-1.16-orange)
![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-1.18-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)

## What is ONNX?

ONNX (Open Neural Network Exchange) is an open standard for ML models.
A model trained in scikit-learn, PyTorch, or TensorFlow can be exported
to ONNX and run anywhere — edge devices, browsers, mobile, embedded systems —
without the original training framework installed.

## What is INT8 Quantisation?

Quantisation reduces model weights from 32-bit floats (FP32) to
8-bit integers (INT8). Benefits:
- Smaller model file (typically 2-4x smaller)
- Lower memory usage at inference time
- Faster inference on CPU (integer arithmetic is faster)
- Minimal accuracy loss for most classification tasks

## Pipeline

Safety Incident Data (CSV)

↓

RandomForest Classifier
(scikit-learn training)

↓

Evaluation + Metrics

↓

ONNX FP32 Export
(skl2onnx conversion)

↓

ONNX INT8 Export
(dynamic quantisation)

↓

Benchmark: Latency + Memory + Throughput

↓

FastAPI Edge Serving
(no GPU, minimal memory)

## Dataset

Synthetic safety incident data: 2,000 samples, 14 features across
three modalities (sensor statistics, equipment metadata, event indicators).
Risk labels: low / medium / high / critical.

## Setup

```bash
git clone https://github.com/Mr-Gilo/edge-ai-deployment.git
cd edge-ai-deployment

conda create -n edge-ai python=3.11 -y
conda activate edge-ai
pip install -r requirements.txt
```

## Running

```bash
# Step 1: Generate data
python data_generator.py

# Step 2: Train and export to ONNX
python train_and_export.py

# Step 3: Benchmark all three model formats
python benchmark.py

# Step 4: Start edge serving API
python serve.py
```

API runs at http://127.0.0.1:8004
Swagger docs at http://127.0.0.1:8004/docs

## Benchmark Results

Measured on CPU (Windows, Intel), 200 inference runs, single sample latency:

| Model | P50 Latency | P95 Latency | Batch (600 samples) | Accuracy |
|-------|-------------|-------------|---------------------|----------|
| Sklearn PKL | 16.302ms | 16.967ms | 26.71ms | 0.780 |
| ONNX FP32 | 0.013ms | 0.014ms | 2.53ms | 0.780 |
| ONNX INT8 | 0.013ms | 0.014ms | 2.50ms | 0.780 |

**ONNX speedup: 1,254x faster than sklearn on single inference**

Accuracy is fully preserved across all three formats.

Note: For Random Forest models, INT8 quantisation preserves accuracy and
inference speed but does not reduce file size significantly, since tree
models use integer decision logic rather than floating point weight matrices.
For neural networks (LSTM, Transformer), INT8 quantisation typically achieves
2-4x file size reduction alongside the inference speedup.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Model status and metadata |
| POST | /predict | Single incident risk prediction |
| POST | /predict-batch | Batch predictions |
| GET | /model-info | Full metadata and benchmarks |
| GET | /feature-names | Expected input features |

## Example Request

```bash
curl -X POST http://127.0.0.1:8004/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [65.0, 8.5, 90.0, 35.0, 1.2, 8.0, 200, 120.0, 1, 0, 1, 0, 1, 0]
  }'
```

Response:
```json
{
  "risk_level": "high",
  "risk_index": 2,
  "latency_ms": 0.812,
  "model": "ONNX INT8",
  "deployment": "edge"
}
```

## Related Projects

- [pdf-extractor](https://github.com/Mr-Gilo/pdf-extractor)
- [multimodal-risk-pipeline](https://github.com/Mr-Gilo/multimodal-risk-pipeline)
- [document-extraction-finetuning](https://github.com/Mr-Gilo/document-extraction-finetuning)

## Roadmap

- [x] Synthetic safety incident dataset
- [x] RandomForest training and evaluation
- [x] ONNX FP32 export and verification
- [x] ONNX INT8 dynamic quantisation
- [x] Latency and throughput benchmarking
- [x] FastAPI edge serving endpoint
- [ ] PyTorch model export (TE-LSTM-AE from research)
- [ ] Browser deployment via ONNX.js
- [ ] Raspberry Pi deployment guide