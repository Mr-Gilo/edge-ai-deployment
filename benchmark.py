"""
Inference Benchmarking: Sklearn vs ONNX FP32 vs ONNX INT8

Measures:
- Inference latency (single sample and batch)
- Memory footprint
- Throughput (samples per second)
- Accuracy preservation after quantisation
"""

import numpy as np
import pandas as pd
import time
import pickle
import json
import os
import psutil
import gc
from sklearn.metrics import accuracy_score
import onnxruntime as rt

os.makedirs("benchmarks", exist_ok=True)


def get_memory_mb():
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_test_data():
    df = pd.read_csv("data/safety_incidents.csv")
    feature_cols = [c for c in df.columns if c != "risk_label"]
    X = df[feature_cols].values.astype(np.float32)

    with open("models/sklearn_model.pkl", "rb") as f:
        saved = pickle.load(f)
    le = saved["label_encoder"]
    y = le.transform(df["risk_label"])

    # Use last 20% as test set (same split as training)
    split = int(len(X) * 0.8)
    return X[split:], y[split:], le


def benchmark_sklearn(X_test, y_test, n_warmup=10, n_runs=200):
    """Benchmark sklearn model inference."""
    with open("models/sklearn_model.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    # Warmup
    for _ in range(n_warmup):
        model.predict(X_test[:1])

    # Single sample latency
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_test[:1])
        latencies.append((time.perf_counter() - start) * 1000)

    # Batch inference
    gc.collect()
    mem_before = get_memory_mb()
    batch_start = time.perf_counter()
    preds = model.predict(X_test)
    batch_time = (time.perf_counter() - batch_start) * 1000
    mem_after = get_memory_mb()

    acc = accuracy_score(y_test, preds)

    return {
        "model": "Sklearn RandomForest (PKL)",
        "single_latency_ms": {
            "mean": round(np.mean(latencies), 3),
            "p50": round(np.percentile(latencies, 50), 3),
            "p95": round(np.percentile(latencies, 95), 3),
            "p99": round(np.percentile(latencies, 99), 3),
        },
        "batch_inference_ms": round(batch_time, 2),
        "throughput_samples_per_sec": round(len(X_test) / (batch_time / 1000), 0),
        "memory_delta_mb": round(mem_after - mem_before, 2),
        "accuracy": round(acc, 4),
        "n_samples": len(X_test)
    }


def benchmark_onnx(model_path, label, X_test, y_test, n_warmup=10, n_runs=200):
    """Benchmark ONNX model inference."""
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess = rt.InferenceSession(model_path, sess_options=sess_options)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Warmup
    for _ in range(n_warmup):
        sess.run([output_name], {input_name: X_test[:1]})

    # Single sample latency
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run([output_name], {input_name: X_test[:1]})
        latencies.append((time.perf_counter() - start) * 1000)

    # Batch inference
    gc.collect()
    mem_before = get_memory_mb()
    batch_start = time.perf_counter()
    raw_preds = sess.run([output_name], {input_name: X_test})[0]
    batch_time = (time.perf_counter() - batch_start) * 1000
    mem_after = get_memory_mb()

    acc = accuracy_score(y_test, raw_preds)

    return {
        "model": label,
        "single_latency_ms": {
            "mean": round(np.mean(latencies), 3),
            "p50": round(np.percentile(latencies, 50), 3),
            "p95": round(np.percentile(latencies, 95), 3),
            "p99": round(np.percentile(latencies, 99), 3),
        },
        "batch_inference_ms": round(batch_time, 2),
        "throughput_samples_per_sec": round(len(X_test) / (batch_time / 1000), 0),
        "memory_delta_mb": round(mem_after - mem_before, 2),
        "accuracy": round(acc, 4),
        "n_samples": len(X_test)
    }


def print_results(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 75)
    print("  EDGE DEPLOYMENT BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Model':<35} {'P50 (ms)':>9} {'P95 (ms)':>9} {'Batch (ms)':>11} {'Acc':>7}")
    print("-" * 75)

    for r in results:
        print(
            f"{r['model']:<35} "
            f"{r['single_latency_ms']['p50']:>9.3f} "
            f"{r['single_latency_ms']['p95']:>9.3f} "
            f"{r['batch_inference_ms']:>11.2f} "
            f"{r['accuracy']:>7.4f}"
        )

    print("-" * 75)

    # Load size data
    with open("models/model_metadata.json") as f:
        meta = json.load(f)

    sizes = meta["model_sizes_mb"]
    print(f"\nModel sizes:")
    print(f"  Sklearn PKL:  {sizes['sklearn_pkl']:.3f} MB")
    print(f"  ONNX FP32:    {sizes['onnx_fp32']:.3f} MB")
    print(f"  ONNX INT8:    {sizes['onnx_int8']:.3f} MB")
    print(f"  INT8 size reduction: {meta['size_reduction_pct']}%")
    print("=" * 75)

    # Speedup calculation
    if len(results) >= 2:
        baseline = results[0]["single_latency_ms"]["p50"]
        for r in results[1:]:
            speedup = baseline / r["single_latency_ms"]["p50"]
            print(f"  {r['model']} speedup vs sklearn: {speedup:.2f}x")


def main():
    print("Loading test data...")
    X_test, y_test, le = load_test_data()
    print(f"Test samples: {len(X_test)}")

    results = []

    print("\nBenchmarking Sklearn model...")
    sklearn_result = benchmark_sklearn(X_test, y_test)
    results.append(sklearn_result)
    print(f"  P50 latency: {sklearn_result['single_latency_ms']['p50']:.3f}ms")

    if os.path.exists("models/risk_classifier.onnx"):
        print("\nBenchmarking ONNX FP32 model...")
        fp32_result = benchmark_onnx(
            "models/risk_classifier.onnx",
            "ONNX FP32",
            X_test, y_test
        )
        results.append(fp32_result)
        print(f"  P50 latency: {fp32_result['single_latency_ms']['p50']:.3f}ms")

    if os.path.exists("models/risk_classifier_int8.onnx"):
        print("\nBenchmarking ONNX INT8 model...")
        int8_result = benchmark_onnx(
            "models/risk_classifier_int8.onnx",
            "ONNX INT8 (quantised)",
            X_test, y_test
        )
        results.append(int8_result)
        print(f"  P50 latency: {int8_result['single_latency_ms']['p50']:.3f}ms")

    print_results(results)

    with open("benchmarks/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmarks/results.json")


if __name__ == "__main__":
    main()