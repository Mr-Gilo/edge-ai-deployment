"""
Train a Random Forest classifier and export to ONNX format.

Demonstrates the full model lifecycle:
1. Train on safety incident data
2. Evaluate performance
3. Export to ONNX (standard format for edge deployment)
4. Export quantised INT8 version
5. Verify ONNX outputs match sklearn outputs
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as rt
from data_generator import generate_safety_dataset

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def load_or_generate_data():
    path = "data/safety_incidents.csv"
    if not os.path.exists(path):
        print("Generating dataset...")
        generate_safety_dataset()
    return pd.read_csv(path)


def prepare_features(df):
    feature_cols = [c for c in df.columns if c != "risk_label"]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["risk_label"])
    return X, y, le, feature_cols


def train_model(X_train, y_train):
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    #Only include classes present in the test set
    present_classes = sorted(set(y_test) | set(y_pred))
    present_names = [le.classes_[i] for i in present_classes]

    report = classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=present_names,
        output_dict=True
    )
    print(f"\nModel Accuracy: {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=present_names
        )
    )
    return acc, report


def export_to_onnx(model, n_features, model_name="risk_classifier"):
    """Export sklearn model to ONNX format."""
    print(f"\nExporting to ONNX...")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15
    )

    onnx_path = f"models/{model_name}.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model saved: {onnx_path} ({size_mb:.2f} MB)")
    return onnx_path, size_mb


def quantise_onnx(onnx_path, model_name="risk_classifier_int8"):
    """
    Quantise ONNX model to INT8.
    Reduces model size and memory footprint for edge deployment.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\nQuantising to INT8...")
    quantised_path = f"models/{model_name}.onnx"

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantised_path,
        weight_type=QuantType.QInt8
    )

    size_mb = os.path.getsize(quantised_path) / (1024 * 1024)
    print(f"INT8 model saved: {quantised_path} ({size_mb:.2f} MB)")
    return quantised_path, size_mb


def verify_onnx_outputs(model, onnx_path, X_test, le):
    """Verify ONNX model produces same predictions as sklearn model."""
    print("\nVerifying ONNX outputs match sklearn...")

    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    # Run ONNX inference
    onnx_preds_raw = sess.run(None, {input_name: X_test[:20].astype(np.float32)})
    onnx_preds = onnx_preds_raw[0]

    # Sklearn predictions
    sklearn_preds = model.predict(X_test[:20])

    matches = np.sum(onnx_preds == sklearn_preds)
    print(f"Predictions match: {matches}/20")

    if matches == 20:
        print("ONNX export verified — outputs identical to sklearn")
    else:
        print(f"Warning: {20 - matches} predictions differ")

    return matches == 20

def numpy_encoder(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_metadata(results: dict):
    with open("models/model_metadata.json", "w") as f:
        json.dump(results, f, indent=2, default=numpy_encoder)
    print(f"\nMetadata saved to models/model_metadata.json")


def main():
    # Load data
    df = load_or_generate_data()
    X, y, le, feature_cols = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    acc, report = evaluate_model(model, X_test, y_test, le)

    # Save sklearn model
    with open("models/sklearn_model.pkl", "wb") as f:
        pickle.dump({"model": model, "label_encoder": le,
                     "feature_cols": feature_cols}, f)

    sklearn_size = os.path.getsize("models/sklearn_model.pkl") / (1024 * 1024)
    print(f"Sklearn model saved ({sklearn_size:.2f} MB)")

    # Export to ONNX
    onnx_path, onnx_size = export_to_onnx(model, len(feature_cols))

    # Quantise
    int8_path, int8_size = quantise_onnx(onnx_path)

    # Verify
    verified = verify_onnx_outputs(model, onnx_path, X_test, le)

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 8,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "classes": list(le.classes_),
        "accuracy": round(acc, 4),
        "model_sizes_mb": {
            "sklearn_pkl": round(sklearn_size, 3),
            "onnx_fp32": round(onnx_size, 3),
            "onnx_int8": round(int8_size, 3)
        },
        "size_reduction_pct": round(
            100 * (1 - int8_size / onnx_size), 1
        ),
        "onnx_verified": verified
    }
    save_metadata(metadata)

    print("\n" + "="*50)
    print("  EXPORT SUMMARY")
    print("="*50)
    print(f"  Sklearn PKL:  {sklearn_size:.3f} MB")
    print(f"  ONNX FP32:    {onnx_size:.3f} MB")
    print(f"  ONNX INT8:    {int8_size:.3f} MB")
    print(f"  Size reduction: {metadata['size_reduction_pct']}%")
    print(f"  Accuracy: {acc:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()