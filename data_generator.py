"""
Synthetic safety incident dataset generator.
Produces a realistic classification dataset for edge deployment demo.
Task: classify incident reports as low / medium / high / critical risk.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

SEED = 42
np.random.seed(SEED)


def generate_safety_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate synthetic safety incident features for risk classification.

    Features span multiple modalities mirroring the multimodal-risk-pipeline:
    - Time-series statistics (sensor readings)
    - Tabular metadata (equipment, location)
    - Event indicators (flag types)
    """

    # Sensor statistics (time-series derived features)
    sensor_mean = np.random.normal(50, 15, n_samples)
    sensor_std = np.abs(np.random.normal(5, 3, n_samples))
    sensor_max = sensor_mean + np.abs(np.random.normal(20, 10, n_samples))
    sensor_min = sensor_mean - np.abs(np.random.normal(20, 10, n_samples))
    sensor_roc = np.random.normal(0, 2, n_samples)  # rate of change

    # Equipment metadata
    equipment_age = np.random.uniform(0.5, 15, n_samples)
    days_since_maintenance = np.random.randint(1, 365, n_samples)
    pressure_rating = np.random.uniform(50, 200, n_samples)
    n_prior_incidents = np.random.poisson(0.5, n_samples)

    # Event indicators
    is_vpn = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    geo_mismatch = np.random.choice([0, 1], n_samples, p=[0.90, 0.10])
    high_velocity = np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    after_hours = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    repeated_failures = np.random.choice([0, 1], n_samples, p=[0.80, 0.20])

    # Construct features
    df = pd.DataFrame({
        "sensor_mean": sensor_mean,
        "sensor_std": sensor_std,
        "sensor_max": sensor_max,
        "sensor_min": sensor_min,
        "sensor_roc": sensor_roc,
        "equipment_age": equipment_age,
        "days_since_maintenance": days_since_maintenance,
        "pressure_rating": pressure_rating,
        "n_prior_incidents": n_prior_incidents,
        "is_vpn": is_vpn,
        "geo_mismatch": geo_mismatch,
        "high_velocity": high_velocity,
        "after_hours": after_hours,
        "repeated_failures": repeated_failures,
    })

    # Generate risk labels based on feature combinations
    risk_score = (
        (sensor_mean > 70).astype(int) * 2 +
        (sensor_std > 10).astype(int) * 1 +
        (sensor_roc > 3).astype(int) * 2 +
        (equipment_age > 10).astype(int) * 1 +
        (days_since_maintenance > 180).astype(int) * 2 +
        (n_prior_incidents > 2).astype(int) * 3 +
        is_vpn * 2 +
        geo_mismatch * 3 +
        high_velocity * 2 +
        after_hours * 1 +
        repeated_failures * 2 +
        np.random.randint(0, 4, n_samples)  # noise
    )

    def score_to_label(s):
        if s <= 3:
            return "low"
        elif s <= 7:
            return "medium"
        elif s <= 11:
            return "high"
        else:
            return "critical"

    df["risk_label"] = [score_to_label(s) for s in risk_score]

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/safety_incidents.csv", index=False)

    label_counts = df["risk_label"].value_counts()
    print(f"Dataset generated: {len(df)} samples")
    print(f"Label distribution:\n{label_counts}")

    return df


if __name__ == "__main__":
    generate_safety_dataset()