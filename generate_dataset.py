"""
Smart Posture Detection - Dataset Generator
============================================
Since we don't have a live webcam, this script generates a realistic
synthetic dataset based on real MediaPipe landmark angle distributions
for 4 posture classes. Run this FIRST before training.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

POSTURE_CLASSES = {
    0: "Good Posture",
    1: "Slouching",
    2: "Forward Head",
    3: "Lateral Tilt"
}

FEATURE_NAMES = [
    "neck_angle",
    "shoulder_tilt",
    "spine_angle",
    "forward_head_ratio",
    "hip_shoulder_alignment",
    "ear_shoulder_distance",
    "shoulder_symmetry",
    "torso_lean"
]

# Realistic angle distributions per posture class (mean, std)
DISTRIBUTIONS = {
    0: {  # Good Posture
        "neck_angle":              (170, 4),
        "shoulder_tilt":           (2,   1.5),
        "spine_angle":             (175, 3),
        "forward_head_ratio":      (0.05, 0.02),
        "hip_shoulder_alignment":  (178, 3),
        "ear_shoulder_distance":   (0.12, 0.02),
        "shoulder_symmetry":       (0.98, 0.01),
        "torso_lean":              (2,   1.5),
    },
    1: {  # Slouching
        "neck_angle":              (145, 8),
        "shoulder_tilt":           (5,   3),
        "spine_angle":             (145, 8),
        "forward_head_ratio":      (0.15, 0.04),
        "hip_shoulder_alignment":  (155, 8),
        "ear_shoulder_distance":   (0.20, 0.04),
        "shoulder_symmetry":       (0.90, 0.04),
        "torso_lean":              (15,  5),
    },
    2: {  # Forward Head Posture
        "neck_angle":              (130, 7),
        "shoulder_tilt":           (3,   2),
        "spine_angle":             (168, 4),
        "forward_head_ratio":      (0.30, 0.05),
        "hip_shoulder_alignment":  (172, 4),
        "ear_shoulder_distance":   (0.32, 0.05),
        "shoulder_symmetry":       (0.94, 0.03),
        "torso_lean":              (5,   2),
    },
    3: {  # Lateral Tilt
        "neck_angle":              (165, 5),
        "shoulder_tilt":           (18,  5),
        "spine_angle":             (165, 5),
        "forward_head_ratio":      (0.08, 0.03),
        "hip_shoulder_alignment":  (168, 5),
        "ear_shoulder_distance":   (0.14, 0.03),
        "shoulder_symmetry":       (0.78, 0.06),
        "torso_lean":              (18,  5),
    }
}

SAMPLES_PER_CLASS = 400


def generate_samples(class_id, n_samples):
    dist = DISTRIBUTIONS[class_id]
    rows = []
    for _ in range(n_samples):
        row = {}
        for feature in FEATURE_NAMES:
            mean, std = dist[feature]
            value = np.random.normal(mean, std)
            # Clip to sensible ranges
            if "angle" in feature:
                value = np.clip(value, 0, 180)
            elif "ratio" in feature or "symmetry" in feature:
                value = np.clip(value, 0, 1)
            else:
                value = max(0, value)
            row[feature] = round(value, 4)
        row["label"] = class_id
        row["posture_class"] = POSTURE_CLASSES[class_id]
        rows.append(row)
    return rows


def main():
    print("Generating posture dataset...")
    all_rows = []
    for class_id in POSTURE_CLASSES:
        rows = generate_samples(class_id, SAMPLES_PER_CLASS)
        all_rows.extend(rows)
        print(f"  Generated {SAMPLES_PER_CLASS} samples for: {POSTURE_CLASSES[class_id]}")

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/posture_dataset.csv", index=False)

    print(f"\nDataset saved to data/posture_dataset.csv")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df["posture_class"].value_counts())
    print(f"\nFeature preview:")
    print(df[FEATURE_NAMES].describe().round(3))


if __name__ == "__main__":
    main()
