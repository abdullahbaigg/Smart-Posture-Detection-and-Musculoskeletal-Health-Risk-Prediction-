"""
Smart Posture Detection - Feature Engineering
==============================================
Computes 8 biomechanical features from raw MediaPipe
x/y/z landmark coordinates (Zenodo MultiPosture dataset).

Label mapping (upperbody_label):
    TUP -> Good Posture        (Trunk Upright)
    TLF -> Forward Lean        (Trunk Leaning Forward)
    TLB -> Backward Lean       (Trunk Leaning Backward)
    TLL -> Lateral Tilt Left   (Trunk Leaning Left)
    TLR -> Lateral Tilt Right  (Trunk Leaning Right)
"""

import numpy as np
import pandas as pd

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    "TUP": "Good Posture",
    "TLF": "Forward Lean",
    "TLB": "Backward Lean",
    "TLL": "Lateral Tilt Left",
    "TLR": "Lateral Tilt Right",
}

POSTURE_COLORS = {
    "Good Posture":       "#1D9E75",
    "Forward Lean":       "#D85A30",
    "Backward Lean":      "#7F77DD",
    "Lateral Tilt Left":  "#EF9F27",
    "Lateral Tilt Right": "#EF9F27",
}

POSTURE_RISK = {
    "Good Posture":       "Low",
    "Forward Lean":       "High",
    "Backward Lean":      "Medium",
    "Lateral Tilt Left":  "Medium",
    "Lateral Tilt Right": "Medium",
}

POSTURE_TIPS = {
    "Good Posture":       ["Great posture! Keep it up.", "Take a break every 30 mins."],
    "Forward Lean":       ["Pull your chin back (chin tucks).", "Raise your screen to eye level.", "Strengthen your core."],
    "Backward Lean":      ["Sit closer to your desk.", "Bring your screen forward.", "Use lumbar support."],
    "Lateral Tilt Left":  ["Level both shoulders.", "Check your left armrest height.", "Stretch your right side."],
    "Lateral Tilt Right": ["Level both shoulders.", "Check your right armrest height.", "Stretch your left side."],
}

FEATURE_NAMES = [
    "neck_angle",
    "shoulder_tilt",
    "spine_lean_angle",
    "forward_head_offset",
    "shoulder_hip_alignment",
    "ear_shoulder_dist",
    "shoulder_symmetry",
    "lateral_lean",
]


def _angle(a, b, c):
    """Angle at point B in the triangle A-B-C (degrees, 0-180)."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))


def extract_features_from_row(row):
    """
    Compute 8 posture features from a single dataset row.
    Expects columns like left_shoulder_x, left_shoulder_y, etc.
    """
    def pt(name):
        return [row[f"{name}_x"], row[f"{name}_y"], row.get(f"{name}_z", 0)]

    nose          = pt("nose")
    left_ear      = pt("left_ear")
    right_ear     = pt("right_ear")
    left_shoulder = pt("left_shoulder")
    right_shoulder= pt("right_shoulder")
    left_hip      = pt("left_hip")
    right_hip     = pt("right_hip")

    mid_ear      = [(left_ear[0]+right_ear[0])/2,      (left_ear[1]+right_ear[1])/2,      0]
    mid_shoulder = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2, 0]
    mid_hip      = [(left_hip[0]+right_hip[0])/2,      (left_hip[1]+right_hip[1])/2,      0]
    vertical_up  = [mid_shoulder[0], mid_shoulder[1] - 0.3, 0]

    # 1. Neck angle: ear-shoulder-hip (lower = more forward lean)
    neck_angle = _angle(mid_ear, mid_shoulder, mid_hip)

    # 2. Shoulder tilt: vertical height difference (lateral imbalance)
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) * 100

    # 3. Spine lean: shoulder-hip vs vertical reference
    spine_lean = _angle(vertical_up, mid_shoulder, mid_hip)

    # 4. Forward head offset: ear x vs shoulder x (positive = forward)
    forward_head_offset = float((mid_shoulder[0] - mid_ear[0]) * 100)

    # 5. Shoulder-hip alignment
    shoulder_hip_align = _angle(mid_ear, mid_shoulder, mid_hip)

    # 6. Ear-shoulder distance (normalized)
    ear_shoulder_dist = _dist(mid_ear, mid_shoulder)

    # 7. Shoulder symmetry (1.0 = perfect)
    l_len = _dist(left_shoulder, left_hip)
    r_len = _dist(right_shoulder, right_hip)
    shoulder_sym = min(l_len, r_len) / (max(l_len, r_len) + 1e-9)

    # 8. Lateral lean: horizontal displacement of mid-shoulder vs mid-hip
    lateral_lean = abs(mid_shoulder[0] - mid_hip[0]) * 100

    return {
        "neck_angle":             round(neck_angle, 4),
        "shoulder_tilt":          round(shoulder_tilt, 4),
        "spine_lean_angle":       round(spine_lean, 4),
        "forward_head_offset":    round(forward_head_offset, 4),
        "shoulder_hip_alignment": round(shoulder_hip_align, 4),
        "ear_shoulder_dist":      round(ear_shoulder_dist, 4),
        "shoulder_symmetry":      round(shoulder_sym, 4),
        "lateral_lean":           round(lateral_lean, 4),
    }


def build_feature_dataset(csv_path):
    """
    Load raw landmark CSV, compute features for every row,
    return a clean feature DataFrame ready for ML.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    records = []
    for _, row in df.iterrows():
        feats = extract_features_from_row(row)
        feats["label_raw"]   = row["upperbody_label"]
        feats["label"]       = LABEL_MAP.get(row["upperbody_label"], row["upperbody_label"])
        feats["subject"]     = row["subject"]
        records.append(feats)

    feat_df = pd.DataFrame(records)
    print(f"Features built: {feat_df.shape}")
    print("Class distribution:")
    print(feat_df["label"].value_counts())
    return feat_df


def calculate_health_risk(history):
    """
    Given a list of session records {posture, duration},
    return (risk_level, bad_posture_pct).
    """
    if not history:
        return "Low", 0.0
    bad = {"Forward Lean", "Backward Lean", "Lateral Tilt Left", "Lateral Tilt Right"}
    bad_mins   = sum(r["duration"] for r in history if r["posture"] in bad)
    total_mins = sum(r["duration"] for r in history)
    ratio      = bad_mins / (total_mins + 1e-9)
    if bad_mins >= 45 or ratio > 0.6:
        return "High",   round(ratio * 100, 1)
    elif bad_mins >= 20 or ratio > 0.3:
        return "Medium", round(ratio * 100, 1)
    return "Low", round(ratio * 100, 1)
