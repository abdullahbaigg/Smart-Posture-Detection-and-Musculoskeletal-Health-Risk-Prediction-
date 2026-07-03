"""
Smart Posture Detection - Model Training & Comparison
=====================================================
Uses the Zenodo MultiPosture dataset (real MediaPipe landmarks).

Pipeline:
  1. Load raw landmark CSV
  2. Engineer 8 biomechanical features per frame
  3. Train & compare 4 ML models
  4. Save best model + plots
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (classification_report, confusion_matrix,
                                      accuracy_score, f1_score)

# local util
sys.path.insert(0, os.path.dirname(__file__))
from utils.features import build_feature_dataset, FEATURE_NAMES

DATA_PATH   = "data/data.csv"
MODEL_DIR   = "models"
PLOT_DIR    = "screenshots"

MODEL_COLORS = {
    "Logistic Regression": "#7F77DD",
    "Random Forest":       "#1D9E75",
    "SVM":                 "#D85A30",
    "Gradient Boosting":   "#EF9F27",
}


# ── 1. Load & feature-engineer ────────────────────────────────────────────────
def load_data():
    feat_df = build_feature_dataset(DATA_PATH)
    feat_df.to_csv("data/features.csv", index=False)
    print("Feature CSV saved → data/features.csv")

    X = feat_df[FEATURE_NAMES].values
    y = feat_df["label"].values
    return X, y, feat_df


# ── 2. Train models ───────────────────────────────────────────────────────────
def train_all(X_tr, X_te, y_tr, y_te):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=12,
                                                      min_samples_leaf=2, random_state=42),
        "SVM":                 SVC(kernel="rbf", C=5.0, gamma="scale",
                                   probability=True, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                          max_depth=4, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, mdl in models.items():
        print(f"\n  Training {name}...")
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, average="weighted")
        cv_s = cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="f1_weighted").mean()
        results[name] = {"model": mdl, "y_pred": y_pred,
                         "acc": acc, "f1": f1, "cv": cv_s}
        print(f"    Acc={acc:.4f}  F1={f1:.4f}  CV-F1={cv_s:.4f}")
        print(classification_report(y_te, y_pred, zero_division=0))

    return results


# ── 3. Plots ──────────────────────────────────────────────────────────────────
def plot_comparison(results):
    names  = list(results.keys())
    accs   = [results[n]["acc"] for n in names]
    f1s    = [results[n]["f1"]  for n in names]
    cvs    = [results[n]["cv"]  for n in names]
    colors = [MODEL_COLORS[n]   for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance — Smart Posture Detection (Zenodo Dataset)",
                 fontsize=13, fontweight="bold")

    for ax, vals, title in zip(axes,
                                [accs, f1s, cvs],
                                ["Test Accuracy", "Weighted F1-Score", "5-Fold CV F1"]):
        bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
        ax.axhline(0.9, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = f"{PLOT_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_confusion_matrices(results, y_te):
    classes = sorted(set(y_te))
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip(axes.flat, results.items()):
        cm = confusion_matrix(y_te, res["y_pred"], labels=classes)
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name}\nAcc={res['acc']:.3f}  F1={res['f1']:.3f}",
                     fontsize=10, fontweight="bold")
        short = [c[:6] for c in classes]
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thresh = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > thresh else "black",
                        fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = f"{PLOT_DIR}/confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_feature_importance(rf_model):
    imp  = rf_model.feature_importances_
    idx  = np.argsort(imp)
    feat = [FEATURE_NAMES[i] for i in idx]
    vals = imp[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feat, vals, color="#1D9E75", edgecolor="white", linewidth=0.8)
    ax.set_title("Feature Importance — Random Forest", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = f"{PLOT_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_class_distribution(feat_df):
    counts = feat_df["label"].value_counts()
    colors = ["#1D9E75","#D85A30","#7F77DD","#EF9F27","#D4537E"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)],
                  edgecolor="white", linewidth=1.2, width=0.5)
    ax.set_title("Class Distribution — Zenodo MultiPosture Dataset",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of frames")
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = f"{PLOT_DIR}/class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


# ── 4. Save artefacts ─────────────────────────────────────────────────────────
def save_artefacts(results, scaler, classes):
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_name = max(results, key=lambda n: results[n]["f1"])

    joblib.dump(results[best_name]["model"], f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler,                      f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(classes,                     f"{MODEL_DIR}/classes.pkl")
    joblib.dump({n: r["model"] for n, r in results.items()},
                f"{MODEL_DIR}/all_models.pkl")

    summary = {n: {"accuracy": round(r["acc"],4),
                   "f1":       round(r["f1"],4),
                   "cv":       round(r["cv"],4)}
               for n, r in results.items()}
    with open(f"{MODEL_DIR}/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest model: {best_name}  (F1={results[best_name]['f1']:.4f})")
    return best_name


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Smart Posture Detection — Training on Zenodo Dataset")
    print("=" * 60)

    os.makedirs(PLOT_DIR, exist_ok=True)

    X, y, feat_df = load_data()
    classes = sorted(np.unique(y))

    plot_class_distribution(feat_df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    print("\nTraining models...")
    results = train_all(X_tr_s, X_te_s, y_tr, y_te)

    plot_comparison(results)
    plot_confusion_matrices(results, y_te)
    plot_feature_importance(results["Random Forest"]["model"])

    best = save_artefacts(results, scaler, classes)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1-Score':>10} {'CV-F1':>10}")
    print("-" * 60)
    for name, res in results.items():
        marker = "  ← BEST" if name == best else ""
        print(f"{name:<22} {res['acc']:>10.4f} {res['f1']:>10.4f} {res['cv']:>10.4f}{marker}")
    print("=" * 60)
    print("\nDone! Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
