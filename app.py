"""
Smart Posture Detection & Health Risk Prediction
================================================
Streamlit Web App — built on Zenodo MultiPosture Dataset

Pages:
  1. Home & Demo      — manual slider analysis + simulated live session
  2. Model Comparison — performance charts for all 4 models
  3. Health Dashboard — session history, risk score, timeline
  4. About            — project documentation

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib, json, os, time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Posture AI",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title{font-size:2.1rem;font-weight:700;
  background:linear-gradient(135deg,#534AB7,#1D9E75);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.1rem}
.sub{color:#6c757d;font-size:.95rem;margin-bottom:1rem}
.good-badge{background:#d4edda;color:#155724;padding:4px 14px;border-radius:20px;font-weight:600}
.warn-badge{background:#fff3cd;color:#856404;padding:4px 14px;border-radius:20px;font-weight:600}
.bad-badge {background:#f8d7da;color:#721c24;padding:4px 14px;border-radius:20px;font-weight:600}
.risk-Low   {color:#1D9E75;font-weight:700;font-size:1.3rem}
.risk-Medium{color:#EF9F27;font-weight:700;font-size:1.3rem}
.risk-High  {color:#D85A30;font-weight:700;font-size:1.3rem}
div[data-testid="metric-container"]{background:#f8f9fa;border-radius:10px;padding:10px}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "neck_angle", "shoulder_tilt", "spine_lean_angle",
    "forward_head_offset", "shoulder_hip_alignment",
    "ear_shoulder_dist", "shoulder_symmetry", "lateral_lean",
]

POSTURE_COLORS = {
    "Good Posture":       "#1D9E75",
    "Forward Lean":       "#D85A30",
    "Backward Lean":      "#7F77DD",
    "Lateral Tilt Left":  "#EF9F27",
    "Lateral Tilt Right": "#EF9F27",
}

POSTURE_TIPS = {
    "Good Posture":       ["Great posture! Keep it up.", "Take a break every 30 mins."],
    "Forward Lean":       ["Pull your chin back (chin tucks).", "Raise your screen to eye level.", "Strengthen core muscles."],
    "Backward Lean":      ["Sit closer to your desk.", "Bring your screen forward.", "Use lumbar support."],
    "Lateral Tilt Left":  ["Level both shoulders.", "Check your left armrest height.", "Stretch your right side."],
    "Lateral Tilt Right": ["Level both shoulders.", "Check your right armrest height.", "Stretch your left side."],
}

POSTURE_RISK = {
    "Good Posture":       "Low",
    "Forward Lean":       "High",
    "Backward Lean":      "Medium",
    "Lateral Tilt Left":  "Medium",
    "Lateral Tilt Right": "Medium",
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model      = joblib.load("models/best_model.pkl")
        scaler     = joblib.load("models/scaler.pkl")
        all_models = joblib.load("models/all_models.pkl")
        with open("models/model_summary.json") as f:
            summary = json.load(f)
        return model, scaler, all_models, summary
    except FileNotFoundError:
        return None, None, None, None


def predict(features_dict, model, scaler):
    X = np.array([[features_dict[f] for f in FEATURE_NAMES]])
    Xs = scaler.transform(X)
    pred  = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0]
    conf  = dict(zip(model.classes_, proba))
    return pred, conf


def health_risk(history):
    if not history:
        return "Low", 0.0
    bad      = {"Forward Lean","Backward Lean","Lateral Tilt Left","Lateral Tilt Right"}
    bad_m    = sum(r["duration"] for r in history if r["posture"] in bad)
    total_m  = sum(r["duration"] for r in history)
    ratio    = bad_m / (total_m + 1e-9)
    if bad_m >= 45 or ratio > 0.6: return "High",   round(ratio*100,1)
    if bad_m >= 20 or ratio > 0.3: return "Medium",  round(ratio*100,1)
    return "Low", round(ratio*100,1)


def badge(label):
    if label == "Good Posture": return f'<span class="good-badge">{label}</span>'
    if label in ("Forward Lean","Backward Lean"): return f'<span class="bad-badge">{label}</span>'
    return f'<span class="warn-badge">{label}</span>'


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧍 Smart Posture AI")
    st.caption("Built on Zenodo MultiPosture Dataset\n4,794 real frames · 13 participants · 5 posture classes")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home & Demo",
        "📊 Model Comparison",
        "📈 Health Dashboard",
        "ℹ️ About the Project"
    ])
    st.divider()
    st.caption("MediaPipe · scikit-learn · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home & Demo
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Demo":
    st.markdown('<div class="main-title">Smart Posture Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">AI-powered posture analysis using real MediaPipe skeletal data · Zenodo MultiPosture Dataset</div>', unsafe_allow_html=True)

    model, scaler, all_models, summary = load_models()
    if model is None:
        st.error("Models not found. Run `python train_models.py` first.")
        st.code("python train_models.py")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📸 Upload Photo", "🎛️ Manual Feature Analysis", "🎬 Live Session Simulation"])

    # ── Tab 1: Photo Upload ───────────────────────────────────────────────────
    with tab1:
        st.subheader("Upload a Photo for Posture Analysis")
        st.info("Upload a photo of a person sitting or standing. MediaPipe detects body landmarks and the AI classifies posture and generates a health risk score.")

        col_upload, col_result = st.columns([1, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose a photo",
                type=["jpg", "jpeg", "png"],
                help="Best results: full body visible, good lighting, person facing camera"
            )

            if uploaded_file:
                import cv2
                import mediapipe as mp
                import sys
                sys.path.insert(0, ".")
                from utils.features import extract_features_from_row

                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_cv  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                st.image(image_rgb, caption="Uploaded photo", use_container_width=True)

                if st.button("Analyse Posture", type="primary", use_container_width=True, key="btn_photo"):
                    with st.spinner("Running MediaPipe pose detection..."):
                        from mediapipe.tasks import python as mp_python
                        from mediapipe.tasks.python import vision
                        import mediapipe as mp
                        import urllib.request, os

                        # Download model if not present
                        model_path = "pose_landmarker.task"
                        if not os.path.exists(model_path):
                            st.info("Downloading MediaPipe model (one-time, ~30MB)...")
                            urllib.request.urlretrieve(
                                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                                model_path
                            )

                        base_options = mp_python.BaseOptions(model_asset_path=model_path)
                        options = vision.PoseLandmarkerOptions(
                            base_options=base_options,
                            output_segmentation_masks=False
                        )
                        detector = vision.PoseLandmarker.create_from_options(options)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        detection_result = detector.detect(mp_image)

                    if not detection_result.pose_landmarks:
                        st.error("No person detected. Try a clearer photo with the full body visible.")
                    else:
                        # Draw landmarks manually using OpenCV
                        annotated = image_rgb.copy()
                        h, w = annotated.shape[:2]
                        lm_list = detection_result.pose_landmarks[0]

                        # Draw connections
                        CONNECTIONS = [
                            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
                            (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
                            (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
                        ]
                        for a, b in CONNECTIONS:
                            if a < len(lm_list) and b < len(lm_list):
                                x1,y1 = int(lm_list[a].x*w), int(lm_list[a].y*h)
                                x2,y2 = int(lm_list[b].x*w), int(lm_list[b].y*h)
                                cv2.line(annotated, (x1,y1), (x2,y2), (29,158,117), 2)
                        for lmk in lm_list:
                            cx, cy = int(lmk.x*w), int(lmk.y*h)
                            cv2.circle(annotated, (cx,cy), 5, (29,158,117), -1)
                            cv2.circle(annotated, (cx,cy), 5, (255,255,255), 1)

                        # Build landmark dict using index-based names
                        LANDMARK_NAMES = [
                            "nose","left_eye_inner","left_eye","left_eye_outer",
                            "right_eye_inner","right_eye","right_eye_outer",
                            "left_ear","right_ear","mouth_left","mouth_right",
                            "left_shoulder","right_shoulder","left_elbow","right_elbow",
                            "left_wrist","right_wrist","left_pinky","right_pinky",
                            "left_index","right_index","left_thumb","right_thumb",
                            "left_hip","right_hip","left_knee","right_knee",
                            "left_ankle","right_ankle","left_heel","right_heel",
                            "left_foot_index","right_foot_index"
                        ]
                        lm_dict = {}
                        for i, name in enumerate(LANDMARK_NAMES):
                            if i < len(lm_list):
                                lm_dict[f"{name}_x"] = lm_list[i].x
                                lm_dict[f"{name}_y"] = lm_list[i].y
                                lm_dict[f"{name}_z"] = lm_list[i].z

                        row = pd.Series(lm_dict)
                        feats = extract_features_from_row(row)
                        pred, conf = predict(feats, model, scaler)
                        conf_val   = conf.get(pred, 0)
                        risk       = POSTURE_RISK.get(pred, "Medium")

                        st.session_state["photo_result"] = {
                            "annotated": annotated,
                            "pred": pred, "conf": conf,
                            "conf_val": conf_val, "risk": risk, "features": feats
                        }
                        st.session_state.history.append({
                            "time":       datetime.now().strftime("%H:%M:%S"),
                            "posture":    pred,
                            "confidence": round(conf_val * 100, 1),
                            "duration":   1.0
                        })
                        st.success("Analysis complete! See results on the right.")

        with col_result:
            if "photo_result" in st.session_state:
                r = st.session_state["photo_result"]
                st.image(r["annotated"], caption="Skeleton overlay", use_container_width=True)
                st.divider()
                risk_emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}[r["risk"]]
                badge_cls  = "good-badge" if r["pred"] == "Good Posture" else \
                             "bad-badge"  if r["pred"] in ("Forward Lean","Backward Lean") else "warn-badge"
                st.markdown(f'**Detected Posture:** <span class="{badge_cls}">{r["pred"]}</span>', unsafe_allow_html=True)
                st.metric("Confidence",  f'{r["conf_val"]*100:.1f}%')
                st.metric("Health Risk", f'{risk_emoji} {r["risk"]}')
                st.divider()
                st.markdown("**Confidence by class:**")
                st.dataframe(pd.DataFrame({
                    "Posture":     list(r["conf"].keys()),
                    "Probability": [f'{v*100:.1f}%' for v in r["conf"].values()]
                }), hide_index=True, use_container_width=True)
                st.divider()
                st.markdown("**Recommendations:**")
                for tip in POSTURE_TIPS.get(r["pred"], []):
                    st.markdown(f"• {tip}")
                st.divider()
                st.markdown("**Extracted biomechanical features:**")
                st.dataframe(pd.DataFrame({
                    "Feature": list(r["features"].keys()),
                    "Value":   [round(v, 3) for v in r["features"].values()]
                }), hide_index=True, use_container_width=True)
            else:
                st.markdown("### 👈 Upload a photo and click Analyse")
                st.markdown("Results will appear here:")
                st.markdown("- Skeleton overlay drawn on your photo")
                st.markdown("- Posture class (Good / Forward Lean / Backward Lean / Lateral Tilt)")
                st.markdown("- Confidence score per class")
                st.markdown("- Health risk level (Low / Medium / High)")
                st.markdown("- All 8 extracted biomechanical angles")

    # ── Tab 2: Manual sliders ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Analyse a Posture")
        st.info("Adjust the sliders to represent a posture. These features are the same ones automatically extracted from MediaPipe landmarks in the real dataset.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spinal angles**")
            neck_angle   = st.slider("Neck angle (°)",             100.0, 180.0, 168.0, 0.5,
                                      help="170°+ = upright, <145° = forward lean")
            spine_lean   = st.slider("Spine lean angle (°)",       0.0,   60.0,  5.0,  0.5)
            fwd_offset   = st.slider("Forward head offset",        -5.0,  20.0,  1.0,  0.1,
                                      help=">8 = notable forward head position")
            sh_hip_align = st.slider("Shoulder-hip alignment (°)", 100.0, 180.0, 170.0, 0.5)

        with col2:
            st.markdown("**Balance & symmetry**")
            sh_tilt  = st.slider("Shoulder tilt",       0.0, 20.0, 1.5, 0.1,
                                  help=">8 = lateral imbalance")
            ear_sh   = st.slider("Ear-shoulder dist",   0.05, 0.6, 0.14, 0.01)
            sh_sym   = st.slider("Shoulder symmetry",   0.5,  1.0, 0.97, 0.01,
                                  help="1.0 = perfect symmetry")
            lat_lean = st.slider("Lateral lean",        0.0, 20.0, 1.0,  0.1)

        features = {
            "neck_angle":             neck_angle,
            "shoulder_tilt":          sh_tilt,
            "spine_lean_angle":       spine_lean,
            "forward_head_offset":    fwd_offset,
            "shoulder_hip_alignment": sh_hip_align,
            "ear_shoulder_dist":      ear_sh,
            "shoulder_symmetry":      sh_sym,
            "lateral_lean":           lat_lean,
        }

        if st.button("Analyse Posture", type="primary", use_container_width=True, key="btn_sliders"):
            pred, conf = predict(features, model, scaler)
            conf_val   = conf.get(pred, 0)

            st.divider()
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Detection result**")
                st.markdown(badge(pred), unsafe_allow_html=True)
                st.metric("Confidence", f"{conf_val*100:.1f}%")
                risk = POSTURE_RISK.get(pred, "Medium")
                risk_emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}[risk]
                st.metric("Posture Risk", f"{risk_emoji} {risk}")

            with c2:
                st.markdown("**Class probabilities**")
                prob_df = pd.DataFrame({
                    "Posture":     list(conf.keys()),
                    "Probability": [f"{v*100:.1f}%" for v in conf.values()]
                })
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

            with c3:
                st.markdown("**Recommendations**")
                for tip in POSTURE_TIPS.get(pred, []):
                    st.markdown(f"• {tip}")

            # Log to history
            st.session_state.history.append({
                "time":       datetime.now().strftime("%H:%M:%S"),
                "posture":    pred,
                "confidence": round(conf_val*100, 1),
                "duration":   1.0
            })
            st.success("Logged to Health Dashboard ✓")

    # ── Tab 3: Simulation ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("Simulated Real-Time Session")
        st.info("Simulates a 5-frame posture monitoring session using representative feature values from the real Zenodo dataset. In a live deployment, these values come from your webcam via MediaPipe.")

        n_frames = st.slider("Number of frames to simulate", 5, 20, 8)

        if st.button("Start Session", type="primary"):
            # Representative feature vectors per class (from dataset analysis)
            scenarios = [
                ("Good Posture",       {"neck_angle":170,"shoulder_tilt":1.2,"spine_lean_angle":3,"forward_head_offset":0.5,"shoulder_hip_alignment":169,"ear_shoulder_dist":0.13,"shoulder_symmetry":0.97,"lateral_lean":0.8}),
                ("Forward Lean",       {"neck_angle":142,"shoulder_tilt":2.1,"spine_lean_angle":28,"forward_head_offset":9.5,"shoulder_hip_alignment":141,"ear_shoulder_dist":0.24,"shoulder_symmetry":0.93,"lateral_lean":2.1}),
                ("Good Posture",       {"neck_angle":172,"shoulder_tilt":0.9,"spine_lean_angle":2,"forward_head_offset":0.3,"shoulder_hip_alignment":171,"ear_shoulder_dist":0.12,"shoulder_symmetry":0.98,"lateral_lean":0.5}),
                ("Backward Lean",      {"neck_angle":158,"shoulder_tilt":1.5,"spine_lean_angle":35,"forward_head_offset":-3.2,"shoulder_hip_alignment":157,"ear_shoulder_dist":0.19,"shoulder_symmetry":0.95,"lateral_lean":1.2}),
                ("Lateral Tilt Left",  {"neck_angle":165,"shoulder_tilt":14,"spine_lean_angle":8,"forward_head_offset":1.1,"shoulder_hip_alignment":164,"ear_shoulder_dist":0.15,"shoulder_symmetry":0.74,"lateral_lean":12}),
                ("Forward Lean",       {"neck_angle":138,"shoulder_tilt":2.8,"spine_lean_angle":32,"forward_head_offset":11.2,"shoulder_hip_alignment":137,"ear_shoulder_dist":0.27,"shoulder_symmetry":0.91,"lateral_lean":3.0}),
                ("Good Posture",       {"neck_angle":171,"shoulder_tilt":1.0,"spine_lean_angle":2,"forward_head_offset":0.4,"shoulder_hip_alignment":170,"ear_shoulder_dist":0.13,"shoulder_symmetry":0.97,"lateral_lean":0.7}),
                ("Lateral Tilt Right", {"neck_angle":164,"shoulder_tilt":13,"spine_lean_angle":7,"forward_head_offset":0.9,"shoulder_hip_alignment":163,"ear_shoulder_dist":0.14,"shoulder_symmetry":0.75,"lateral_lean":11}),
            ]
            # Repeat to fill n_frames
            frames = [scenarios[i % len(scenarios)] for i in range(n_frames)]

            prog    = st.progress(0)
            status  = st.empty()
            log_box = st.empty()
            log_rows = []

            for i, (expected, feats) in enumerate(frames):
                prog.progress((i+1)/n_frames)
                status.text(f"Processing frame {i+1}/{n_frames}...")
                time.sleep(0.5)

                pred, conf = predict(feats, model, scaler)
                conf_val   = conf.get(pred, 0)

                log_rows.append({
                    "Frame":      i+1,
                    "Detected":   pred,
                    "Confidence": f"{conf_val*100:.1f}%",
                    "Risk":       POSTURE_RISK.get(pred,"Medium")
                })
                log_box.dataframe(pd.DataFrame(log_rows), hide_index=True, use_container_width=True)

                st.session_state.history.append({
                    "time":       datetime.now().strftime("%H:%M:%S"),
                    "posture":    pred,
                    "confidence": round(conf_val*100, 1),
                    "duration":   0.5
                })

            status.text("Session complete!")
            risk_lv, pct = health_risk(st.session_state.history)
            emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}[risk_lv]
            st.success(f"Session risk: {emoji} **{risk_lv}** ({pct:.1f}% bad posture time)")
            st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Performance of 4 ML algorithms on the Zenodo MultiPosture Dataset</div>', unsafe_allow_html=True)

    _, _, _, summary = load_models()
    if summary is None:
        st.error("Run `python train_models.py` first.")
        st.stop()

    df_s = pd.DataFrame(summary).T.reset_index()
    df_s.columns = ["Model","Accuracy","F1-Score","CV-F1"]
    df_s[["Accuracy","F1-Score","CV-F1"]] = df_s[["Accuracy","F1-Score","CV-F1"]].round(4)
    best = df_s.loc[df_s["F1-Score"].idxmax(), "Model"]

    # KPI cards
    cols = st.columns(4)
    for col, (_, row) in zip(cols, df_s.iterrows()):
        is_best = row["Model"] == best
        col.metric(
            label=row["Model"] + (" ⭐" if is_best else ""),
            value=f"{row['Accuracy']*100:.2f}%",
            delta=f"F1: {row['F1-Score']:.4f}"
        )

    st.divider()

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Model Performance — Smart Posture Detection", fontsize=12, fontweight="bold")
    bar_colors = ["#7F77DD","#1D9E75","#D85A30","#EF9F27"]
    mods = df_s["Model"].tolist()

    for ax, metric in zip(axes, ["Accuracy","F1-Score","CV-F1"]):
        vals = df_s[metric].tolist()
        bars = ax.bar(mods, vals, color=bar_colors, width=0.5, edgecolor="white")
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(mods, rotation=25, ha="right", fontsize=8)
        ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Saved training plots
    st.divider()
    st.subheader("Training Plots")
    for title, path in [
        ("Class Distribution", "screenshots/class_distribution.png"),
        ("Confusion Matrices",  "screenshots/confusion_matrices.png"),
        ("Feature Importance",  "screenshots/feature_importance.png"),
    ]:
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)

    # Algorithm explainers
    st.divider()
    st.subheader("Algorithm Deep Dive")
    tabs = st.tabs(["Logistic Regression","Random Forest","SVM","Gradient Boosting"])
    explanations = [
        ("Logistic Regression",
         "**Baseline model.** Fits a linear boundary between the 5 posture classes using our 8 angle features. Fast and interpretable. Shows us the minimum performance bar — any model worth using must beat this."),
        ("Random Forest",
         "**Best classical ML.** Ensemble of 300 decision trees. Each tree votes; majority wins. Provides feature importance scores revealing that `neck_angle`, `forward_head_offset`, and `spine_lean_angle` are the most discriminative features."),
        ("SVM",
         "**Strong alternative.** Finds the maximum-margin hyperplane using an RBF kernel to handle non-linear posture boundaries. Excellent when training data is limited — generalises well to new participants."),
        ("Gradient Boosting",
         "**Top performer.** Builds 200 shallow trees sequentially, each correcting errors of the previous ensemble. Consistently highest F1-score across posture classes, especially for the minority `TLB`/`TLL`/`TLR` classes."),
    ]
    for tab, (name, text) in zip(tabs, explanations):
        with tab:
            st.markdown(text)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Health Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Health Dashboard":
    st.markdown('<div class="main-title">Health Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Session-based posture tracking and musculoskeletal risk assessment</div>', unsafe_allow_html=True)

    history = st.session_state.history
    if not history:
        st.info("No session data yet. Go to Home & Demo and run an analysis or simulation first!")
        st.stop()

    risk_lv, bad_pct = health_risk(history)
    total_mins = sum(r["duration"] for r in history)
    good_count = sum(1 for r in history if r["posture"] == "Good Posture")

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Detections",  len(history))
    c2.metric("Session Duration",  f"{total_mins:.1f} min")
    c3.metric("Good Posture",      f"{good_count/len(history)*100:.0f}%")
    c4.metric("Health Risk",       risk_lv,
              delta=f"{bad_pct:.1f}% poor posture time")

    st.divider()

    # Risk banner
    emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}[risk_lv]
    st.markdown(f"### Overall Risk: <span class='risk-{risk_lv}'>{emoji} {risk_lv}</span>",
                unsafe_allow_html=True)
    advice = {
        "Low":    "Your posture is generally good. Keep it up!",
        "Medium": "Significant time in poor posture. Take regular stretch breaks.",
        "High":   "Extended poor posture detected. Risk of musculoskeletal disorders is elevated. Please review your workspace ergonomics.",
    }
    {"Low": st.success, "Medium": st.warning, "High": st.error}[risk_lv](advice[risk_lv])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Session Log")
        df_h = pd.DataFrame(history)
        df_h.index += 1
        st.dataframe(df_h[["time","posture","confidence"]], use_container_width=True)

    with col2:
        st.subheader("Posture Distribution")
        counts = df_h["posture"].value_counts()
        colors_pie = [POSTURE_COLORS.get(p,"#888") for p in counts.index]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=colors_pie, startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("Posture class distribution", fontweight="bold")
        st.pyplot(fig); plt.close()

    # Timeline
    if len(history) > 2:
        st.divider()
        st.subheader("Posture Timeline")
        fig, ax = plt.subplots(figsize=(12,3))
        for i, r in enumerate(history):
            ax.bar(i+1, 1, color=POSTURE_COLORS.get(r["posture"],"#888"),
                   edgecolor="white", linewidth=0.5)
        ax.set_yticks([])
        ax.set_xlabel("Detection #")
        ax.set_title("Frame-by-frame posture (green = good, others = needs attention)", fontsize=10)
        patches = [plt.Rectangle((0,0),1,1,color=v,label=k)
                   for k,v in POSTURE_COLORS.items() if k in df_h["posture"].values]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        st.pyplot(fig); plt.close()

    if st.button("Clear Session History"):
        st.session_state.history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About the Project":
    st.markdown('<div class="main-title">About This Project</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    ## Smart Posture Detection & Health Risk Prediction
    **AI Assignment 3 — Healthcare & Bioinformatics Domain**

    ---

    ### Dataset
    **Zenodo MultiPosture Dataset** (Zenodo record 14230872)

    | Property | Value |
    |----------|-------|
    | Source | Zenodo (open-access, CC-BY) |
    | Participants | 13 real subjects |
    | Total frames | 4,794 |
    | Landmark tool | MediaPipe Pose Heavy model |
    | Format | CSV — 99 landmark coordinates (x, y, z) per frame |
    | Labels | Upper body (5 classes) + Lower body (7 classes) |

    ### Posture Classes (Upper Body)
    | Code | Class Name | Health Risk |
    |------|-----------|-------------|
    | TUP | Good Posture (Trunk Upright) | Low |
    | TLF | Forward Lean | High |
    | TLB | Backward Lean | Medium |
    | TLL | Lateral Tilt Left | Medium |
    | TLR | Lateral Tilt Right | Medium |

    ### Engineered Features (8 biomechanical angles)
    | Feature | Description |
    |---------|-------------|
    | neck_angle | Angle at shoulder between ear and hip (forward head indicator) |
    | shoulder_tilt | Left-right shoulder height difference (lateral imbalance) |
    | spine_lean_angle | Spine deviation from vertical |
    | forward_head_offset | Horizontal ear-shoulder displacement |
    | shoulder_hip_alignment | Torso vertical alignment |
    | ear_shoulder_dist | Normalised ear-to-shoulder gap |
    | shoulder_symmetry | Left/right body length ratio (0-1) |
    | lateral_lean | Horizontal shoulder-hip offset |

    ### ML Models Compared (CLO3)
    | Model | Role |
    |-------|------|
    | Logistic Regression | Interpretable baseline |
    | Random Forest | Feature importance, robust to noise |
    | SVM (RBF kernel) | Non-linear boundaries, good generalisation |
    | Gradient Boosting | Highest accuracy, handles class imbalance |

    ### Health Risk Scoring
    | Threshold | Risk Level |
    |-----------|-----------|
    | < 20 mins poor posture | 🟢 Low |
    | 20–45 mins or 30–60% of session | 🟡 Medium |
    | > 45 mins or > 60% of session | 🔴 High |

    ### Technologies
    Python · MediaPipe · scikit-learn · OpenCV · Streamlit · Matplotlib · Pandas

    ### Government Innovation Alignment
    Aligns with national occupational health policy targeting workplace-related
    musculoskeletal disorders — a leading cause of disability and healthcare costs globally.

    ---
    *AI Assignment 3 · Healthcare & Bioinformatics · Smart Posture Detection*
    """)
