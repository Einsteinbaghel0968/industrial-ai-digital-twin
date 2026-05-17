"""
app.py — Industrial Digital Twin v3
Pure Python + Streamlit. Simulation drives every chart.
Run: streamlit run app.py
Yash Kumar Baghel | DSEU Delhi
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from twin_engine import (
    get_reading, run_batch, generate_all_faults,
    FAULT_PROFILES, FAULT_DESC, BASE, UNITS
)
from ml_engine import AnomalyDetector, PCAEngine, FFTEngine, RULEngine

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin v3 | Yash Kumar Baghel",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-card .label { font-size: 12px; color: #8b949e; margin-bottom: 4px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #e6edf3; }
    .metric-card .unit  { font-size: 11px; color: #8b949e; }
    .metric-card .delta { font-size: 11px; margin-top: 3px; }
    .status-normal   { color: #3fb950; font-weight: 700; }
    .status-warning  { color: #d29922; font-weight: 700; }
    .status-critical { color: #f85149; font-weight: 700; }
    .alert-box {
        padding: 10px 14px;
        border-radius: 8px;
        margin: 4px 0;
        font-size: 13px;
    }
    .alert-normal   { background: rgba(63,185,80,0.1);  border-left: 3px solid #3fb950; color: #3fb950; }
    .alert-warning  { background: rgba(210,153,34,0.1); border-left: 3px solid #d29922; color: #d29922; }
    .alert-critical { background: rgba(248,81,73,0.1);  border-left: 3px solid #f85149; color: #f85149; }
    .pipeline-comp {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-size: 12px;
        color: #8b949e;
    }
    .pipeline-fault { border-color: #f85149 !important; color: #f85149 !important; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stSelectbox label, .stSlider label, .stRadio label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "pressure": "#58a6ff",
    "flow":     "#3fb950",
    "temp":     "#d29922",
    "rpm":      "#bc8cff",
    "vib":      "#f85149",
    "eff":      "#4fd1c5",
}

STATUS_COLOR = {"NORMAL": "#3fb950", "WARNING": "#d29922", "CRITICAL": "#f85149"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e", size=11),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", showgrid=True),
)


# ── Session state ────────────────────────────────────────────
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "tick" not in st.session_state:
        st.session_state.tick = 0
    if "running" not in st.session_state:
        st.session_state.running = False
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "detector" not in st.session_state:
        st.session_state.detector = AnomalyDetector()
    if "pca" not in st.session_state:
        st.session_state.pca = PCAEngine()
    if "fft" not in st.session_state:
        st.session_state.fft = FFTEngine()
    if "rul" not in st.session_state:
        st.session_state.rul = RULEngine()
    if "train_df" not in st.session_state:
        st.session_state.train_df = None
    if "pca_df" not in st.session_state:
        st.session_state.pca_df = None
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []

init_state()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Digital Twin v3")
    st.markdown("**Yash Kumar Baghel**  \nDSEU Delhi · Mechatronics")
    st.divider()

    fault_mode = st.selectbox(
        "Fault / Operating Mode",
        list(FAULT_PROFILES.keys()),
        index=0,
    )
    st.markdown(f"<div class='alert-box alert-{'normal' if fault_mode=='Normal' else 'critical'}'>{FAULT_DESC[fault_mode]}</div>",
                unsafe_allow_html=True)

    st.divider()
    refresh_rate = st.slider("Refresh rate (sec)", 0.5, 3.0, 1.0, 0.5)
    window = st.slider("History window (ticks)", 30, 200, 80)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start" if not st.session_state.running else "⏸ Pause",
                     use_container_width=True):
            st.session_state.running = not st.session_state.running

    with col2:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.history = []
            st.session_state.tick = 0
            st.session_state.alert_log = []
            st.session_state.running = False

    st.divider()
    st.markdown("**Train AI models**")
    st.caption("Runs simulation for all 6 fault modes, trains anomaly detector, PCA, RUL.")
    if st.button("Train on 900 samples", use_container_width=True):
        with st.spinner("Generating training data..."):
            train_df = generate_all_faults(n_per=150)
            st.session_state.train_df = train_df

        with st.spinner("Training Isolation Forest..."):
            normal_df = train_df[train_df["fault_mode"] == "Normal"]
            st.session_state.detector.fit(normal_df)

        with st.spinner("Running PCA..."):
            pca_df = st.session_state.pca.fit_transform(train_df)
            st.session_state.pca_df = pca_df

        with st.spinner("Fitting RUL model..."):
            st.session_state.rul.fit(normal_df)

        st.session_state.trained = True
        st.success(f"Trained on {len(train_df)} samples · 6 fault classes")

    if st.session_state.trained:
        st.markdown('<span class="status-normal">✓ Models trained</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">⚠ Models not trained yet</span>', unsafe_allow_html=True)

    st.divider()
    page = st.radio("View", [
        "Live Monitor",
        "AI Anomaly Detection",
        "PCA Health Space",
        "FFT Vibration",
        "RUL Prediction",
        "Correlation Analysis",
        "Raw Data"
    ])


# ── Tick simulation ──────────────────────────────────────────
if st.session_state.running:
    st.session_state.tick += 1
    reading = get_reading(fault_mode, st.session_state.tick)

    # ML scoring if trained
    if st.session_state.trained:
        reading["anomaly_score"] = st.session_state.detector.score_single(reading)
        s = reading["anomaly_score"]
        reading["status"] = "CRITICAL" if s >= 0.6 else ("WARNING" if s >= 0.3 else "NORMAL")

    st.session_state.history.append(reading)

    # Alert log
    if reading["status"] != "NORMAL":
        st.session_state.alert_log.insert(0, {
            "tick": reading["tick"],
            "status": reading["status"],
            "score": reading["anomaly_score"],
            "fault": fault_mode,
        })
        st.session_state.alert_log = st.session_state.alert_log[:20]

    time.sleep(refresh_rate)
    st.rerun()

# Get last window of history
hist = st.session_state.history[-window:] if st.session_state.history else []
df_hist = pd.DataFrame(hist) if hist else pd.DataFrame()
last = hist[-1] if hist else None


# ── Header ───────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown("# Industrial Digital Twin")
    st.markdown(f"**Mode:** `{fault_mode}`  •  **Tick:** `{st.session_state.tick}`  •  "
                f"**Status:** " +
                (f'<span class="status-{last["status"].lower()}">{last["status"]}</span>'
                 if last else '<span class="status-warning">NOT STARTED</span>'),
                unsafe_allow_html=True)
with h_col2:
    if st.session_state.running:
        st.markdown('<div style="text-align:right;font-size:13px;color:#3fb950">● LIVE</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right;font-size:13px;color:#8b949e">◼ PAUSED</div>',
                    unsafe_allow_html=True)


# ── KPI Metrics row ──────────────────────────────────────────
m = last if last else {k: BASE[k] for k in ["pressure","flow","temp","rpm","vib","eff"]}
sensor_labels = {
    "pressure": ("Pressure", "bar"),
    "flow": ("Flow Rate", "L/min"),
    "temp": ("Temperature", "°C"),
    "rpm": ("RPM", "rev/min"),
    "vib": ("Vibration", "mm/s"),
    "eff": ("Efficiency", "%"),
}

cols = st.columns(6)
for col, (key, (label, unit)) in zip(cols, sensor_labels.items()):
    val = m[key] if isinstance(m, dict) else BASE[key]
    base_val = BASE[key]
    delta = round(val - base_val, 1)
    delta_color = "#f85149" if abs(delta) > base_val * 0.08 else "#3fb950"
    col.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value" style="color:{COLORS[key]}">{val}</div>
      <div class="unit">{unit}</div>
      <div class="delta" style="color:{delta_color}">
        {"▲" if delta > 0 else "▼"} {abs(delta)} from base
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: LIVE MONITOR
# ═══════════════════════════════════════════════════════════════
if page == "Live Monitor":

    # Pipeline diagram
    st.markdown("### Component Pipeline")
    components = [
        ("⚡", "Motor", "rpm"),
        ("💧", "Pump", "pressure"),
        ("🔄", "Valve", "flow"),
        ("〰", "Hyd. Line", "temp"),
        ("⚙️", "Actuator", "eff"),
    ]
    pcols = st.columns(9)  # 5 comps + 4 arrows
    fault_comp = {
        "Hydraulic Leak": "Pump",
        "Overheat": "Hyd. Line",
        "Flow Blockage": "Valve",
        "Bearing Fault": "Motor",
        "Pressure Drop": "Pump",
    }
    faulty = fault_comp.get(fault_mode, "")

    for i, (icon, name, sensor) in enumerate(components):
        is_fault = (name == faulty and fault_mode != "Normal")
        cls = "pipeline-fault" if is_fault else ""
        val = m[sensor] if isinstance(m, dict) else BASE[sensor]
        pcols[i * 2].markdown(
            f'<div class="pipeline-comp {cls}">'
            f'{icon}<br><b>{name}</b><br>'
            f'<span style="color:{"#f85149" if is_fault else "#58a6ff"}">{val}</span>'
            f'<br>{UNITS[sensor]}</div>',
            unsafe_allow_html=True
        )
        if i < 4:
            pcols[i * 2 + 1].markdown(
                '<div style="text-align:center;font-size:22px;color:#3d444d;padding-top:18px">→</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if not df_hist.empty:
        # Two live charts side by side
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Pressure & Flow")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_hist["tick"], y=df_hist["pressure"],
                                     name="Pressure (bar)", line=dict(color=COLORS["pressure"], width=2)),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=df_hist["tick"], y=df_hist["flow"],
                                     name="Flow (L/min)", line=dict(color=COLORS["flow"], width=2)),
                          secondary_y=True)
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                              legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
            fig.update_yaxes(title_text="bar", secondary_y=False,
                             gridcolor="#21262d", color="#8b949e")
            fig.update_yaxes(title_text="L/min", secondary_y=True,
                             gridcolor="#21262d", color="#8b949e")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Temperature & Vibration")
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(x=df_hist["tick"], y=df_hist["temp"],
                                      name="Temp (°C)", line=dict(color=COLORS["temp"], width=2)),
                           secondary_y=False)
            fig2.add_trace(go.Scatter(x=df_hist["tick"], y=df_hist["vib"],
                                      name="Vibration (mm/s)", line=dict(color=COLORS["vib"], width=2)),
                           secondary_y=True)
            fig2.update_layout(**PLOTLY_LAYOUT, height=280,
                               legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
            fig2.update_yaxes(gridcolor="#21262d", color="#8b949e")
            st.plotly_chart(fig2, use_container_width=True)

        # Efficiency + anomaly score
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Efficiency & RPM")
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(x=df_hist["tick"], y=df_hist["eff"],
                                      name="Efficiency (%)", line=dict(color=COLORS["eff"], width=2)),
                           secondary_y=False)
            fig3.add_trace(go.Scatter(x=df_hist["tick"], y=df_hist["rpm"],
                                      name="RPM", line=dict(color=COLORS["rpm"], width=1.5,
                                      dash="dot")), secondary_y=True)
            fig3.update_layout(**PLOTLY_LAYOUT, height=260,
                               legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
            fig3.update_yaxes(gridcolor="#21262d", color="#8b949e")
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            st.markdown("#### Live Anomaly Score")
            fig4 = go.Figure()
            colors_score = [
                "#3fb950" if s < 0.3 else ("#d29922" if s < 0.6 else "#f85149")
                for s in df_hist["anomaly_score"]
            ]
            fig4.add_trace(go.Scatter(
                x=df_hist["tick"], y=df_hist["anomaly_score"],
                fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
                line=dict(color="#58a6ff", width=2),
                name="Anomaly Score"
            ))
            fig4.add_hline(y=0.6, line_color="#f85149", line_dash="dash",
                           annotation_text="Critical", annotation_font_color="#f85149")
            fig4.add_hline(y=0.3, line_color="#d29922", line_dash="dash",
                           annotation_text="Warning", annotation_font_color="#d29922")
            fig4.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_range=[0, 1])
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("Press **▶ Start** in the sidebar to begin simulation.")

    # Alert log
    if st.session_state.alert_log:
        st.markdown("### Alert Log")
        for a in st.session_state.alert_log[:8]:
            cls = "alert-critical" if a["status"] == "CRITICAL" else "alert-warning"
            st.markdown(
                f'<div class="alert-box {cls}">'
                f'Tick {a["tick"]} — {a["status"]} — Score: {a["score"]} — Fault: {a["fault"]}'
                f'</div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════
# PAGE: AI ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════
elif page == "AI Anomaly Detection":
    st.markdown("### AI Anomaly Detection")
    st.caption("Isolation Forest + Z-score ensemble. Trained on normal operation data only. Scores each tick 0–1.")

    if not st.session_state.trained:
        st.warning("Train models first (sidebar → Train on 900 samples)")
    elif df_hist.empty:
        st.info("Start simulation to see live anomaly scoring.")
    else:
        # Re-score entire history with trained model
        df_scored = df_hist.copy()
        df_scored["ml_score"] = st.session_state.detector.score(df_scored)
        df_scored["ml_status"] = df_scored["ml_score"].apply(
            lambda s: "CRITICAL" if s >= 0.6 else ("WARNING" if s >= 0.3 else "NORMAL")
        )

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current score", f'{df_scored["ml_score"].iloc[-1]:.3f}')
        c2.metric("Critical ticks", int((df_scored["ml_status"] == "CRITICAL").sum()))
        c3.metric("Warning ticks",  int((df_scored["ml_status"] == "WARNING").sum()))
        c4.metric("Normal ticks",   int((df_scored["ml_status"] == "NORMAL").sum()))

        # Score timeline
        fig = go.Figure()
        for status, color in [("NORMAL","#3fb950"),("WARNING","#d29922"),("CRITICAL","#f85149")]:
            mask = df_scored["ml_status"] == status
            fig.add_trace(go.Scatter(
                x=df_scored[mask]["tick"], y=df_scored[mask]["ml_score"],
                mode="markers", marker=dict(color=color, size=5),
                name=status
            ))
        fig.add_hline(y=0.6, line_color="#f85149", line_dash="dash")
        fig.add_hline(y=0.3, line_color="#d29922", line_dash="dash")
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title="Anomaly Score Timeline",
                          legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
        st.plotly_chart(fig, use_container_width=True)

        # Score distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df_scored["ml_score"], nbinsx=40,
            marker_color="#58a6ff", opacity=0.75, name="Score distribution"
        ))
        fig2.add_vline(x=0.3, line_color="#d29922", line_dash="dash", annotation_text="Warning")
        fig2.add_vline(x=0.6, line_color="#f85149", line_dash="dash", annotation_text="Critical")
        fig2.update_layout(**PLOTLY_LAYOUT, height=260, title="Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: PCA HEALTH SPACE
# ═══════════════════════════════════════════════════════════════
elif page == "PCA Health Space":
    st.markdown("### PCA System Health Visualization")
    st.caption("6 sensor streams → 2 principal components. Normal states cluster tightly. Faults scatter outward.")

    if not st.session_state.trained or st.session_state.pca_df is None:
        st.warning("Train models first.")
    else:
        pca_df  = st.session_state.pca_df
        train_df = st.session_state.train_df

        c1, c2, c3 = st.columns(3)
        c1.metric("PC1 variance", f'{st.session_state.pca.explained[0]*100:.1f}%')
        c2.metric("PC2 variance", f'{st.session_state.pca.explained[1]*100:.1f}%')
        c3.metric("Total explained", f'{sum(st.session_state.pca.explained)*100:.1f}%')

        FAULT_COLORS_PCA = {
            "Normal": "#3fb950", "Hydraulic Leak": "#58a6ff",
            "Overheat": "#d29922", "Flow Blockage": "#f85149",
            "Bearing Fault": "#bc8cff", "Pressure Drop": "#ff7b72",
        }

        fig = go.Figure()
        full = pd.concat([train_df.reset_index(drop=True), pca_df[["PC1","PC2"]]], axis=1)

        for fault, color in FAULT_COLORS_PCA.items():
            sub = full[full["fault_mode"] == fault]
            fig.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"],
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.5),
                name=fault
            ))

        # Current state dot
        if last and st.session_state.pca.fitted:
            pc1, pc2 = st.session_state.pca.transform_single(last)
            fig.add_trace(go.Scatter(
                x=[pc1], y=[pc2],
                mode="markers",
                marker=dict(color="white", size=14, symbol="star",
                            line=dict(color="#58a6ff", width=2)),
                name="Current state"
            ))

        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                          xaxis_title=f"PC1 ({st.session_state.pca.explained[0]*100:.1f}%)",
                          yaxis_title=f"PC2 ({st.session_state.pca.explained[1]*100:.1f}%)",
                          legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
        st.plotly_chart(fig, use_container_width=True)

        # Loadings
        st.markdown("#### Sensor Contributions (Loadings)")
        loadings = st.session_state.pca.loadings()
        fig2 = go.Figure()
        x = list(loadings.index)
        fig2.add_trace(go.Bar(x=x, y=loadings["PC1"], name="PC1",
                              marker_color="#58a6ff", opacity=0.8))
        fig2.add_trace(go.Bar(x=x, y=loadings["PC2"], name="PC2",
                              marker_color="#3fb950", opacity=0.8))
        fig2.update_layout(**PLOTLY_LAYOUT, height=260, barmode="group",
                           legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: FFT VIBRATION
# ═══════════════════════════════════════════════════════════════
elif page == "FFT Vibration":
    st.markdown("### FFT Vibration Spectrum Analysis")
    st.caption("Real FFT on vibration data from simulation. Identifies dominant frequencies, harmonics, bearing faults.")

    if len(hist) < 30:
        st.info("Run simulation for at least 30 ticks to see FFT.")
    else:
        vib_signal = np.array([h["vib"] for h in hist])
        result = st.session_state.fft.analyze(vib_signal)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dominant Freq", f'{result["dom_freq"]} Hz')
        c2.metric("RMS Amplitude", f'{result["rms"]} mm/s')
        c3.metric("Harmonic Ratio", f'{result["harm_ratio"]}')
        c4.metric("Bearing Fault", "YES ⚠" if result["bearing_fault"] else "No ✓")

        st.markdown(f"**Diagnosis:** {result['diagnosis']}")

        # FFT spectrum
        freqs = np.array(result["freqs"])
        amps  = np.array(result["amps"])
        mask  = freqs < 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=freqs[mask], y=amps[mask],
            marker_color=[
                "#f85149" if abs(f - result["dom_freq"]) < 2 or
                             abs(f - result["dom_freq"] * 2) < 2
                else "#58a6ff"
                for f in freqs[mask]
            ],
            name="FFT Amplitude"
        ))
        fig.add_vline(x=result["dom_freq"], line_color="#d29922",
                      line_dash="dash", annotation_text=f'Fund: {result["dom_freq"]}Hz')
        fig.add_vline(x=result["dom_freq"] * 2, line_color="#bc8cff",
                      line_dash="dot", annotation_text=f'2x: {result["dom_freq"]*2:.0f}Hz')
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          xaxis_title="Frequency (Hz)",
                          yaxis_title="Amplitude (mm/s)")
        st.plotly_chart(fig, use_container_width=True)

        # Time domain signal
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=vib_signal, mode="lines",
            line=dict(color=COLORS["vib"], width=1.2),
            name="Vibration signal"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=220,
                           xaxis_title="Sample", yaxis_title="mm/s",
                           title="Raw vibration signal (time domain)")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: RUL PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "RUL Prediction":
    st.markdown("### Remaining Useful Life (RUL) Prediction")
    st.caption("Linear regression on efficiency degradation. Predicts hours until efficiency drops below 80%.")

    if not st.session_state.trained:
        st.warning("Train models first.")
    elif len(hist) < 20:
        st.info("Need at least 20 ticks of simulation.")
    else:
        rul = st.session_state.rul.predict(
            st.session_state.tick, float(df_hist["eff"].iloc[-1])
        )
        status_color = {"Critical": "#f85149", "Warning": "#d29922", "OK": "#3fb950", "Stable": "#3fb950"}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RUL Estimate", f'{rul["rul_h"]} h')
        c2.metric("Days remaining", f'{rul["rul_d"]} d')
        c3.metric("Degradation", f'{rul["degradation"]}%')
        c4.metric("Status", rul["status"])

        # Efficiency trend + regression line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["tick"], y=df_hist["eff"],
            mode="markers", marker=dict(color="#58a6ff", size=4, opacity=0.5),
            name="Efficiency data"
        ))

        # Regression line
        tmin, tmax = df_hist["tick"].min(), df_hist["tick"].max()
        t_ext = np.array([[tmin], [tmax + 100]])
        reg_line = np.clip(st.session_state.rul.model.predict(t_ext), 60, 95)
        fig.add_trace(go.Scatter(
            x=[tmin, tmax + 100], y=reg_line.flatten(),
            mode="lines", line=dict(color="#f85149", dash="dash", width=2),
            name="Degradation trend"
        ))
        fig.add_hline(y=80, line_color="#d29922", line_dash="dot",
                      annotation_text="EoL threshold (80%)")
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          xaxis_title="Tick", yaxis_title="Efficiency (%)",
                          legend=dict(bgcolor="#161b22", font=dict(color="#8b949e")))
        st.plotly_chart(fig, use_container_width=True)

        # 30-day forecast
        forecast = st.session_state.rul.forecast(st.session_state.tick, days=30)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=forecast["day"], y=forecast["eff"],
            fill="tozeroy", fillcolor="rgba(63,185,80,0.06)",
            line=dict(color="#3fb950", width=2), name="Predicted efficiency"
        ))
        fig2.add_hline(y=80, line_color="#d29922", line_dash="dot",
                       annotation_text="EoL threshold")
        fig2.update_layout(**PLOTLY_LAYOUT, height=260,
                           xaxis_title="Days ahead", yaxis_title="Efficiency (%)",
                           title="30-Day Efficiency Forecast")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "Correlation Analysis":
    st.markdown("### Sensor Correlation Analysis")
    st.caption("Pearson correlation between all sensor parameters from current simulation run.")

    if df_hist.empty:
        st.info("Run simulation first.")
    else:
        corr = df_hist[["pressure","flow","temp","rpm","vib","eff"]].corr().round(3)

        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
            color_continuous_midpoint=0,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title="Pearson Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter matrix for selected sensors
        st.markdown("#### Sensor Pair Scatter")
        s1 = st.selectbox("Sensor X", ["pressure","flow","temp","rpm","vib","eff"], index=0)
        s2 = st.selectbox("Sensor Y", ["pressure","flow","temp","rpm","vib","eff"], index=4)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_hist[s1], y=df_hist[s2],
            mode="markers",
            marker=dict(color=df_hist["anomaly_score"], colorscale="RdYlGn_r",
                        size=5, opacity=0.7, showscale=True,
                        colorbar=dict(title="Anomaly score")),
            text=df_hist["tick"].apply(lambda t: f"Tick {t}"),
        ))
        r = corr.loc[s1, s2]
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                           xaxis_title=s1, yaxis_title=s2,
                           title=f"r = {r} ({s1} vs {s2})")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: RAW DATA
# ═══════════════════════════════════════════════════════════════
elif page == "Raw Data":
    st.markdown("### Raw Simulation Data")

    if df_hist.empty:
        st.info("Run simulation first.")
    else:
        st.caption(f"{len(df_hist)} ticks in current window")
        st.dataframe(
            df_hist.sort_values("tick", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=400,
        )
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", csv,
            file_name=f"digital_twin_{fault_mode.lower().replace(' ','_')}.csv",
            mime="text/csv"
        )

        if st.session_state.train_df is not None:
            st.markdown("#### Training Dataset (900 samples)")
            st.dataframe(st.session_state.train_df, use_container_width=True, height=300)
            csv2 = st.session_state.train_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Training CSV", csv2,
                               file_name="training_data.csv", mime="text/csv")
