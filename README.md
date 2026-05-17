# Digital Twin v3 — Streamlit App
**Yash Kumar Baghel | DSEU Delhi**

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's actually connected
- Sidebar fault mode → twin_engine.py generates real sensor values
- Every chart reads from session_state history (real simulation data)
- "Train" button runs 900-sample simulation → fits Isolation Forest + PCA + RUL
- Anomaly scores come from trained ML model, not random numbers
- FFT runs on actual vibration data collected from simulation
- RUL regression fits on actual efficiency values

## Pages
| Page | What it does |
|------|-------------|
| Live Monitor | Real-time sensor charts from simulation |
| AI Anomaly Detection | Isolation Forest scores on live data |
| PCA Health Space | 6D→2D health visualization with current state marker |
| FFT Vibration | Real FFT on vibration signal collected from sim |
| RUL Prediction | Regression on efficiency trend → hours/days remaining |
| Correlation Analysis | Pearson matrix from current simulation run |
| Raw Data | Download CSV of any run |
