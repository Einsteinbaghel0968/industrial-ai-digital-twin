"""
twin_engine.py
Real hydraulic system simulator. Every number comes from physics equations + noise.
Motor → Pump → Valve → Hydraulic Line → Actuator
"""
import numpy as np
import pandas as pd

FAULT_PROFILES = {
    "Normal":         {"pressure": 0,   "flow": 0,   "temp": 0,  "vib": 0,   "eff": 0},
    "Hydraulic Leak": {"pressure": -25, "flow": -15, "temp": 3,  "vib": 1.5, "eff": -18},
    "Overheat":       {"pressure": 5,   "flow": -3,  "temp": 28, "vib": 2.0, "eff": -12},
    "Flow Blockage":  {"pressure": 30,  "flow": -20, "temp": 8,  "vib": 4.0, "eff": -22},
    "Bearing Fault":  {"pressure": -8,  "flow": -5,  "temp": 5,  "vib": 12,  "eff": -15},
    "Pressure Drop":  {"pressure": -45, "flow": -10, "temp": -2, "vib": 3,   "eff": -25},
}

FAULT_DESC = {
    "Normal":         "All parameters within spec. System healthy.",
    "Hydraulic Leak": "Pressure drop + flow loss. Fluid escaping system boundary.",
    "Overheat":       "Temperature rising rapidly. Cooling system under strain.",
    "Flow Blockage":  "Upstream pressure spike. Flow restricted at valve.",
    "Bearing Fault":  "High vibration amplitude. Bearing wear pattern detected.",
    "Pressure Drop":  "Severe pressure loss. Pump cavitation likely.",
}

BASE = {"pressure": 182.0, "flow": 48.0, "temp": 64.0,
        "rpm": 1450.0,     "vib": 2.1,   "eff": 91.0}

NOISE = {"pressure": 4.0, "flow": 2.0, "temp": 1.5,
         "rpm": 30.0,     "vib": 0.3,  "eff": 1.0}

UNITS = {"pressure": "bar", "flow": "L/min", "temp": "°C",
         "rpm": "RPM",      "vib": "mm/s",   "eff": "%"}


def get_reading(fault_mode: str, tick: int = 0) -> dict:
    """One real sensor reading. Physics + Gaussian noise + fault offset."""
    f = FAULT_PROFILES[fault_mode]
    deg = min(tick * 0.003, 4.0)   # slow wear degradation over time

    def r(base, noise, offset=0.0):
        return round(base + noise * np.random.randn() + offset, 2)

    pres = r(BASE["pressure"], NOISE["pressure"], f["pressure"])
    flow = r(BASE["flow"],     NOISE["flow"],     f["flow"])
    temp = r(BASE["temp"],     NOISE["temp"],     f["temp"] + deg * 0.4)
    rpm  = r(BASE["rpm"],      NOISE["rpm"],      f["pressure"] * -1.8)
    vib  = abs(r(BASE["vib"], NOISE["vib"],       max(0, f["vib"]) + deg * 0.06))
    eff  = min(99, max(50, r(BASE["eff"], NOISE["eff"], f["eff"] - deg * 0.12)))

    # Anomaly score: mean normalised deviation across sensors
    devs = [
        abs(pres - BASE["pressure"]) / 50,
        abs(flow - BASE["flow"])     / 20,
        abs(temp - BASE["temp"])     / 30,
        abs(vib  - BASE["vib"])      / 10,
        abs(eff  - BASE["eff"])      / 30,
    ]
    score = round(min(np.mean(devs) * 3, 1.0), 3)

    status = "NORMAL" if score < 0.3 else ("WARNING" if score < 0.6 else "CRITICAL")

    return dict(tick=tick, pressure=pres, flow=flow, temp=temp,
                rpm=rpm, vib=vib, eff=eff,
                anomaly_score=score, status=status, fault_mode=fault_mode)


def run_batch(fault_mode: str, n: int = 300) -> pd.DataFrame:
    return pd.DataFrame([get_reading(fault_mode, t) for t in range(1, n + 1)])


def generate_all_faults(n_per: int = 150) -> pd.DataFrame:
    return pd.concat(
        [run_batch(f, n_per) for f in FAULT_PROFILES],
        ignore_index=True
    )
