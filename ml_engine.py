"""
ml_engine.py
All AI/ML — runs on real simulation data, not fake numbers.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, fftfreq
from scipy.stats import zscore
from scipy.signal import find_peaks

SENSORS = ["pressure", "flow", "temp", "rpm", "vib", "eff"]


# ── Anomaly Detector ─────────────────────────────────────────
class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = IsolationForest(contamination=0.06, random_state=42, n_estimators=100)
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        X = self.scaler.fit_transform(df[SENSORS])
        self.model.fit(X)
        self.fitted = True

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[SENSORS])
        raw  = self.model.decision_function(X)
        isf  = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        zs   = np.clip(np.abs(zscore(X, axis=0)).max(axis=1) / 5.0, 0, 1)
        return np.round((isf + zs) / 2, 3)

    def score_single(self, row: dict) -> float:
        df = pd.DataFrame([row])
        return float(self.score(df)[0])


# ── PCA Health ───────────────────────────────────────────────
class PCAEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=2)
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X  = self.scaler.fit_transform(df[SENSORS])
        Xp = self.pca.fit_transform(X)
        self.fitted = True
        out = df[["fault_mode"]].copy().reset_index(drop=True)
        out["PC1"] = Xp[:, 0]
        out["PC2"] = Xp[:, 1]
        return out

    def transform_single(self, row: dict) -> tuple:
        df = pd.DataFrame([row])
        X  = self.scaler.transform(df[SENSORS])
        Xp = self.pca.transform(X)
        return float(Xp[0, 0]), float(Xp[0, 1])

    @property
    def explained(self):
        return self.pca.explained_variance_ratio_

    def loadings(self) -> pd.DataFrame:
        return pd.DataFrame(self.pca.components_.T, index=SENSORS, columns=["PC1", "PC2"])


# ── FFT Vibration ─────────────────────────────────────────────
class FFTEngine:
    def __init__(self, fs: float = 100.0):
        self.fs = fs

    def analyze(self, signal: np.ndarray) -> dict:
        N   = len(signal)
        sig = signal - signal.mean()
        yf  = np.abs(fft(sig))[:N // 2]
        xf  = fftfreq(N, 1 / self.fs)[:N // 2]

        peaks, _ = find_peaks(yf, height=yf.max() * 0.1, distance=3)
        dom_idx  = peaks[np.argmax(yf[peaks])] if len(peaks) else np.argmax(yf[1:]) + 1
        dom_freq = float(xf[dom_idx])
        dom_amp  = float(yf[dom_idx])

        harm_idx   = int(np.argmin(np.abs(xf - dom_freq * 2)))
        harm_ratio = float(yf[harm_idx] / dom_amp) if dom_amp > 0 else 0.0
        rms        = float(np.sqrt(np.mean(signal ** 2)))

        return dict(
            freqs       = xf.tolist(),
            amps        = yf.tolist(),
            dom_freq    = round(dom_freq, 1),
            dom_amp     = round(dom_amp, 3),
            harm_ratio  = round(harm_ratio, 3),
            rms         = round(rms, 3),
            bearing_fault = harm_ratio > 0.4,
            diagnosis   = self._diagnose(rms, harm_ratio, dom_freq),
        )

    def _diagnose(self, rms, harm_ratio, dom_freq):
        if harm_ratio > 0.4:  return "Bearing wear — elevated 2× harmonic"
        if rms > 5.0:         return "High RMS — imbalance or misalignment"
        if dom_freq > 80:     return "High-freq spike — possible cavitation"
        return "Vibration within normal range"


# ── RUL Predictor ────────────────────────────────────────────
class RULEngine:
    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold
        self.model     = LinearRegression()
        self.fitted    = False

    def fit(self, df: pd.DataFrame):
        X = df[["tick"]].values
        y = df["eff"].values
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, current_tick: int, current_eff: float) -> dict:
        slope = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)
        if slope >= 0:
            return dict(rul_h=9999, rul_d=999, degradation=0.0, status="Stable", slope=slope)
        eof_tick = (self.threshold - intercept) / slope
        rul_ticks = max(0, eof_tick - current_tick)
        rul_h = round(rul_ticks * 0.5)
        rul_d = round(rul_h / 24)
        deg   = round((91 - current_eff) / 91 * 100, 1)
        status = "Critical" if rul_d < 7 else ("Warning" if rul_d < 30 else "OK")
        return dict(rul_h=rul_h, rul_d=rul_d, degradation=deg, status=status, slope=slope)

    def forecast(self, current_tick: int, days: int = 30) -> pd.DataFrame:
        future = np.arange(current_tick, current_tick + days * 48, 48).reshape(-1, 1)
        eff    = np.clip(self.model.predict(future), 60, 95)
        return pd.DataFrame({"day": range(len(future)), "eff": np.round(eff, 2)})
