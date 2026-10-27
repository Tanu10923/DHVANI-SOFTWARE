"""DHVANI shared pipeline. Used by BOTH training and live prediction (single source of truth).
Developer: T.T"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, iirnotch, filtfilt, lfilter, lfilter_zi, welch
from scipy.stats import skew, kurtosis

CONFIG = {
    "sampling_rate": 250, "channels": 2, "delimiter": ",", "baud": 115200,
    "window_seconds": 2.0, "hop_seconds": 1.0,
    "bandpass": (0.5, 40.0), "notch_hz": 50.0,  # 50 Hz mains (India); use 60 for US
    "adc_max": 16383, "labels": ["YES", "NO"],  # keep in sync with frontend LABELS in index.html
}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


class CausalEEGFilter:
    """Stateful band-pass and mains-notch filter for sample-by-sample display."""
    def __init__(self, cfg=CONFIG):
        self.channels = cfg["channels"]
        self.sos = butter(4, cfg["bandpass"], btype="band", fs=cfg["sampling_rate"], output="sos")
        self.band_zi = [None] * self.channels
        self.notch_enabled = bool(cfg["notch_hz"] and cfg["notch_hz"] < cfg["sampling_rate"] / 2)
        if self.notch_enabled:
            self.b, self.a = iirnotch(cfg["notch_hz"], 30, cfg["sampling_rate"])
            self.notch_zi = [None] * self.channels

    def process(self, sample):
        values = np.asarray(sample, dtype=float)
        if values.shape != (self.channels,):
            raise ValueError(f"Expected {self.channels} EEG channel values")
        filtered = np.empty(self.channels, dtype=float)
        for channel, value in enumerate(values):
            if self.band_zi[channel] is None:
                self.band_zi[channel] = sosfilt_zi(self.sos) * value
            band, self.band_zi[channel] = sosfilt(
                self.sos, np.asarray([value]), zi=self.band_zi[channel]
            )
            output = band
            if self.notch_enabled:
                if self.notch_zi[channel] is None:
                    self.notch_zi[channel] = lfilter_zi(self.b, self.a) * band[0]
                output, self.notch_zi[channel] = lfilter(
                    self.b, self.a, band, zi=self.notch_zi[channel]
                )
            filtered[channel] = output[0]
        return filtered


def samples_per_window(cfg=CONFIG):
    return int(round(cfg["sampling_rate"] * cfg["window_seconds"]))


def hop_samples(cfg=CONFIG):
    return max(1, int(round(cfg["sampling_rate"] * cfg["hop_seconds"])))


def make_windows(x, cfg=CONFIG):
    """Only COMPLETE windows are produced. x: (n, ch)."""
    n, h = samples_per_window(cfg), hop_samples(cfg)
    return [x[i:i + n] for i in range(0, len(x) - n + 1, h)]


def preprocess(w, cfg=CONFIG):
    """DC removal -> band-pass -> notch -> per-channel z-score. w: (n, ch)."""
    fs = cfg["sampling_rate"]
    lo, hi = cfg["bandpass"]
    if hi >= fs / 2:
        raise ValueError("Band-pass upper edge must be below Nyquist (fs/2)")
    w = np.asarray(w, dtype=float)
    if w.shape[0] != samples_per_window(cfg):
        raise ValueError("Incomplete window")
    w = w - w.mean(axis=0)
    w = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=fs, output="sos"), w, axis=0)
    if cfg["notch_hz"] and cfg["notch_hz"] < fs / 2:
        b, a = iirnotch(cfg["notch_hz"], 30, fs)
        w = filtfilt(b, a, w, axis=0)
    return (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-9)


def feature_names(cfg=CONFIG):
    stats = ["mean", "std", "ptp", "rms", "skew", "kurtosis"]
    names = []
    for c in range(cfg["channels"]):
        names += [f"ch{c+1}_{s}" for s in stats]
        names += [f"ch{c+1}_{b}_abs" for b in BANDS] + [f"ch{c+1}_{b}_rel" for b in BANDS]
    return names


def extract_features(pw, cfg=CONFIG):
    """pw: preprocessed (n, ch). Fixed order == feature_names()."""
    fs, feats = cfg["sampling_rate"], []
    for c in range(pw.shape[1]):
        x = pw[:, c]
        feats += [x.mean(), x.std(), np.ptp(x), np.sqrt(np.mean(x ** 2)), skew(x), kurtosis(x)]
        f, p = welch(x, fs=fs, nperseg=min(len(x), fs))
        absb = [float(np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])) for lo, hi in BANDS.values()]
        tot = sum(absb) + 1e-12
        feats += absb + [a / tot for a in absb]
    v = np.asarray(feats, dtype=float)
    if not np.all(np.isfinite(v)) or len(v) != len(feature_names(cfg)):
        raise ValueError("Invalid feature vector")
    return v


def window_to_features(raw_window, cfg=CONFIG):
    return extract_features(preprocess(raw_window, cfg), cfg)
