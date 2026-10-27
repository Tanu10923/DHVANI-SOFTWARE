"""Real dataset building, training, evaluation and prediction. Developer: T.T"""
import json, threading, time, re
from pathlib import Path
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pipeline as P

ROOT = Path(__file__).parent
REC_DIR = ROOT / "data" / "prod" / "recordings"
MODEL_PATH = ROOT / "models" / "prod" / "model.joblib"
REC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_recording(label, rows, cfg, device):
    n = len(list(REC_DIR.glob(f"{label}_*.csv"))) + 1
    rid = f"{label}_{n:02d}"
    arr = np.asarray(rows, dtype=float)
    header = "sample," + ",".join(f"ch{i+1}" for i in range(arr.shape[1]))
    np.savetxt(REC_DIR / f"{rid}.csv", np.column_stack([np.arange(len(arr)), arr]),
               delimiter=",", header=header, comments="", fmt="%.6g")
    meta = dict(label=label, recording_id=rid, datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
                duration_s=len(arr) / cfg["sampling_rate"], sampling_rate=cfg["sampling_rate"],
                channels=arr.shape[1], samples=len(arr), device=device, mode="prod")
    (REC_DIR / f"{rid}.json").write_text(json.dumps(meta, indent=1))
    return meta


def list_recordings():
    return [json.loads(p.read_text()) for p in sorted(REC_DIR.glob("*.json"))]


class ModelService:
    def __init__(self):
        self.bundle = None
        self.training = dict(status="idle", progress=0, stage="", error=None)
        self.results = None
        self.load()

    def load(self):
        try:
            if not MODEL_PATH.exists():
                self.bundle = None
                return False
            b = joblib.load(MODEL_PATH)
            if b["feature_names"] != P.feature_names(b["cfg"]):
                raise ValueError("Saved model feature layout does not match current pipeline")
            unknown = set(b["model"].classes_) - set(P.CONFIG["labels"])
            if unknown:
                raise ValueError(f"Saved model has commands no longer configured ({', '.join(sorted(unknown))}). Retrain the model.")
            self.bundle, self.results = b, b["results"]
            return True
        except Exception as e:
            self.bundle, self.training["error"] = None, f"Model failed to load: {e}"
            return False

    def status(self):
        recs = list_recordings()
        per = {l: sum(r["label"] == l for r in recs) for l in P.CONFIG["labels"]}
        return dict(trained=self.bundle is not None, loaded=self.bundle is not None,
                    version=self.bundle["version"] if self.bundle else None,
                    recordings_per_command=per, recordings=len(recs),
                    training=self.training, results=self.results,
                    n_features=len(P.feature_names()))

    def _set(self, stage, progress):
        self.training.update(stage=stage, progress=progress)

    def train_async(self):
        if self.training["status"] == "running":
            return False
        self.training = dict(status="running", progress=0, stage="Starting", error=None)
        threading.Thread(target=self._train, daemon=True).start()
        return True

    def _train(self):
        try:
            cfg = dict(P.CONFIG)
            self._set("Loading recordings", 5)
            X, y, grp, rec_ids = [], [], [], {}
            recs = list_recordings()
            for m in recs:
                if m["sampling_rate"] != cfg["sampling_rate"] or m["channels"] != cfg["channels"]:
                    raise ValueError(f"{m['recording_id']} was recorded at {m['sampling_rate']} Hz/"
                                     f"{m['channels']} ch; current config differs")
            for i, m in enumerate(recs):
                data = np.genfromtxt(REC_DIR / f"{m['recording_id']}.csv", delimiter=",", skip_header=1)[:, 1:]
                self._set("Preprocessing + windowing + features", 10 + 60 * i // max(1, len(recs)))
                for w in P.make_windows(data, cfg):
                    X.append(P.window_to_features(w, cfg)); y.append(m["label"]); grp.append(m["recording_id"])
            if not X:
                raise ValueError("No complete 2-second windows found in recordings")
            X, y, grp = np.array(X), np.array(y), np.array(grp)
            # Adaptive: train only on configured commands that have at least 2 recordings
            counts = {l: len(set(grp[y == l])) for l in cfg["labels"]}
            labels = [l for l in cfg["labels"] if counts[l] >= 2]
            skipped = [l for l in cfg["labels"] if counts[l] < 2]
            if len(labels) < 2:
                raise ValueError("Need at least 2 commands with 2+ recordings each. "
                                 f"Usable now: {', '.join(labels) or 'none'}")
            keep = np.isin(y, labels)
            X, y, grp = X[keep], y[keep], grp[keep]
            # Held-out TEST = last recording of each command (no window leakage across recordings)
            test_ids = set()
            for l in labels:
                ids = sorted(set(grp[y == l]))
                test_ids.add(ids[-1])
            te = np.isin(grp, list(test_ids))
            self._set("Training", 75)
            clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X[~te], y[~te])
            self._set("Evaluating", 88)
            pred = clf.predict(X[te])
            p, r, f, _ = precision_recall_fscore_support(y[te], pred, labels=labels, average="macro", zero_division=0)
            results = dict(accuracy=float(accuracy_score(y[te], pred)), precision=float(p), recall=float(r), f1=float(f),
                           labels=labels, skipped_labels=skipped, confusion_matrix=confusion_matrix(y[te], pred, labels=labels).tolist(),
                           n_windows=int(len(X)), n_train_windows=int((~te).sum()), n_test_windows=int(te.sum()),
                           n_features=int(X.shape[1]), test_recordings=sorted(test_ids))
            self._set("Saving", 95)
            final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X, y)  # all data
            version = time.strftime("v%Y%m%d-%H%M%S")
            joblib.dump(dict(model=final, cfg=cfg, feature_names=P.feature_names(cfg), version=version,
                             results=results), MODEL_PATH)
            self.load()
            self.training.update(status="done", progress=100, stage="Model ready")
        except Exception as e:
            self.training.update(status="error", error=f"Model training failed: {e}", stage="")

    def predict(self, raw_window):
        if not self.bundle:
            raise RuntimeError("Model not trained")
        v = P.window_to_features(raw_window, self.bundle["cfg"])  # same pipeline as training
        m = self.bundle["model"]
        proba = m.predict_proba([v])[0]
        i = int(np.argmax(proba))
        return dict(label=str(m.classes_[i]), confidence=float(proba[i]))
