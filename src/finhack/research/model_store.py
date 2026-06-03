"""Train, persist, and load Case 4 enhanced models.

Single source of truth for fitting the post-earnings direction classifier.
The training entry points pick LightGBM when available, otherwise fall back
to scaled logistic regression. The artifact written by ``save_trained_model``
is consumed by:

- ``case4_features.predict_enhanced`` → produces signals for the legacy
  heuristic-lane comparison and for ``paper_signals.build_earnings_paper_signals``.
- Any future scripts that want to score new events without re-training.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "case4_enhanced.pkl"
)


@dataclass(slots=True)
class TrainedModel:
    backend: str
    feature_names: list[str]
    feature_means: list[float]
    feature_stds: list[float]
    model_blob: bytes
    train_events: int
    extras: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "feature_names": list(self.feature_names),
            "feature_means": list(self.feature_means),
            "feature_stds": list(self.feature_stds),
            "model_blob": self.model_blob,
            "train_events": int(self.train_events),
            "extras": dict(self.extras),
        }


def _try_lightgbm():  # noqa: ANN202
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception as exc:  # noqa: BLE001
        logger.info("LightGBM not available, will fall back to logistic: %s", exc)
        return None


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    return (x - means) / stds, means, stds


def _fit_lightgbm(x: np.ndarray, y: np.ndarray, *, lgb) -> bytes:
    """Fit LightGBM with a fixed conservative boost budget.

    Earlier versions either:
    - Ran 400 rounds with no real early stopping (``valid_sets=[train_set]``
      is a no-op since training loss is monotonically decreasing), which
      meant the booster always overfit and produced spuriously confident
      probabilities; or
    - Used an internal 15% time-aware holdout for early stopping, which
      triggered too early on a 95-row noisy validation set and collapsed
      the entire output distribution into [0.50, 0.53].

    Both extremes mislead. The first hides that the OOS accuracy uplift
    is negative behind confident-looking calls; the second admits the
    model has no edge but produces no usable signals downstream.

    Compromise: a small fixed budget (80 rounds), modest leaf capacity,
    gentle L1/L2, and explicit acknowledgement that with the current
    feature set the booster's confidence spread is largely a function of
    how long it runs — not actual calibration.
    """
    n = int(len(y))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 23,
        "min_data_in_leaf": max(15, n // 40),
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.02,
        "lambda_l2": 0.02,
        "verbose": -1,
        "num_threads": -1,
    }
    train_set = lgb.Dataset(x, label=y, free_raw_data=False)
    num_rounds = 80 if n >= 200 else 40
    booster = lgb.train(params, train_set, num_boost_round=num_rounds)
    return booster.model_to_string().encode("utf-8")


def _predict_lightgbm(blob: bytes, x: np.ndarray, *, lgb) -> np.ndarray:
    booster = lgb.Booster(model_str=blob.decode("utf-8"))
    return booster.predict(x)


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> bytes:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x, y)
    return pickle.dumps(clf)


def _predict_logistic(blob: bytes, x: np.ndarray) -> np.ndarray:
    clf = pickle.loads(blob)
    return clf.predict_proba(x)[:, 1]


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    extras: dict[str, Any] | None = None,
) -> TrainedModel:
    """Fit a binary classifier on standardized features.

    Picks LightGBM when available; otherwise scaled logistic regression.
    The chosen backend is recorded on the returned ``TrainedModel`` so
    downstream prediction selects the matching path.
    """
    if x_train.size == 0 or y_train.size == 0:
        raise ValueError("Empty training set")
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError("Need at least two classes in training labels")

    x_std, means, stds = _standardize(x_train.astype(float))

    lgb = _try_lightgbm()
    if lgb is not None:
        try:
            blob = _fit_lightgbm(x_std, y_train.astype(int), lgb=lgb)
            return TrainedModel(
                backend="lightgbm",
                feature_names=list(feature_names),
                feature_means=means.tolist(),
                feature_stds=stds.tolist(),
                model_blob=blob,
                train_events=int(len(y_train)),
                extras=dict(extras or {}),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightGBM fit failed (%s); falling back to logistic", exc)

    blob = _fit_logistic(x_std, y_train.astype(int))
    return TrainedModel(
        backend="logistic",
        feature_names=list(feature_names),
        feature_means=means.tolist(),
        feature_stds=stds.tolist(),
        model_blob=blob,
        train_events=int(len(y_train)),
        extras=dict(extras or {}),
    )


def predict_proba(model: TrainedModel, x: np.ndarray) -> np.ndarray:
    """Standardize + score; returns P(positive) per row."""
    means = np.asarray(model.feature_means, dtype=float)
    stds = np.asarray(model.feature_stds, dtype=float)
    stds = np.where(stds < 1e-9, 1.0, stds)
    x_std = (x.astype(float) - means) / stds
    if model.backend == "lightgbm":
        lgb = _try_lightgbm()
        if lgb is None:
            raise RuntimeError("LightGBM artifact loaded but lightgbm not installed")
        return _predict_lightgbm(model.model_blob, x_std, lgb=lgb)
    return _predict_logistic(model.model_blob, x_std)


def predict_label(model: TrainedModel, x: np.ndarray) -> np.ndarray:
    return (predict_proba(model, x) >= 0.5).astype(int)


def save_trained_model(model: TrainedModel, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "backend": model.backend,
                "feature_names": model.feature_names,
                "train_events": model.train_events,
                "extras": model.extras,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_trained_model(path: Path = DEFAULT_ARTIFACT_PATH) -> TrainedModel | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load model artifact %s: %s", path, exc)
        return None
    return TrainedModel(
        backend=str(payload["backend"]),
        feature_names=list(payload["feature_names"]),
        feature_means=list(payload["feature_means"]),
        feature_stds=list(payload["feature_stds"]),
        model_blob=bytes(payload["model_blob"]),
        train_events=int(payload.get("train_events", 0)),
        extras=dict(payload.get("extras", {})),
    )


def predict_with_saved_model(
    *,
    pre_ret: float,
    features: dict[str, Any],
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
) -> tuple[int, float] | None:
    """Score a single event with the persisted model.

    Returns ``(sign, confidence)`` or ``None`` when the artifact is absent.
    Used by ``case4_features.predict_enhanced`` for the heuristic lane and
    by ``paper_signals.build_earnings_paper_signals`` for forward signals.
    """
    model = load_trained_model(artifact_path)
    if model is None:
        return None
    row = []
    for name in model.feature_names:
        if name == "baseline_pre_7d_return_pct":
            row.append(float(pre_ret))
        elif name == "baseline_pre_7d_abs_return":
            row.append(abs(float(pre_ret)))
        else:
            val = features.get(name, 0.0)
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(0.0)
    arr = np.asarray([row], dtype=float)
    proba = predict_proba(model, arr)[0]
    sign = 1 if proba >= 0.5 else -1
    confidence = float(max(proba, 1.0 - proba))
    return int(sign), confidence
