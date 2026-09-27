"""Chronological training and probability checks for pre-fight models.

The earliest events fit the model, the next events calibrate its probabilities,
and the latest events are held out once for evaluation. Saved models deliberately
remain fitted on the training window so their reported test scores apply to the
exact artifacts used for predictions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


MIN_PARTITION_ROWS = 30
REGULARIZATION_C = 0.25
METHOD_CLASSES = {"Decision", "KO/TKO", "Submission"}


def _event_dates(dates):
    parsed = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(
            "Every training fight needs a valid event_date. Repair missing dates "
            "before training; a random split would leak future information."
        )
    return parsed.dt.normalize()


def chronological_split(dates, min_rows=MIN_PARTITION_ROWS):
    """Return positional train/calibration/test indices without splitting dates.

    Boundaries target 60%/20%/20% of fights, subject to whole event dates and a
    minimum row count in each partition. Input order need not be chronological.
    """
    dates = _event_dates(dates)
    counts = dates.value_counts().sort_index()
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")
    if len(counts) < 3 or len(dates) < 3 * min_rows:
        raise ValueError(
            f"Chronological evaluation needs at least {3 * min_rows} fights "
            f"across at least 3 event dates ({min_rows} per partition). "
            "Collect more historical fights before training."
        )

    cumulative = counts.cumsum().to_numpy()
    best = None
    for first in range(len(counts) - 2):
        train_rows = cumulative[first]
        if train_rows < min_rows:
            continue
        candidates = np.arange(first + 1, len(counts) - 1)
        valid = candidates[
            (cumulative[candidates] - train_rows >= min_rows)
            & (len(dates) - cumulative[candidates] >= min_rows)
        ]
        if not len(valid):
            continue
        second = valid[np.argmin(np.abs(cumulative[valid] - len(dates) * 0.8))]
        error = abs(train_rows - len(dates) * 0.6) + abs(cumulative[second] - len(dates) * 0.8)
        if best is None or error < best[0]:
            best = (error, first, second)
    if best is None:
        raise ValueError(
            f"Cannot keep event dates together with {min_rows} fights per partition. "
            "Collect fights from more event dates before training."
        )

    first_date, second_date = counts.index[best[1]], counts.index[best[2]]
    return {
        "train": np.flatnonzero((dates <= first_date).to_numpy()),
        "calibration": np.flatnonzero(((dates > first_date) & (dates <= second_date)).to_numpy()),
        "test": np.flatnonzero((dates > second_date).to_numpy()),
    }


def _validate_split(split, dates, min_rows):
    """Validate an explicitly shared split before either model filters labels."""
    result = {}
    for name in ("train", "calibration", "test"):
        raw = np.asarray(split.get(name, []))
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"The {name} split must contain integer positional indices.")
        positions = raw.astype(int)
        if len(positions) < min_rows or (positions < 0).any() or (positions >= len(dates)).any():
            raise ValueError(f"The {name} split needs at least {min_rows} valid fight indices.")
        result[name] = positions
    combined = np.concatenate(list(result.values()))
    if len(combined) != len(dates) or not np.array_equal(np.sort(combined), np.arange(len(dates))):
        raise ValueError("The chronological split must contain every fight exactly once.")
    if not (
        dates.iloc[result["train"]].max() < dates.iloc[result["calibration"]].min()
        and dates.iloc[result["calibration"]].max() < dates.iloc[result["test"]].min()
    ):
        raise ValueError("Training, calibration, and test event dates must be strictly chronological.")
    return result


@dataclass
class TemperatureModel:
    """Scale fitted logits with one positive temperature, preserving ordering.

    The caller applies the separately saved scaler first. A scalar temperature
    preserves the winner model's fighter-swap symmetry: P(A, B) = 1 - P(B, A).
    Keeping this class in an importable module also makes joblib artifacts work
    when loaded by app.py rather than the training command.
    """

    base_model: LogisticRegression
    temperature: float
    feature_names_in_: np.ndarray

    @property
    def classes_(self):
        return self.base_model.classes_

    @property
    def n_features_in_(self):
        return self.base_model.n_features_in_

    @property
    def coef_(self):
        return self.base_model.coef_ / self.temperature

    @property
    def intercept_(self):
        return self.base_model.intercept_ / self.temperature

    def predict_proba(self, features):
        return _probabilities(self.base_model.decision_function(features), self.temperature)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def _probabilities(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be positive and finite.")
    logits = np.asarray(logits) / temperature
    if logits.ndim == 1:
        positive = expit(logits)
        return np.column_stack((1 - positive, positive))
    return softmax(logits, axis=1)


def _fit_temperature(base_model, features, labels):
    logits = base_model.decision_function(features)

    def objective(log_temperature):
        return log_loss(labels, _probabilities(logits, np.exp(log_temperature)), labels=base_model.classes_)

    # Fixed bounds stop tiny samples from producing arbitrarily extreme scales.
    result = minimize_scalar(objective, bounds=(np.log(0.25), np.log(10.0)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Probability calibration failed. Check calibration data before saving models.")
    return float(np.exp(result.x))


def probability_metrics(labels, probabilities, classes):
    """JSON-safe proper scoring rules plus a ten-bin reliability summary."""
    labels, classes = np.asarray(labels), np.asarray(classes)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = classes[probabilities.argmax(axis=1)]
    one_hot = (labels[:, None] == classes[None, :]).astype(float)
    binary = len(classes) == 2
    if binary:
        bin_probability, observed = probabilities[:, 1], one_hot[:, 1]
        brier = np.mean((bin_probability - observed) ** 2)
    else:
        bin_probability = probabilities.max(axis=1)
        observed = (predictions == labels).astype(float)
        brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    bins, ece = [], 0.0
    bucket = np.minimum((bin_probability * 10).astype(int), 9)
    for number in range(10):
        selected = bucket == number
        count = int(selected.sum())
        mean_probability = float(bin_probability[selected].mean()) if count else None
        observed_frequency = float(observed[selected].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(mean_probability - observed_frequency)
        bins.append({
            "lower": number / 10,
            "upper": (number + 1) / 10,
            "count": count,
            "mean_probability": mean_probability,
            "observed_frequency": observed_frequency,
        })
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "brier_score": float(brier),
        "ece": float(ece),
        "calibration_bins": bins,
        "calibration_kind": "positive_class" if binary else "top_class",
        "brier_definition": "binary squared error" if binary else "sum of squared errors across classes",
    }


def _partition_summary(dates, labels, positions):
    selected_dates = dates.iloc[positions]
    return {
        "start": selected_dates.min().date().isoformat(),
        "end": selected_dates.max().date().isoformat(),
        "rows": int(len(positions)),
        "class_counts": {str(label): int(count) for label, count in labels.iloc[positions].value_counts().items()},
    }


def _train(features, labels, dates, split, winner, min_rows):
    if not isinstance(features, pd.DataFrame) or features.empty or features.columns.duplicated().any():
        raise ValueError("Training features must be a nonempty DataFrame with unique column names.")
    if not all(isinstance(name, str) for name in features.columns):
        raise ValueError("Training feature names must be strings.")
    try:
        features = features.astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError("Training features must be numeric.") from error
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Training features must be finite. Repair missing or infinite values before training.")
    labels = pd.Series(np.asarray(labels)).reset_index(drop=True)
    dates = _event_dates(dates)
    if not (len(features) == len(labels) == len(dates)):
        raise ValueError("Features, labels, and event dates must have matching row counts and order.")
    features = features.reset_index(drop=True)
    split = chronological_split(dates, min_rows=min_rows) if split is None else _validate_split(split, dates, min_rows)
    expected = {0, 1} if winner else METHOD_CLASSES
    if winner and (labels.isna().any() or not set(labels.unique()).issubset(expected)):
        raise ValueError("Winner labels must be 0 or 1 for every fight.")
    if not winner:
        invalid = set(labels.dropna().unique()) - expected
        if invalid:
            raise ValueError("Method labels must be Decision, KO/TKO, Submission, or None for excluded results.")
        # Preserve the winner model's event boundaries when excluding DQs/NCs.
        split = {name: positions[labels.iloc[positions].notna().to_numpy()] for name, positions in split.items()}
    for name, positions in split.items():
        if len(positions) < min_rows:
            raise ValueError(
                f"The {name} partition has only {len(positions)} usable fights; "
                f"at least {min_rows} are required. Collect more history before training."
            )
        if name in ("train", "calibration") and set(labels.iloc[positions].unique()) != expected:
            raise ValueError(
                f"The {name} partition needs examples of every "
                f"{'winner' if winner else 'finish-method'} class. Collect more history before training."
            )
    if winner:
        labels = labels.astype(int)

    train, calibration, test = (split[name] for name in ("train", "calibration", "test"))
    train_features, train_labels = features.iloc[train], labels.iloc[train]
    if winner:
        train_features = pd.concat([train_features, -train_features], ignore_index=True)
        train_labels = pd.concat([train_labels, 1 - train_labels], ignore_index=True)
    scaler = StandardScaler(with_mean=not winner)
    scaled_train = scaler.fit_transform(train_features)
    base_model = LogisticRegression(C=REGULARIZATION_C, max_iter=3000, fit_intercept=not winner, random_state=42)
    # Mirroring is augmentation, not evidence of twice as many independent fights.
    weights = np.full(len(train_labels), 0.5) if winner else None
    base_model.fit(scaled_train, train_labels, sample_weight=weights)
    scaled_calibration = scaler.transform(features.iloc[calibration])
    temperature = _fit_temperature(base_model, scaled_calibration, labels.iloc[calibration])
    model = TemperatureModel(base_model, temperature, features.columns.to_numpy())
    scaled_test = scaler.transform(features.iloc[test])
    test_labels = labels.iloc[test]

    prior = np.array([(labels.iloc[train] == label).mean() for label in model.classes_])
    metrics = {
        "uncalibrated": probability_metrics(test_labels, base_model.predict_proba(scaled_test), model.classes_),
        "calibrated": probability_metrics(test_labels, model.predict_proba(scaled_test), model.classes_),
        "train_prior": probability_metrics(test_labels, np.tile(prior, (len(test), 1)), model.classes_),
    }
    if winner and "elo_diff" in features:
        elo_probability = expit(np.log(10) * features.iloc[test]["elo_diff"].to_numpy() / 400)
        metrics["elo"] = probability_metrics(test_labels, np.column_stack((1 - elo_probability, elo_probability)), model.classes_)

    partitions = {name: _partition_summary(dates, labels, positions) for name, positions in split.items()}
    report = {
        "model": "winner" if winner else "method",
        "protocol": "Chronological whole-date train/calibration/test; no test refit or test-set tuning.",
        "regularization_c": REGULARIZATION_C,
        "temperature": temperature,
        "features": features.columns.tolist(),
        "classes": [item.item() if isinstance(item, np.generic) else item for item in model.classes_],
        "partitions": partitions,
        "trained_through": partitions["train"]["end"],
        "calibrated_through": partitions["calibration"]["end"],
        "held_out_start": partitions["test"]["start"],
        "mirrored_training": winner,
        "metrics": metrics,
    }
    return model, scaler, report


def train_winner(features, labels, dates, split=None, min_rows=MIN_PARTITION_ROWS):
    """Fit a symmetric winner model and evaluate it on unseen future events."""
    return _train(features, labels, dates, split, winner=True, min_rows=min_rows)


def train_method(features, labels, dates, split=None, min_rows=MIN_PARTITION_ROWS):
    """Fit the unconditional finish-method model from order-invariant features."""
    return _train(features, labels, dates, split, winner=False, min_rows=min_rows)
