import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import training


class ChronologicalTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.features = pd.DataFrame({
            "elo_diff": rng.normal(0, 150, 300),
            "strike_diff": rng.normal(0, 2, 300),
        })
        self.dates = pd.Series(pd.date_range("2020-01-01", periods=60).repeat(5))
        self.labels = pd.Series((np.arange(300) % 3 == 0).astype(int))
        self.methods = pd.Series(np.tile(["Decision", "KO/TKO", "Submission"], 100))

    def test_split_keeps_whole_dates_and_is_independent_of_input_order(self):
        split = training.chronological_split(self.dates)
        self.assertEqual([len(split[name]) for name in split], [180, 60, 60])
        previous_end = None
        for positions in split.values():
            selected_dates = self.dates.iloc[positions]
            if previous_end is not None:
                self.assertLess(previous_end, selected_dates.min())
            previous_end = selected_dates.max()
        permutation = np.random.default_rng(2).permutation(len(self.dates))
        shuffled = self.dates.iloc[permutation].reset_index(drop=True)
        shuffled_split = training.chronological_split(shuffled)
        for name in split:
            self.assertEqual(set(self.dates.iloc[split[name]]), set(shuffled.iloc[shuffled_split[name]]))

    def test_missing_dates_and_small_history_fail_without_random_fallback(self):
        with self.assertRaisesRegex(ValueError, "valid event_date"):
            training.chronological_split(["2020-01-01", None, "bad"])
        with self.assertRaisesRegex(ValueError, "Collect more historical"):
            training.chronological_split(self.dates.iloc[:50])

    def test_winner_probabilities_are_complementary_for_fighter_swap(self):
        model, scaler, report = training.train_winner(self.features, self.labels, self.dates)
        sample = self.features.iloc[:8]
        forward = model.predict_proba(scaler.transform(sample))[:, 1]
        reverse = model.predict_proba(scaler.transform(-sample))[:, 1]
        np.testing.assert_allclose(forward + reverse, 1.0, atol=1e-14)
        np.testing.assert_allclose(model.predict_proba(scaler.transform(sample * 0)), 0.5, atol=1e-14)
        self.assertFalse(scaler.with_mean)
        self.assertEqual(list(model.feature_names_in_), list(self.features.columns))
        self.assertEqual(model.n_features_in_, 2)
        self.assertGreater(model.temperature, 0)
        self.assertEqual(report["partitions"]["test"]["rows"], 60)
        self.assertIn("elo", report["metrics"])
        self.assertEqual(sum(item["count"] for item in report["metrics"]["calibrated"]["calibration_bins"]), 60)
        json.dumps(report, allow_nan=False)

    def test_test_rows_cannot_change_fitted_weights_scaler_or_calibration(self):
        model, scaler, report = training.train_winner(self.features, self.labels, self.dates)
        altered_features, altered_labels = self.features.copy(), self.labels.copy()
        test = training.chronological_split(self.dates)["test"]
        altered_features.iloc[test] *= 1000
        altered_labels.iloc[test] = 1 - altered_labels.iloc[test]
        other_model, other_scaler, other_report = training.train_winner(altered_features, altered_labels, self.dates)
        np.testing.assert_array_equal(model.coef_, other_model.coef_)
        np.testing.assert_array_equal(scaler.scale_, other_scaler.scale_)
        self.assertEqual(model.temperature, other_model.temperature)
        self.assertNotEqual(report["metrics"]["calibrated"]["log_loss"], other_report["metrics"]["calibrated"]["log_loss"])

    def test_single_class_calibration_fails_actionably(self):
        labels = self.labels.copy()
        labels.iloc[training.chronological_split(self.dates)["calibration"]] = 1
        with self.assertRaisesRegex(ValueError, "calibration partition needs examples"):
            training.train_winner(self.features, labels, self.dates)

    def test_method_filtering_preserves_shared_boundaries_and_probability_sum(self):
        split = training.chronological_split(self.dates)
        labels = self.methods.copy()
        labels.iloc[::11] = None
        features = self.features.abs()
        features["scheduled_rounds"] = 3
        model, scaler, report = training.train_method(features, labels, self.dates, split=split)
        np.testing.assert_allclose(model.predict_proba(scaler.transform(features)).sum(axis=1), 1)
        self.assertEqual(set(model.classes_), training.METHOD_CLASSES)
        self.assertEqual(report["trained_through"], self.dates.iloc[split["train"]].max().date().isoformat())
        self.assertEqual(report["held_out_start"], self.dates.iloc[split["test"]].min().date().isoformat())
        self.assertEqual(report["partitions"]["test"]["rows"], int(labels.iloc[split["test"]].notna().sum()))
        self.assertEqual(report["metrics"]["calibrated"]["calibration_kind"], "top_class")
        json.dumps(report, allow_nan=False)

    def test_bad_shared_split_cannot_leak_dates_or_duplicate_rows(self):
        split = training.chronological_split(self.dates)
        split["train"][-1], split["test"][0] = split["test"][0], split["train"][-1]
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            training.train_winner(self.features, self.labels, self.dates, split=split)
        split = training.chronological_split(self.dates)
        split["test"][0] = split["train"][0]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            training.train_winner(self.features, self.labels, self.dates, split=split)

    def test_serialized_model_loads_with_identical_probabilities(self):
        model, scaler, _ = training.train_winner(self.features, self.labels, self.dates)
        sample = scaler.transform(self.features.iloc[:4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            joblib.dump(model, path)
            restored = joblib.load(path)
        np.testing.assert_array_equal(model.predict_proba(sample), restored.predict_proba(sample))
        self.assertEqual(type(restored).__module__, "training")

    def test_proper_scoring_rules_for_known_binary_and_multiclass_values(self):
        binary = training.probability_metrics([0, 1], [[0.75, 0.25], [0.25, 0.75]], [0, 1])
        self.assertAlmostEqual(binary["brier_score"], 0.0625)
        self.assertAlmostEqual(binary["log_loss"], -np.log(0.75))
        self.assertAlmostEqual(binary["ece"], 0.25)
        multiclass = training.probability_metrics(["A", "B"], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], ["A", "B", "C"])
        self.assertAlmostEqual(multiclass["brier_score"], 0.06)
        self.assertAlmostEqual(multiclass["ece"], 0.2)


if __name__ == "__main__":
    unittest.main()
