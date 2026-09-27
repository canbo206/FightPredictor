import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np
import psycopg2

import app
import model


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.first = {key: 10.0 for key, _ in model.BASE_FEATURES}
        self.second = {key: 20.0 for key, _ in model.BASE_FEATURES}
        for profile, name, fid in ((self.first, "Fighter One", 1), (self.second, "Fighter Two", 2)):
            profile.update(name=name, fighter_id=fid, age_missing=0.0,
                           reach_missing=0.0, height_missing=0.0, layoff_missing=0.0,
                           rate_coverage_pct=100.0)
        self.first["reach_missing"] = 1.0
        self.first["reach_in"] = 70.0
        self.profiles = {1: self.first, 2: self.second}
        self.winner = Mock(classes_=np.array([0, 1]), coef_=np.ones((1, len(model.FEATURE_NAMES))))
        self.winner.predict_proba.return_value = np.array([[0.25, 0.75]])
        self.method = Mock(classes_=np.array(["Decision", "KO/TKO", "Submission"]))
        self.method.predict_proba.return_value = np.array([[0.6, 0.3, 0.1]])
        self.scaler = Mock()
        self.scaler.transform.side_effect = lambda frame: frame.to_numpy()
        self.method_scaler = Mock()
        self.artifacts = self.winner, self.scaler, self.method, self.method_scaler

    def predict(self, first="Fighter One", second="Fighter Two", rounds=5):
        with patch.object(model, "current_profiles", return_value=self.profiles):
            return model.get_matchup_prediction(*self.artifacts, first, second, rounds)

    def test_output_and_feature_order(self):
        result = self.predict()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["probabilities"], {"fighter1": 0.75, "fighter2": 0.25})
        self.assertEqual(result["predicted_method"], "Decision")
        self.assertEqual(result["tracked_fights"], {"fighter1": 10, "fighter2": 20})
        self.assertEqual(len(result["stats"]), len(model.BASE_FEATURES))
        self.assertEqual(result["stats"][0]["edge"], "fighter2")
        reach = result["stats"][model.BASE_COLS.index("reach_in")]
        self.assertIsNone(reach["fighter1"])
        self.assertEqual(list(self.scaler.transform.call_args.args[0].columns), model.FEATURE_NAMES)
        method_frame = self.method_scaler.transform.call_args.args[0]
        self.assertEqual(list(method_frame.columns), model.METHOD_FEATURE_NAMES)
        self.assertEqual(method_frame.iloc[0]["scheduled_rounds"], 5)

    def test_invalid_matchups_never_load_history(self):
        with patch.object(model, "current_profiles") as profiles:
            for first, second, rounds in [("", "B", 3), (None, "B", 3), ("Same", " same ", 3), ("A", "B", 4), ("A", "B", "3")]:
                with self.subTest(first=first, second=second, rounds=rounds), self.assertRaises(ValueError):
                    model.get_matchup_prediction(*self.artifacts, first, second, rounds)
            profiles.assert_not_called()

    def test_unknown_and_ambiguous_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Fighter not found"):
            self.predict("Not Listed")
        with self.assertRaisesRegex(ValueError, "several fighters"):
            self.predict("Fighter")

    def test_two_partial_names_cannot_resolve_to_same_fighter(self):
        with self.assertRaisesRegex(ValueError, "same fighter"):
            self.predict("One", "Fighter One")

    def test_query_failure_closes_connection(self):
        conn = Mock()
        conn.cursor.return_value.__enter__ = Mock(side_effect=psycopg2.ProgrammingError("missing table"))
        conn.cursor.return_value.__exit__ = Mock()
        with patch.object(model, "get_connection", return_value=conn), self.assertRaises(psycopg2.ProgrammingError):
            model.load_fight_data()
        conn.close.assert_called_once()

    def test_terminal_displays_shared_result(self):
        result = self.predict()
        output = io.StringIO()
        with patch.object(model, "get_matchup_prediction", return_value=result), redirect_stdout(output):
            model.predict_matchup(*self.artifacts, "Fighter One", "Fighter Two", 5)
        self.assertIn("75.0%", output.getvalue())
        self.assertIn("Predicted method: Decision", output.getvalue())
        self.assertNotIn("Confidence:", output.getvalue())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.handler = object.__new__(app.Handler)
        self.handler.send_json = Mock()
        app.load_models.cache_clear()

    def tearDown(self):
        app.load_models.cache_clear()

    def test_database_error_has_actionable_response(self):
        self.handler.run_api(Mock(side_effect=psycopg2.OperationalError("private connection details")))
        status, body = self.handler.send_json.call_args.args
        self.assertEqual(status, 503)
        self.assertIn("Start your database", body["error"])
        self.assertNotIn("private connection details", body["error"])

    def test_models_missing_or_wrong_feature_order(self):
        with patch.object(app.joblib, "load", side_effect=FileNotFoundError()), self.assertRaises(app.ModelUnavailable):
            app.load_models()
        artifact = Mock(feature_version_=model.FEATURE_VERSION,
                        feature_names_in_=list(reversed(model.FEATURE_NAMES)))
        with patch.object(app.joblib, "load", return_value=artifact), self.assertRaises(app.ModelUnavailable):
            app.load_models()

    def test_old_feature_version_is_rejected(self):
        artifact = Mock(feature_version_="1", feature_names_in_=model.FEATURE_NAMES)
        with patch.object(app.joblib, "load", return_value=artifact), self.assertRaises(app.ModelUnavailable):
            app.load_models()

    def test_api_returns_prediction_without_retraining(self):
        self.handler.path = "/api/predict"
        payload = json.dumps({"fighter1": "A", "fighter2": "B", "rounds": 3}).encode()
        self.handler.headers = Mock()
        self.handler.headers.get_content_type.return_value = "application/json"
        self.handler.headers.get.return_value = str(len(payload))
        self.handler.rfile = io.BytesIO(payload)
        with patch.object(app, "load_models", return_value=(1, 2, 3, 4)), patch.object(model, "get_matchup_prediction", return_value={"ok": True}) as predict, patch.object(model, "train_and_save") as train:
            self.handler.do_POST()
            predict.assert_called_once_with(1, 2, 3, 4, "A", "B", 3)
            train.assert_not_called()
            self.handler.send_json.assert_called_once_with(200, {"ok": True})

    def test_malformed_json_is_rejected(self):
        self.handler.path = "/api/predict"
        self.handler.headers = Mock()
        self.handler.headers.get_content_type.return_value = "application/json"
        self.handler.headers.get.return_value = "3"
        self.handler.rfile = io.BytesIO(b"bad")
        self.handler.do_POST()
        self.assertEqual(self.handler.send_json.call_args.args[0], 400)


if __name__ == "__main__":
    unittest.main()
