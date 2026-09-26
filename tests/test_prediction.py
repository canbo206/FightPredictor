import io
import json
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from unittest.mock import Mock, patch

import numpy as np
import psycopg2

import app
import model


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.conn = Mock()
        self.cur = self.conn.cursor.return_value
        self.s1 = [Decimal("10")] * len(model.FEATURES) + ["Fighter One"]
        self.s2 = [Decimal("20")] * len(model.FEATURES) + ["Fighter Two"]
        self.s1[model.BASE_COLS.index("reach_in")] = None
        self.cur.fetchone.side_effect = [self.s1, self.s2]
        self.winner = Mock(classes_=np.array([0, 1]))
        self.winner.predict_proba.return_value = np.array([[0.25, 0.75]])
        self.method = Mock(classes_=np.array(["Decision", "KO/TKO", "Submission"]))
        self.method.predict_proba.return_value = np.array([[0.6, 0.3, 0.1]])
        self.scaler, self.method_scaler = Mock(), Mock()
        self.artifacts = (self.winner, self.scaler, self.method, self.method_scaler)

    def predict(self, fighter1="Fighter One", fighter2="Fighter Two", rounds=5):
        with patch.object(model, "get_connection", return_value=self.conn):
            return model.get_matchup_prediction(*self.artifacts, fighter1, fighter2, rounds)

    def test_serializable_probabilities_stats_and_feature_order(self):
        result = self.predict()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["probabilities"], {"fighter1": 0.75, "fighter2": 0.25})
        self.assertEqual(result["predicted_method"], "Decision")
        self.assertEqual(result["confidence"], {"score": 5, "label": "MEDIUM"})
        self.assertEqual(len(result["stats"]), len(model.STAT_LABELS))
        by_label = {row["label"]: row for row in result["stats"]}
        self.assertEqual(by_label["Sig Strikes/Rd"]["edge"], "fighter2")
        self.assertEqual(by_label["Sig Absorbed/Rd"]["edge"], "fighter1")
        self.assertIsNone(by_label["Reach (in)"]["fighter1"])
        self.assertIsNone(by_label["Reach (in)"]["edge"])
        self.assertEqual(by_label["Head/Body/Leg"]["fighter1"], "10/10/10")
        frame = self.scaler.transform.call_args.args[0]
        self.assertEqual(list(frame.columns), model.FEATURE_NAMES)
        method_frame = self.method_scaler.transform.call_args.args[0]
        self.assertEqual(method_frame.iloc[0]["scheduled_rounds"], 5)
        self.assertTrue((method_frame.iloc[0][model.FEATURE_NAMES] >= 0).all())
        self.cur.close.assert_called_once()
        self.conn.close.assert_called_once()

    def test_invalid_matchups_never_connect(self):
        with patch.object(model, "get_connection") as connect:
            for first, second, rounds in [("", "B", 3), (None, "B", 3), ("Same", " same ", 3), ("A", "B", 4), ("A", "B", "3")]:
                with self.subTest(first=first, second=second, rounds=rounds), self.assertRaises(ValueError):
                    model.get_matchup_prediction(*self.artifacts, first, second, rounds)
            connect.assert_not_called()

    def test_unknown_fighter_closes_connection(self):
        self.cur.fetchone.side_effect = [None, self.s2]
        with self.assertRaisesRegex(ValueError, "Fighter not found"):
            self.predict()
        self.conn.close.assert_called_once()

    def test_query_failure_closes_connection(self):
        self.cur.execute.side_effect = psycopg2.ProgrammingError("missing view")
        with self.assertRaises(psycopg2.ProgrammingError):
            self.predict()
        self.cur.close.assert_called_once()
        self.conn.close.assert_called_once()

    def test_two_partial_names_cannot_resolve_to_same_fighter(self):
        self.cur.fetchone.side_effect = [self.s1, self.s1]
        with self.assertRaisesRegex(ValueError, "same fighter"):
            self.predict("Fighter", "Fighter One")

    def test_terminal_displays_shared_result(self):
        result = self.predict()
        output = io.StringIO()
        with patch.object(model, "get_matchup_prediction", return_value=result), redirect_stdout(output):
            model.predict_matchup(*self.artifacts, "Fighter One", "Fighter Two", 5)
        self.assertIn("75.0%", output.getvalue())
        self.assertIn("Predicted method: Decision", output.getvalue())
        self.assertIn("Confidence: 5/10 (MEDIUM)", output.getvalue())


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

    def test_models_missing_or_wrong_feature_count(self):
        for kwargs in ({"side_effect": FileNotFoundError()}, {"return_value": Mock(n_features_in_=2)}):
            with self.subTest(kwargs=kwargs), patch.object(app.joblib, "load", **kwargs), self.assertRaises(app.ModelUnavailable):
                app.load_models()

    def test_api_returns_prediction_without_retraining(self):
        self.handler.path = "/api/predict"
        payload = json.dumps({"fighter1": "A", "fighter2": "B", "rounds": 3}).encode()
        self.handler.headers = Mock()
        self.handler.headers.get_content_type.return_value = "application/json"
        self.handler.headers.get.return_value = str(len(payload))
        self.handler.rfile = io.BytesIO(payload)
        with patch.object(app, "load_models", return_value=(1, 2, 3, 4)), patch.object(model, "get_matchup_prediction", return_value={"ok": True}) as predict, patch.object(model, "train_model") as train:
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
