import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import psycopg2

import api
import database


class FastApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_docs_health_and_existing_website(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertIn("/api/predict", self.client.get("/openapi.json").json()["paths"])
        self.assertIn("matchup-form", self.client.get("/").text)
        self.assertEqual(self.client.get("/database.py").status_code, 404)

    def test_prediction_reuses_engine_without_training(self):
        with patch.object(api.legacy, "load_models", return_value=(1, 2, 3, 4)), \
             patch.object(api.model, "get_matchup_prediction", return_value={"winner": "A"}) as predict, \
             patch.object(api.model, "train_and_save") as train:
            result = self.client.post("/api/predict", json={"fighter1": " A ", "fighter2": "B"})
            self.assertEqual(result.json(), {"winner": "A"})
            predict.assert_called_once_with(1, 2, 3, 4, "A", "B", 3)
            train.assert_not_called()

    def test_invalid_inputs_never_load_models(self):
        with patch.object(api.legacy, "load_models") as load:
            for payload in [{}, {"fighter1": "A", "fighter2": " a "},
                            {"fighter1": "A", "fighter2": "B", "rounds": "3"},
                            {"fighter1": "A", "fighter2": "B", "rounds": True},
                            {"fighter1": "A", "fighter2": "B", "rounds": 4}]:
                self.assertEqual(self.client.post("/api/predict", json=payload).status_code, 422)
            load.assert_not_called()

    def test_dependency_failures_are_actionable_and_do_not_leak_credentials(self):
        with patch.object(api.legacy, "list_fighters", side_effect=psycopg2.OperationalError("secret")):
            response = self.client.get("/api/fighters")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("secret", response.text)
        with patch.object(api.legacy, "load_models", side_effect=api.legacy.ModelUnavailable("Train models first")):
            response = self.client.post("/api/predict", json={"fighter1": "A", "fighter2": "B"})
            self.assertEqual(response.status_code, 503)

    def test_database_url_takes_precedence(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://example/db"}), \
             patch.object(database.psycopg2, "connect") as connect:
            database.get_connection()
            connect.assert_called_once_with("postgresql://example/db", connect_timeout=5)

    def test_pg_environment_overrides_local_defaults(self):
        with patch.dict(os.environ, {"PGHOST": "db", "PGPORT": "5433"}, clear=True), \
             patch.object(database.psycopg2, "connect") as connect:
            database.get_connection()
            self.assertEqual(connect.call_args.kwargs["host"], "db")
            self.assertEqual(connect.call_args.kwargs["port"], "5433")
