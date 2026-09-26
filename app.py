"""Local web interface for the existing predictor. Run with: python app.py."""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import psycopg2

import model


WEB_DIR = Path(__file__).resolve().parent / "web"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


class ModelUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def load_models():
    """Load trusted, locally trained artifacts once; never retrain on a request."""
    try:
        artifacts = tuple(joblib.load(model.MODEL_DIR / name) for name in (
            "ufc_model.pkl", "ufc_scaler.pkl", "ufc_method_model.pkl", "ufc_method_scaler.pkl"))
        expected = (len(model.FEATURES), len(model.FEATURES), len(model.FEATURES) + 1, len(model.FEATURES) + 1)
        if any(getattr(artifact, "n_features_in_", None) != size for artifact, size in zip(artifacts, expected)):
            raise ValueError("Saved model feature counts do not match the predictor.")
        return artifacts
    except Exception as exc:
        raise ModelUnavailable("Saved models are missing or incompatible. Run python model.py to train them, then restart the website.") from exc


def list_fighters():
    conn = model.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM v_fighter_metrics ORDER BY name")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    def send_content(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, payload):
        self.send_content(status, json.dumps(payload, allow_nan=False).encode(), "application/json; charset=utf-8")

    def run_api(self, action):
        try:
            self.send_json(200, action())
        except ModelUnavailable as exc:
            self.send_json(503, {"error": str(exc)})
        except psycopg2.OperationalError:
            self.send_json(503, {"error": "Cannot connect to PostgreSQL. Start your database and check the connection settings in model.py, then retry."})
        except psycopg2.ProgrammingError:
            self.send_json(503, {"error": "The prediction views are unavailable. Complete the database setup in README.md, then retry."})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            logging.exception("Prediction request failed")
            self.send_json(500, {"error": "The prediction could not be completed. Check the terminal running app.py for details."})

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/fighters":
            self.run_api(lambda: {"fighters": list_fighters()})
        elif path in ASSETS:
            filename, content_type = ASSETS[path]
            self.send_content(200, (WEB_DIR / filename).read_bytes(), content_type)
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            self.send_json(404, {"error": "Page not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "Send the matchup as JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Send a matchup object.")
        except (ValueError, UnicodeError):
            self.send_json(400, {"error": "Invalid matchup request."})
            return
        self.run_api(lambda: model.get_matchup_prediction(
            *load_models(), payload.get("fighter1"), payload.get("fighter2"), payload.get("rounds", 3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"FightPredictor is ready at http://127.0.0.1:{args.port}", flush=True)
    print("Press Ctrl+C to stop. Predictions use your saved models and local PostgreSQL data.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
