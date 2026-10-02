"""FastAPI adapter for the existing predictor and website; no training on requests."""

import logging
from typing import Annotated

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator
import psycopg2

import app as legacy
import model


app = FastAPI(title="FightPredictor API", version="1.0.0")
Name = Annotated[str, StringConstraints(strict=True, strip_whitespace=True,
                                       min_length=1, max_length=120)]


class Matchup(BaseModel):
    fighter1: Name
    fighter2: Name
    rounds: int = Field(default=3, strict=True)

    @field_validator("rounds")
    @classmethod
    def valid_rounds(cls, value):
        if value not in (3, 5):
            raise ValueError("Rounds must be 3 or 5.")
        return value

    @model_validator(mode="after")
    def different_fighters(self):
        if self.fighter1.casefold() == self.fighter2.casefold():
            raise ValueError("Choose two different fighters.")
        return self


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse(status_code=422, content={
        "error": "Provide two different fighter names and rounds as the number 3 or 5."})


def run_prediction_action(action):
    try:
        return action()
    except legacy.ModelUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except psycopg2.OperationalError:
        return JSONResponse(status_code=503, content={
            "error": "Cannot connect to PostgreSQL. Check DATABASE_URL or PG* environment settings."})
    except psycopg2.ProgrammingError:
        return JSONResponse(status_code=503, content={
            "error": "Fight history is unavailable. Check the schema setup in README.md."})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        logging.exception("Prediction request failed")
        return JSONResponse(status_code=500, content={"error": "Prediction failed. Check server logs."})


@app.get("/health")
def health():
    """Liveness only: this does not promise database/model readiness."""
    return {"status": "ok"}


@app.get("/api/fighters")
def fighters():
    return run_prediction_action(lambda: {"fighters": legacy.list_fighters()})


@app.post("/api/predict")
def predict(matchup: Matchup):
    return run_prediction_action(lambda: model.get_matchup_prediction(
        *legacy.load_models(), matchup.fighter1, matchup.fighter2, matchup.rounds))


@app.get("/{asset_path:path}", include_in_schema=False)
def website(asset_path: str):
    asset = legacy.ASSETS.get("/" + asset_path)
    if asset is None:
        return JSONResponse(status_code=404, content={"error": "Page not found."})
    filename, content_type = asset
    return FileResponse(legacy.WEB_DIR / filename, media_type=content_type,
                        headers={"X-Content-Type-Options": "nosniff"})
