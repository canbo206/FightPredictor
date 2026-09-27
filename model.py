"""Train and run pre-fight UFC forecasts for the terminal and local website."""

import argparse
from datetime import date
from functools import lru_cache
import json
from pathlib import Path

import joblib
import pandas as pd
import psycopg2

from feature_engine import (
    BASE_FEATURES, FEATURE_NAMES, FEATURE_VERSION, METHOD_FEATURE_NAMES,
    build_training_data, matchup_features, method_features, profiles_as_of,
)
from training import chronological_split, train_winner, train_method


MODEL_DIR = Path(__file__).resolve().parent / "models"
FEATURES = [(key + "_diff", key) for key, _ in BASE_FEATURES]
BASE_COLS = [key for key, _ in BASE_FEATURES]


def get_connection():
    return psycopg2.connect(
        host="localhost", database="ufc_analytics", user="postgres",
        password="ufc123", port="5432", connect_timeout=5,
    )


def load_fight_data():
    """Read paired round totals, outcomes, exposure time, and fighter metadata.

    Include draws/no-contests in observed statistics even though their outcomes
    do not become binary training labels. No database changes are made here.
    """
    query = """
        SELECT f.fight_id, f.event_id, e.event_date, rs.fighter_id,
            f.fighter1_id, f.fighter2_id, f.winner_id,
            (rs.fighter_id = f.winner_id)::int AS won,
            f.win_method, f.scheduled_rounds, f.win_round, f.win_time,
            f.weight_class, a.name, a.reach_in, a.height_in, a.dob, a.stance,
            count(*) AS rounds,
            sum(rs.sig_strikes_landed) AS sig_l, sum(rs.sig_strikes_attempted) AS sig_a,
            sum(rs.total_strikes_landed) AS tot_l,
            sum(rs.head_strikes_landed) AS head_l,
            sum(rs.body_strikes_landed) AS body_l,
            sum(rs.leg_strikes_landed) AS leg_l,
            sum(rs.takedowns_landed) AS td_l, sum(rs.takedowns_attempted) AS td_a,
            sum(rs.submission_attempts) AS sub, sum(rs.reversals) AS rev,
            sum(rs.ctrl_time_seconds) AS ctrl, sum(rs.knockdowns) AS kd,
            sum(opp.sig_strikes_landed) AS osig_l,
            sum(opp.sig_strikes_attempted) AS osig_a,
            sum(opp.takedowns_landed) AS otd_l,
            sum(opp.takedowns_attempted) AS otd_a,
            sum(opp.knockdowns) AS okd
        FROM fights f
        JOIN events e ON e.event_id = f.event_id
        JOIN round_stats rs ON rs.fight_id = f.fight_id
        JOIN round_stats opp ON opp.fight_id = rs.fight_id
            AND opp.round_number = rs.round_number AND opp.fighter_id <> rs.fighter_id
        JOIN fighters a ON a.fighter_id = rs.fighter_id
        GROUP BY f.fight_id, e.event_date, rs.fighter_id,
            a.name, a.reach_in, a.height_in, a.dob, a.stance
        ORDER BY e.event_date, f.fight_id, rs.fighter_id
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return pd.DataFrame(cur.fetchall(), columns=[item[0] for item in cur.description])
    finally:
        conn.close()


def build_features(df):
    return build_training_data(df)


def method_class(value):
    if not isinstance(value, str):
        return None
    if "Submission" in value:
        return "Submission"
    if "KO" in value:
        return "KO/TKO"
    if "Decision" in value:
        return "Decision"
    return None


def train_model(features, labels, meta):
    """Train and calibrate on separate past event windows."""
    return train_winner(features, labels, meta["event_date"])


def train_method_model(features, meta, split=None):
    method_frame = meta[METHOD_FEATURE_NAMES].astype(float)
    labels = meta["win_method"].map(method_class)
    return train_method(method_frame, labels, meta["event_date"], split=split)


@lru_cache(maxsize=1)
def current_profiles(as_of):
    """Cache histories for this date; restart the server after updating the DB."""
    return profiles_as_of(load_fight_data(), as_of)


def resolve_fighter(profiles, name):
    candidates = [profile for profile in profiles.values()
                  if name.casefold() in profile["name"].casefold()]
    exact = [profile for profile in candidates if name.casefold() == profile["name"].casefold()]
    if exact:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"Fighter not found: {name}")
    raise ValueError(f"'{name}' matches several fighters. Enter a full name from the list.")


def validate_matchup(fighter1, fighter2, rounds):
    if not isinstance(fighter1, str) or not isinstance(fighter2, str):
        raise ValueError("Enter two fighter names.")
    fighter1, fighter2 = fighter1.strip(), fighter2.strip()
    if not fighter1 or not fighter2:
        raise ValueError("Enter two fighter names.")
    if len(fighter1) > 120 or len(fighter2) > 120:
        raise ValueError("Fighter names must be 120 characters or fewer.")
    if type(rounds) is not int or rounds not in (3, 5):
        raise ValueError("Scheduled rounds must be 3 or 5.")
    if fighter1.casefold() == fighter2.casefold():
        raise ValueError("Choose two different fighters.")
    return fighter1, fighter2


def get_matchup_prediction(model, scaler, method_model, method_scaler, fighter1, fighter2, rounds=3):
    """Use the exact same history/feature code used for backtesting."""
    fighter1, fighter2 = validate_matchup(fighter1, fighter2, rounds)
    as_of = date.today().isoformat()
    profiles = current_profiles(as_of)
    first = resolve_fighter(profiles, fighter1)
    second = resolve_fighter(profiles, fighter2)
    if first["fighter_id"] == second["fighter_id"]:
        raise ValueError("Both names matched the same fighter. Choose two different fighters.")

    feature_row = pd.DataFrame([matchup_features(first, second)], columns=FEATURE_NAMES)
    prob = dict(zip(model.classes_, model.predict_proba(scaler.transform(feature_row))[0]))
    prob_a, prob_b = float(prob[1]), float(prob[0])
    method_frame = pd.DataFrame([method_features(first, second, rounds)], columns=METHOD_FEATURE_NAMES)
    method_probs = dict(zip(
        map(str, method_model.classes_),
        map(float, method_model.predict_proba(method_scaler.transform(method_frame))[0]),
    ))

    # A learned coefficient determines direction, not a blanket rule such as
    # "younger is always better." This is a contribution, NOT a betting edge.
    contributions = scaler.transform(feature_row)[0] * model.coef_[0]
    stats = []
    missing_flags = {"age_years": "age_missing", "reach_in": "reach_missing",
                     "height_in": "height_missing", "layoff_days": "layoff_missing"}
    for i, (key, label) in enumerate(BASE_FEATURES):
        values = [None if missing_flags.get(key) and p[missing_flags[key]]
                  else float(p[key]) for p in (first, second)]
        contribution = float(contributions[i])
        direction = "even" if abs(contribution) < 1e-10 else (
            "fighter1" if contribution > 0 else "fighter2")
        stats.append({"label": label, "fighter1": values[0], "fighter2": values[1],
                      "edge": direction, "contribution": contribution,
                      "difference": None if None in values else abs(values[0] - values[1])})

    tracked = {"fighter1": int(first["total_fights"]), "fighter2": int(second["total_fights"])}
    warnings = []
    if min(tracked.values()) < 3:
        warnings.append("Limited tracked fight history: estimates rely more heavily on baseline assumptions.")
    if min(first["rate_coverage_pct"], second["rate_coverage_pct"]) < 100:
        warnings.append("Some statistics are incomplete; rates exclude observations where the required totals or duration are unavailable.")
    return {
        "fighter1": first["name"], "fighter2": second["name"], "rounds": rounds,
        "probabilities": {"fighter1": prob_a, "fighter2": prob_b},
        "method_probabilities": method_probs,
        "predicted_method": max(method_probs, key=method_probs.get),
        "stats": stats, "tracked_fights": tracked, "as_of": as_of,
        "feature_version": FEATURE_VERSION, "warnings": warnings,
    }


def predict_matchup(model, scaler, method_model, method_scaler, fighter1, fighter2, rounds=3):
    try:
        result = get_matchup_prediction(
            model, scaler, method_model, method_scaler, fighter1, fighter2, rounds)
    except ValueError as exc:
        print(f"  {exc}")
        return
    print(f"\n{result['fighter1']} vs {result['fighter2']} ({rounds} rounds)")
    for key in ("fighter1", "fighter2"):
        print(f"  {result[key]:<30} {result['probabilities'][key]:.1%}")
    print(f"  Predicted method: {result['predicted_method']}")
    print("  " + " | ".join(f"{key}: {value:.1%}" for key, value in result["method_probabilities"].items()))
    print(f"  Tracked fights: {result['tracked_fights']['fighter1']} / {result['tracked_fights']['fighter2']}")
    print(f"\n  {'Feature':<28} {'F1':>9} {'F2':>9}  Model lean")
    for stat in result["stats"]:
        v1 = "N/A" if stat["fighter1"] is None else f"{stat['fighter1']:.2f}"
        v2 = "N/A" if stat["fighter2"] is None else f"{stat['fighter2']:.2f}"
        lean = "Neutral" if stat["edge"] == "even" else result[stat["edge"]]
        print(f"  {stat['label']:<28} {v1:>9} {v2:>9}  {lean}")
    for warning in result["warnings"]:
        print(f"  {warning}")
    print("  Model lean is not betting value; no bookmaker prices are included.")


def train_and_save():
    raw = load_fight_data()
    if raw.empty:
        raise ValueError("No fight data found. Run the scraper before training.")
    raw = raw[pd.to_datetime(raw["event_date"]) < pd.Timestamp(date.today())]
    features, labels, meta = build_features(raw)
    split = chronological_split(meta["event_date"])
    winner, scaler, winner_report = train_winner(features, labels, meta["event_date"], split=split)
    method, method_scaler, method_report = train_method_model(features, meta, split=split)
    artifacts = (winner, scaler, method, method_scaler)
    for artifact in artifacts:
        artifact.feature_version_ = FEATURE_VERSION
    MODEL_DIR.mkdir(exist_ok=True)
    for artifact, name in zip(artifacts, (
        "ufc_model.pkl", "ufc_scaler.pkl", "ufc_method_model.pkl", "ufc_method_scaler.pkl",
    )):
        temporary = MODEL_DIR / (name + ".tmp")
        joblib.dump(artifact, temporary)
        temporary.replace(MODEL_DIR / name)
    report = {
        "feature_version": FEATURE_VERSION, "winner_features": FEATURE_NAMES,
        "method_features": METHOD_FEATURE_NAMES, "generated_on": date.today().isoformat(),
        "history_start": str(raw["event_date"].min()), "history_end": str(raw["event_date"].max()),
        "winner": winner_report, "method": method_report,
        "market_benchmark": "Not available: no historical bookmaker odds are stored.",
        "deployment": "Uses the evaluated training-window models and later calibration window; test outcomes were not used to fit parameters.",
    }
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    current_profiles.cache_clear()
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true", help="Train, evaluate, save, and exit.")
    args = parser.parse_args()
    models = train_and_save()
    if not args.train_only:
        predict_matchup(*models, "Islam Makhachev", "Dustin Poirier", rounds=5)
        print("\n--- Interactive Fight Predictor ---\nType 'quit' to exit\n")
        while True:
            try:
                f1 = input("Enter Fighter 1 name: ").strip()
                if f1.lower() == "quit":
                    break
                f2 = input("Enter Fighter 2 name: ").strip()
                if f2.lower() == "quit":
                    break
                rounds = 5 if input("Scheduled rounds (3 or 5) [3]: ").strip() == "5" else 3
                predict_matchup(*models, f1, f2, rounds)
            except (EOFError, KeyboardInterrupt):
                break
