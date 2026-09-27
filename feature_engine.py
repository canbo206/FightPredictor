"""Shared, point-in-time UFC features for training and live predictions.

All records on an event date are observed together, *after* every prediction on
that date. Ratings and statistics therefore never use the fight being predicted
or a result from another fight that day. These are tracked-history statistics;
the database does not necessarily contain a fighter's entire career.

The shrinkage assumptions below are deliberately fixed, transparent reference
values, not estimates from the full dataset. Fifteen prior minutes stabilize
rate features, 40 prior strike attempts / four prior takedown attempts stabilize
percentages, and four prior results stabilize form. Recent form has a 730-day
half-life. These are modeling choices to evaluate, not proven betting edges.

Static roster attributes are not timestamped in the source database. We use the
earliest supplied row for each fighter's name, DOB, height and reach, assuming
those attributes were already known. No fight statistics or results are filled
from future rows. Missing attributes have explicit indicator features.
"""

from dataclasses import dataclass, field
import math
import re

import pandas as pd


FEATURE_VERSION = "2"

# Positive differences mean a larger value for fighter 1, not necessarily an
# advantage. The trained model determines how each difference affects a matchup.
BASE_FEATURES = [
    ("sig_landed_per_min", "Sig. strikes landed / min"),
    ("sig_absorbed_per_min", "Sig. strikes absorbed / min"),
    ("sig_attempts_per_min", "Sig. strike attempts / min"),
    ("sig_net_per_min", "Net sig. strikes / min"),
    ("sig_accuracy_pct", "Sig. strike accuracy (%)"),
    ("sig_defense_pct", "Sig. strike defense (%)"),
    ("td_landed_per15", "Takedowns landed / 15 min"),
    ("td_accuracy_pct", "Takedown accuracy (%)"),
    ("td_defense_pct", "Takedown defense (%)"),
    ("td_absorbed_per15", "Takedowns conceded / 15 min"),
    ("control_pct", "Control time (%)"),
    ("kd_per15", "Knockdowns / 15 min"),
    ("kd_absorbed_per15", "Knockdowns absorbed / 15 min"),
    ("sub_attempts_per15", "Submission attempts / 15 min"),
    ("recent_sig_net_per_min", "Recent net sig. strikes / min"),
    ("recent_win_pct", "Recent tracked win rate (%)"),
    ("win_pct", "Tracked win rate (%)"),
    ("finish_win_pct", "Tracked finish wins / results (%)"),
    ("finish_loss_pct", "Tracked finish losses / results (%)"),
    ("elo", "Opponent-adjusted Elo rating"),
    ("opponent_elo", "Prior opponent strength (Elo)"),
    ("age_years", "Age (years)"),
    ("reach_in", "Reach (in)"),
    ("height_in", "Height (in)"),
    ("layoff_days", "Days since last tracked fight"),
    ("total_fights", "Prior tracked fights"),
    ("observed_minutes", "Tracked minutes with known duration"),
    ("rate_coverage_pct", "Complete rate-data coverage (%)"),
    ("age_missing", "Age unavailable (1 = missing)"),
    ("reach_missing", "Reach unavailable (1 = missing)"),
    ("height_missing", "Height unavailable (1 = missing)"),
    ("layoff_missing", "Prior fight date unavailable (1 = missing)"),
]
FEATURE_NAMES = [key + "_diff" for key, _ in BASE_FEATURES]

# A finish model needs the *levels* of both fighters' pace and vulnerability.
# Absolute differences alone cannot distinguish two high-output fighters from
# two low-output fighters with the same difference.
METHOD_LEVEL_KEYS = [
    "sig_landed_per_min", "sig_absorbed_per_min", "sig_attempts_per_min",
    "td_landed_per15", "control_pct", "kd_per15", "kd_absorbed_per15",
    "sub_attempts_per15", "finish_win_pct", "finish_loss_pct",
    "age_years", "observed_minutes", "rate_coverage_pct",
]
METHOD_FEATURE_NAMES = (
    ["abs_" + name for name in FEATURE_NAMES]
    + [key + "_sum" for key in METHOD_LEVEL_KEYS]
    + ["scheduled_rounds"]
)

PRIOR_MINUTES = 15.0
PRIOR_RESULTS = 4.0
RECENT_HALF_LIFE_DAYS = 730.0
INITIAL_ELO = 1500.0
ELO_K = 32.0

# Rates are per minute internally; control is seconds per observed minute.
RATE_PRIORS = {
    "sig_l": 3.0, "osig_l": 3.0, "sig_a": 7.5,
    "td_l": 1.5 / 15, "otd_l": 1.5 / 15,
    "ctrl": 12.0, "kd": 0.3 / 15, "okd": 0.3 / 15,
    "sub": 0.5 / 15,
}
ATTEMPT_PAIRS = {
    "sig": ("sig_l", "sig_a", 40.0, 0.45),
    "osig": ("osig_l", "osig_a", 40.0, 0.45),
    "td": ("td_l", "td_a", 4.0, 0.35),
    "otd": ("otd_l", "otd_a", 4.0, 0.35),
}


def _number(value):
    """Return a finite float, preserving the difference between zero/missing."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _date(value):
    date = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(date):
        return None
    return date.tz_localize(None).normalize()


def elapsed_minutes(row):
    """Known elapsed time, only when all rounds' totals appear to be captured.

    The source uses five-minute rounds. A round-2 finish at 2:30 is 7.5 minutes,
    not two complete rounds. Missing/invalid finish times, mismatched captured
    round counts, and nonstandard schedules are excluded, never guessed. The
    captured-round count can establish completeness only with the database's
    unique (fight, fighter, round) constraint and sequential source round rows.
    """
    final_round = _number(row.get("win_round"))
    captured = _number(row.get("rounds"))
    scheduled = _number(row.get("scheduled_rounds"))
    if (
        final_round is None or final_round != int(final_round)
        or scheduled not in (3.0, 5.0)
        or not 1 <= final_round <= scheduled
        or captured != final_round
    ):
        return None
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(row.get("win_time", "")))
    if match is None:
        return None
    minutes, seconds = (int(part) for part in match.groups())
    if seconds >= 60 or minutes * 60 + seconds > 300:
        return None
    elapsed = (final_round - 1) * 300 + minutes * 60 + seconds
    return elapsed / 60 if elapsed > 0 else None


def _known_result(row):
    # SQL boolean expressions return NULL for draws/no contests. Never coerce
    # NULL to a loss or include it in the binary winner target.
    if "winner_id" in row:
        winner = row.get("winner_id")
        if pd.isna(winner):
            return None
        if winner == row["fighter_id"]:
            return 1.0
        if winner in (row["fighter1_id"], row["fighter2_id"]):
            return 0.0
        return None
    result = _number(row.get("won"))
    return result if result in (0.0, 1.0) else None


def _is_finish(method):
    method = str(method).lower()
    return "ko" in method or "submission" in method


@dataclass
class _History:
    metadata: dict
    elo: float = INITIAL_ELO
    fights: int = 0
    decided: int = 0
    wins: float = 0.0
    finish_wins: int = 0
    finish_losses: int = 0
    minutes: float = 0.0
    rate_fights: int = 0
    opponent_sum: float = 0.0
    last_date: object = None
    recent_wins: float = 0.0
    recent_results: float = 0.0
    rates: dict = field(default_factory=lambda: {key: [0.0, 0.0] for key in RATE_PRIORS})
    attempts: dict = field(default_factory=lambda: {key: [0.0, 0.0] for key in ATTEMPT_PAIRS})
    recent_rates: dict = field(default_factory=lambda: {key: [0.0, 0.0] for key in ("sig_l", "osig_l")})

    def recent_weight(self, date):
        if self.last_date is None:
            return 1.0
        return 0.5 ** ((date - self.last_date).days / RECENT_HALF_LIFE_DAYS)

    def observe(self, row, opponent_elo):
        date = row["event_date"]
        weight = self.recent_weight(date)
        self.recent_wins *= weight
        self.recent_results *= weight
        for totals in self.recent_rates.values():
            totals[0] *= weight
            totals[1] *= weight

        self.fights += 1
        self.opponent_sum += opponent_elo
        self.last_date = date
        won = _known_result(row)
        if won is not None:
            self.decided += 1
            self.wins += won
            self.recent_results += 1
            self.recent_wins += won
            if _is_finish(row.get("win_method")):
                self.finish_wins += int(won == 1)
                self.finish_losses += int(won == 0)

        values = {key: _number(row.get(key)) for key in set(RATE_PRIORS) | {
            column for pair in ATTEMPT_PAIRS.values() for column in pair[:2]
        }}
        values = {key: value if value is not None and value >= 0 else None
                  for key, value in values.items()}
        for key, (landed, attempted, _, _) in ATTEMPT_PAIRS.items():
            successes, opportunities = values[landed], values[attempted]
            if successes is not None and opportunities is not None:
                if successes <= opportunities:
                    self.attempts[key][0] += successes
                    self.attempts[key][1] += opportunities
                else:
                    values[landed] = values[attempted] = None

        minutes = elapsed_minutes(row)
        if minutes is None:
            return
        self.minutes += minutes
        if values["ctrl"] is not None and values["ctrl"] > minutes * 60:
            values["ctrl"] = None
        if all(values[key] is not None for key in values):
            self.rate_fights += 1
        for key in RATE_PRIORS:
            value = values[key]
            if value is None:
                continue
            self.rates[key][0] += value
            self.rates[key][1] += minutes
            if key in self.recent_rates:
                self.recent_rates[key][0] += value
                self.recent_rates[key][1] += minutes

    def profile(self, date):
        def rate(key, recent=False):
            totals = self.recent_rates[key] if recent else self.rates[key]
            weight = self.recent_weight(date) if recent else 1.0
            numerator, minutes = (value * weight for value in totals)
            return (numerator + PRIOR_MINUTES * RATE_PRIORS[key]) / (minutes + PRIOR_MINUTES)

        def accuracy(key):
            landed, attempted = self.attempts[key]
            _, _, prior_attempts, prior_accuracy = ATTEMPT_PAIRS[key]
            return 100 * (landed + prior_attempts * prior_accuracy) / (attempted + prior_attempts)

        reach = _number(self.metadata.get("reach_in"))
        height = _number(self.metadata.get("height_in"))
        reach = reach if reach is not None and reach > 0 else None
        height = height if height is not None and height > 0 else None
        dob = _date(self.metadata.get("dob"))
        age = (date - dob).days / 365.25 if dob is not None else None
        age = age if age is not None and 15 <= age <= 65 else None
        recent_weight = self.recent_weight(date)
        values = {
            "sig_landed_per_min": rate("sig_l"),
            "sig_absorbed_per_min": rate("osig_l"),
            "sig_attempts_per_min": rate("sig_a"),
            "sig_net_per_min": rate("sig_l") - rate("osig_l"),
            "sig_accuracy_pct": accuracy("sig"),
            "sig_defense_pct": 100 - accuracy("osig"),
            "td_landed_per15": rate("td_l") * 15,
            "td_accuracy_pct": accuracy("td"),
            "td_defense_pct": 100 - accuracy("otd"),
            "td_absorbed_per15": rate("otd_l") * 15,
            "control_pct": rate("ctrl") / 60 * 100,
            "kd_per15": rate("kd") * 15,
            "kd_absorbed_per15": rate("okd") * 15,
            "sub_attempts_per15": rate("sub") * 15,
            "recent_sig_net_per_min": rate("sig_l", True) - rate("osig_l", True),
            "recent_win_pct": 100 * (self.recent_wins * recent_weight + 0.5 * PRIOR_RESULTS)
                              / (self.recent_results * recent_weight + PRIOR_RESULTS),
            "win_pct": 100 * (self.wins + 0.5 * PRIOR_RESULTS) / (self.decided + PRIOR_RESULTS),
            "finish_win_pct": 100 * (self.finish_wins + 0.25 * PRIOR_RESULTS) / (self.decided + PRIOR_RESULTS),
            "finish_loss_pct": 100 * (self.finish_losses + 0.25 * PRIOR_RESULTS) / (self.decided + PRIOR_RESULTS),
            "elo": self.elo,
            "opponent_elo": (self.opponent_sum + INITIAL_ELO * PRIOR_RESULTS) / (self.fights + PRIOR_RESULTS),
            "age_years": age if age is not None else 30.0,
            "reach_in": reach if reach is not None else 70.0,
            "height_in": height if height is not None else 70.0,
            "layoff_days": float((date - self.last_date).days) if self.last_date is not None else 180.0,
            "total_fights": float(self.fights),
            "observed_minutes": self.minutes,
            "rate_coverage_pct": 100 * self.rate_fights / self.fights if self.fights else 0.0,
            "age_missing": float(age is None),
            "reach_missing": float(reach is None),
            "height_missing": float(height is None),
            "layoff_missing": float(self.last_date is None),
        }
        return {
            "fighter_id": self.metadata["fighter_id"],
            "name": self.metadata.get("name") or str(self.metadata["fighter_id"]),
            **values,
            "sample_size": self.fights,
            "decided_fights": self.decided,
            "rate_fights": self.rate_fights,
            "last_fight_date": self.last_date.date().isoformat() if self.last_date is not None else None,
        }


def _prepare(df):
    if df.empty:
        return [], {}
    required = {"fight_id", "fighter_id", "fighter1_id", "fighter2_id", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing fight-history columns: " + ", ".join(sorted(missing)))
    frame = df.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    if frame["event_date"].isna().any():
        raise ValueError("Every fight-history row needs a valid event_date.")
    if frame.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("Expected one history row per fight and fighter.")
    frame = frame.sort_values(["event_date", "fight_id", "fighter_id"])
    records = frame.to_dict("records")
    histories = {}
    for row in records:
        if row["fighter_id"] not in histories:
            histories[row["fighter_id"]] = _History(metadata={
                key: row.get(key) for key in ("fighter_id", "name", "dob", "reach_in", "height_in")
            })
    return records, histories


def matchup_features(profile1, profile2):
    """Ordered, antisymmetric winner features: fighter 1 minus fighter 2."""
    return {key + "_diff": float(profile1[key]) - float(profile2[key])
            for key, _ in BASE_FEATURES}


def method_features(profile1, profile2, rounds):
    """Ordered, fighter-order-invariant method features with absolute levels."""
    if isinstance(rounds, bool) or rounds not in (3, 5):
        raise ValueError("Scheduled rounds must be 3 or 5.")
    differences = matchup_features(profile1, profile2)
    return {
        **{"abs_" + key: abs(value) for key, value in differences.items()},
        **{key + "_sum": float(profile1[key]) + float(profile2[key]) for key in METHOD_LEVEL_KEYS},
        "scheduled_rounds": float(rounds),
    }


def _dates(records):
    """Yield date batches without assuming an intra-event ordering."""
    from itertools import groupby

    for date, rows in groupby(records, key=lambda row: row["event_date"]):
        yield date, list(rows)


def _observe_date(rows, histories):
    ratings = {fighter_id: history.elo for fighter_id, history in histories.items()}
    fights = {}
    for row in rows:
        fights.setdefault(row["fight_id"], {})[row["fighter_id"]] = row

    # Compute every same-date Elo update from the same prior ratings. Draws
    # affect Elo only when identified explicitly; no contests do not affect it.
    updates = {}
    for fight_rows in fights.values():
        first = next(iter(fight_rows.values()))
        first_id, second_id = first["fighter1_id"], first["fighter2_id"]
        if first_id not in fight_rows or second_id not in fight_rows:
            continue
        row1, row2 = fight_rows[first_id], fight_rows[second_id]
        score, other_score = _known_result(row1), _known_result(row2)
        if score is None or other_score is None:
            if "draw" in str(row1.get("win_method")).lower():
                score = other_score = 0.5
            else:
                continue
        if score + other_score != 1:
            continue
        expected = 1 / (1 + 10 ** ((ratings[second_id] - ratings[first_id]) / 400))
        delta = ELO_K * (score - expected)
        updates[first_id] = updates.get(first_id, 0) + delta
        updates[second_id] = updates.get(second_id, 0) - delta

    for row in rows:
        opponent_id = row["fighter2_id"] if row["fighter_id"] == row["fighter1_id"] else row["fighter1_id"]
        histories[row["fighter_id"]].observe(row, ratings.get(opponent_id, INITIAL_ELO))
    for fighter_id, delta in updates.items():
        histories[fighter_id].elo += delta


def profiles_as_of(df, as_of):
    """Profiles using only results dated strictly before ``as_of``.

    Every fighter present in the input roster gets a profile, including debuts.
    Future rows contribute static roster attributes only, as documented above.
    An entire date is excluded because source data lacks fight start timestamps.
    """
    cutoff = _date(as_of)
    if cutoff is None:
        raise ValueError("A valid prediction date is required.")
    records, histories = _prepare(df)
    for date, rows in _dates(records):
        if date >= cutoff:
            break
        _observe_date(rows, histories)
    return {fighter_id: history.profile(cutoff) for fighter_id, history in histories.items()}


def build_training_data(df):
    """Return winner X/y and aligned metadata, including method input columns.

    Unlabelled bouts still supply historical statistics but are not treated as
    losses. All fights on one date receive profiles from before that date.
    """
    records, histories = _prepare(df)
    features, labels, metadata, fight_ids = [], [], [], []
    for date, rows in _dates(records):
        snapshots = {row["fighter_id"]: histories[row["fighter_id"]].profile(date) for row in rows}
        fights = {}
        for row in rows:
            fights.setdefault(row["fight_id"], {})[row["fighter_id"]] = row
        for fight_id, fight_rows in fights.items():
            first = next(iter(fight_rows.values()))
            first_id, second_id = first["fighter1_id"], first["fighter2_id"]
            if first_id not in fight_rows or second_id not in fight_rows:
                continue
            row1, row2 = fight_rows[first_id], fight_rows[second_id]
            score, other_score = _known_result(row1), _known_result(row2)
            if score is None or other_score is None or score + other_score != 1:
                continue
            rounds = _number(row1.get("scheduled_rounds"))
            if rounds not in (3.0, 5.0):
                continue
            profile1, profile2 = snapshots[first_id], snapshots[second_id]
            fight_ids.append(fight_id)
            features.append(matchup_features(profile1, profile2))
            labels.append(int(score))
            metadata.append({
                "event_date": date,
                "event_id": row1.get("event_id"),
                "win_method": row1.get("win_method"),
                **method_features(profile1, profile2, int(rounds)),
            })
        _observe_date(rows, histories)

    index = pd.Index(fight_ids, name="fight_id")
    return (
        pd.DataFrame(features, index=index, columns=FEATURE_NAMES, dtype=float),
        pd.Series(labels, index=index, name="won", dtype=int),
        pd.DataFrame(metadata, index=index, columns=["event_date", "event_id", "win_method", *METHOD_FEATURE_NAMES]),
    )
