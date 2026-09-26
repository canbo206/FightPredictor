import psycopg2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent / "models"


# Database connection

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ufc_analytics",
        user="postgres",
        password="ufc123",
        port="5432",
        connect_timeout=5,
    )


# ── Feature definitions ────────────────────────────────────────────────────
# Each feature is a differential: Fighter1's value minus Fighter2's value.
# A positive value means Fighter1 has the edge on that metric.
# (display_name, base_column_in_views)
FEATURES = [
    # Offense
    ("sig_diff",          "avg_sig_strikes_per_rd"),
    ("sig_acc_diff",      "sig_strike_accuracy_pct"),
    ("total_str_diff",    "avg_total_strikes_per_rd"),
    ("head_diff",         "avg_head_strikes_per_rd"),
    ("body_diff",         "avg_body_strikes_per_rd"),
    ("leg_diff",          "avg_leg_strikes_per_rd"),
    ("td_diff",           "avg_td_per_rd"),
    ("td_acc_diff",       "td_accuracy_pct"),
    ("ctrl_diff",         "avg_ctrl_sec_per_rd"),
    ("sub_diff",          "avg_sub_attempts_per_rd"),
    ("rev_diff",          "avg_reversals_per_rd"),
    ("kd_diff",           "avg_knockdowns_per_rd"),
    # Defense
    ("sig_absorbed_diff", "avg_sig_absorbed_per_rd"),
    ("sig_def_diff",      "sig_strike_defense_pct"),
    ("td_absorbed_diff",  "avg_td_absorbed_per_rd"),
    ("td_def_diff",       "td_defense_pct"),
    ("kd_absorbed_diff",  "avg_kd_absorbed_per_rd"),
    # Record / experience
    ("winrate_diff",      "win_pct"),
    ("finishrate_diff",   "finish_rate_pct"),
    ("experience_diff",   "total_fights"),
    # Physical
    ("reach_diff",        "reach_in"),
    ("height_diff",       "height_in"),
    ("age_diff",          "age_years"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
BASE_COLS = [f[1] for f in FEATURES]


# ── Load one row per (fight, fighter) with that fight's raw totals ──────────
# Aggregated to fight level (summed over rounds), plus what the opponent did in
# the same rounds (for defense) and the result. build_features() then turns this
# into leak-free "career up to BEFORE this fight" averages.

def load_fight_data():
    query = """
        SELECT
            f.fight_id,
            e.event_date,
            rs.fighter_id,
            f.fighter1_id,
            f.fighter2_id,
            (rs.fighter_id = f.winner_id)::int AS won,
            f.win_method,
            f.scheduled_rounds,
            a.reach_in, a.height_in, a.dob,
            count(*)                        AS rounds,
            sum(rs.sig_strikes_landed)      AS sig_l,   sum(rs.sig_strikes_attempted)   AS sig_a,
            sum(rs.total_strikes_landed)    AS tot_l,
            sum(rs.head_strikes_landed)     AS head_l,
            sum(rs.body_strikes_landed)     AS body_l,
            sum(rs.leg_strikes_landed)      AS leg_l,
            sum(rs.takedowns_landed)        AS td_l,    sum(rs.takedowns_attempted)     AS td_a,
            sum(rs.submission_attempts)     AS sub,     sum(rs.reversals)               AS rev,
            sum(rs.ctrl_time_seconds)       AS ctrl,    sum(rs.knockdowns)              AS kd,
            sum(opp.sig_strikes_landed)     AS osig_l,  sum(opp.sig_strikes_attempted)  AS osig_a,
            sum(opp.takedowns_landed)       AS otd_l,   sum(opp.takedowns_attempted)    AS otd_a,
            sum(opp.knockdowns)             AS okd
        FROM fights f
        JOIN events e       ON e.event_id = f.event_id
        JOIN round_stats rs ON rs.fight_id = f.fight_id
        JOIN round_stats opp ON opp.fight_id = rs.fight_id
                            AND opp.round_number = rs.round_number
                            AND opp.fighter_id <> rs.fighter_id
        JOIN fighters a     ON a.fighter_id = rs.fighter_id
        WHERE f.winner_id IS NOT NULL
        GROUP BY f.fight_id, e.event_date, rs.fighter_id, f.fighter1_id, f.fighter2_id,
                 f.winner_id, f.win_method, f.scheduled_rounds, a.reach_in, a.height_in, a.dob
    """
    from sqlalchemy import create_engine
    engine = create_engine("postgresql+psycopg2://postgres:ufc123@localhost:5432/ufc_analytics")
    return pd.read_sql(query, engine)


SUM_COLS = ["rounds", "sig_l", "sig_a", "tot_l", "head_l", "body_l", "leg_l",
            "td_l", "td_a", "sub", "rev", "ctrl", "kd",
            "osig_l", "osig_a", "otd_l", "otd_a", "okd"]


def _prefight_metrics(d):
    """Given the per-(fight,fighter) rows, add each fighter's career-to-BEFORE
    -this-fight averages as columns named exactly like BASE_COLS."""
    d = d.sort_values(["fighter_id", "event_date", "fight_id"]).copy()
    gb = d.groupby("fighter_id")

    # Cumulative-up-to-but-excluding-this-fight = cumsum minus the current row.
    for c in SUM_COLS:
        d[c + "_p"] = gb[c].cumsum() - d[c]
    d["fights_p"] = gb.cumcount()                      # number of prior fights
    d["wins_p"] = gb["won"].cumsum() - d["won"]
    d["finish_win"] = (d["won"].eq(1) & d["win_method"].map(
        lambda m: method_class(m) in ("KO/TKO", "Submission"))).astype(int)
    d["finwins_p"] = gb["finish_win"].cumsum() - d["finish_win"]

    r = d["rounds_p"].replace(0, np.nan)               # avoid div-by-zero on debuts

    d["avg_sig_strikes_per_rd"]    = d["sig_l_p"]  / r
    d["sig_strike_accuracy_pct"]   = d["sig_l_p"]  / d["sig_a_p"].replace(0, np.nan) * 100
    d["avg_total_strikes_per_rd"]  = d["tot_l_p"]  / r
    d["avg_head_strikes_per_rd"]   = d["head_l_p"] / r
    d["avg_body_strikes_per_rd"]   = d["body_l_p"] / r
    d["avg_leg_strikes_per_rd"]    = d["leg_l_p"]  / r
    d["avg_td_per_rd"]             = d["td_l_p"]   / r
    d["td_accuracy_pct"]           = d["td_l_p"]   / d["td_a_p"].replace(0, np.nan) * 100
    d["avg_ctrl_sec_per_rd"]       = d["ctrl_p"]   / r
    d["avg_sub_attempts_per_rd"]   = d["sub_p"]    / r
    d["avg_reversals_per_rd"]      = d["rev_p"]    / r
    d["avg_knockdowns_per_rd"]     = d["kd_p"]     / r
    d["avg_sig_absorbed_per_rd"]   = d["osig_l_p"] / r
    d["sig_strike_defense_pct"]    = (1 - d["osig_l_p"] / d["osig_a_p"].replace(0, np.nan)) * 100
    d["avg_td_absorbed_per_rd"]    = d["otd_l_p"]  / r
    d["td_defense_pct"]            = (1 - d["otd_l_p"] / d["otd_a_p"].replace(0, np.nan)) * 100
    d["avg_kd_absorbed_per_rd"]    = d["okd_p"]    / r
    d["win_pct"]                   = d["wins_p"]   / d["fights_p"].replace(0, np.nan) * 100
    d["finish_rate_pct"]           = d["finwins_p"] / d["wins_p"].replace(0, np.nan) * 100
    d["total_fights"]              = d["fights_p"]
    d["age_years"] = (pd.to_datetime(d["event_date"]) - pd.to_datetime(d["dob"])).dt.days / 365.25
    # reach_in / height_in are static and already present
    return d


# Compute leak-free differential features (Fighter1 minus Fighter2)

def build_features(df):
    d = _prefight_metrics(df)

    f1 = d[d.fighter_id == d.fighter1_id].set_index("fight_id")
    f2 = d[d.fighter_id == d.fighter2_id].set_index("fight_id")
    f1, f2 = f1.align(f2, join="inner", axis=0)

    features = pd.DataFrame(index=f1.index)
    for name, base in FEATURES:
        features[name] = f1[base] - f2[base]
    features = features.fillna(0)

    labels = f1["won"].astype(int)                    # did fighter1 win?
    meta = pd.DataFrame({
        "win_method": f1["win_method"],
        "scheduled_rounds": f1["scheduled_rounds"],
    }, index=f1.index)

    print(f"Feature matrix: {features.shape[0]} fights x {features.shape[1]} features (point-in-time)")
    print(f"Fighter 1 win rate in dataset: {labels.mean():.1%}")
    return features, labels, meta


def method_class(m):
    """Map a raw win_method string to one of KO/TKO, Submission, Decision."""
    if not m:
        return None
    if "Submission" in m:
        return "Submission"
    if "KO" in m or "TKO" in m:
        return "KO/TKO"
    if "Decision" in m:
        return "Decision"
    return None  # DQ etc. — excluded


# ── Train the win/loss model ───────────────────────────────────────────────

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nWinner model accuracy: {acc:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Fighter2 wins", "Fighter1 wins"]))

    importance_df = pd.DataFrame({
        "feature": features.columns.tolist(),
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\nTop feature importance (|coefficient|, predicting Fighter1 win):")
    print(importance_df.head(12).to_string(index=False))

    return model, scaler


# ── Train the win-method model (KO/TKO vs Submission vs Decision) ───────────

def train_method_model(features, meta):
    # Method is order-invariant, so use the magnitude of each edge, plus the
    # bout length (5-round fights go to decision far more often than 3-round).
    X = features.abs().copy()
    X["scheduled_rounds"] = meta["scheduled_rounds"].values
    y = meta["win_method"].map(method_class)

    mask = y.notnull()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)  # multinomial by default for multiclass
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"\nMethod model accuracy: {acc:.1%}  (baseline = predict most common class)")
    print(f"Method distribution: {y.value_counts().to_dict()}")

    return model, scaler


# ── Predict a matchup ──────────────────────────────────────────────────────

STAT_LABELS = [
    ("Sig Strikes/Rd",   "avg_sig_strikes_per_rd"),
    ("Strike Acc %",     "sig_strike_accuracy_pct"),
    ("Total Strikes/Rd", "avg_total_strikes_per_rd"),
    ("Head/Body/Leg",    None),  # special composite line
    ("Takedowns/Rd",     "avg_td_per_rd"),
    ("TD Accuracy %",    "td_accuracy_pct"),
    ("Ctrl Sec/Rd",      "avg_ctrl_sec_per_rd"),
    ("Sub Attempts/Rd",  "avg_sub_attempts_per_rd"),
    ("Knockdowns/Rd",    "avg_knockdowns_per_rd"),
    ("Sig Absorbed/Rd",  "avg_sig_absorbed_per_rd"),
    ("Strike Defense %", "sig_strike_defense_pct"),
    ("TD Defense %",     "td_defense_pct"),
    ("Win %",            "win_pct"),
    ("Finish Rate %",    "finish_rate_pct"),
    ("Reach (in)",       "reach_in"),
    ("Age",              "age_years"),
]

# Metrics where a LOWER value is the advantage (so the edge goes to the smaller
# number): strikes/takedowns/knockdowns absorbed, and age.
LOWER_IS_BETTER = {
    "avg_sig_absorbed_per_rd", "avg_td_absorbed_per_rd",
    "avg_kd_absorbed_per_rd", "age_years",
}


def get_matchup_prediction(model, scaler, method_model, method_scaler, fighter1, fighter2, rounds=3):
    """Return the same prediction data for the terminal and the local website."""
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

    conn = get_connection()
    cur = conn.cursor()

    def get_stats(name):
        cur.execute("""
            SELECT
                m.avg_sig_strikes_per_rd, m.sig_strike_accuracy_pct, m.avg_total_strikes_per_rd,
                m.avg_head_strikes_per_rd, m.avg_body_strikes_per_rd, m.avg_leg_strikes_per_rd,
                m.avg_td_per_rd, m.td_accuracy_pct, m.avg_ctrl_sec_per_rd,
                m.avg_sub_attempts_per_rd, m.avg_reversals_per_rd, m.avg_knockdowns_per_rd,
                m.avg_sig_absorbed_per_rd, m.sig_strike_defense_pct, m.avg_td_absorbed_per_rd,
                m.td_defense_pct, m.avg_kd_absorbed_per_rd,
                COALESCE(r.win_pct, 0), COALESCE(r.finish_rate_pct, 0), m.total_fights,
                m.reach_in, m.height_in,
                CASE WHEN m.dob IS NOT NULL THEN date_part('year', age(m.dob)) END,
                m.name
            FROM v_fighter_metrics m
            LEFT JOIN v_fighter_record r ON r.fighter_id = m.fighter_id
            WHERE m.name ILIKE %s
            ORDER BY (lower(m.name) = lower(%s)) DESC, m.total_fights DESC, m.name
            LIMIT 1
        """, ("%" + name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%", name))
        return cur.fetchone()

    try:
        s1 = get_stats(fighter1)
        s2 = get_stats(fighter2)
    finally:
        cur.close()
        conn.close()

    if not s1:
        raise ValueError(f"Fighter not found: {fighter1}")
    if not s2:
        raise ValueError(f"Fighter not found: {fighter2}")

    # Resolved display names (in case ILIKE matched a fuller name)
    name1, name2 = s1[-1], s2[-1]
    if name1 == name2:
        raise ValueError("Both names matched the same fighter. Choose two different fighters.")

    # Build the differential feature vector (first len(FEATURES) columns).
    def num(v):
        return float(v) if v is not None else 0.0

    diffs = [num(s1[i]) - num(s2[i]) for i in range(len(FEATURES))]

    # Winner model (directional features)
    feature_row = pd.DataFrame([diffs], columns=FEATURE_NAMES)
    prob = dict(zip(model.classes_, model.predict_proba(scaler.transform(feature_row))[0]))
    prob_a = float(prob[1])
    prob_b = float(prob[0])

    # Method model (order-invariant edges + bout length)
    method_row = [abs(d) for d in diffs] + [rounds]
    method_frame = pd.DataFrame([method_row], columns=FEATURE_NAMES + ["scheduled_rounds"])
    m_prob = method_model.predict_proba(method_scaler.transform(method_frame))[0]
    method_probs = {str(label): float(value) for label, value in zip(method_model.classes_, m_prob)}
    pred_method = max(method_probs, key=method_probs.get)

    # Confidence score out of 10 (distance of the win prob from a coin flip)
    conf_score = min(10, round(abs(prob_a - 0.5) * 20))
    if conf_score < 5:
        conf_label = "LOW"
    elif conf_score <= 8:
        conf_label = "MEDIUM"
    else:
        conf_label = "HIGH"

    stats = []
    for label, base in STAT_LABELS:
        if base is None:
            zones = [BASE_COLS.index(f"avg_{zone}_strikes_per_rd") for zone in ("head", "body", "leg")]
            stats.append({"label": label, "fighter1": "/".join(f"{num(s1[i]):.0f}" for i in zones),
                          "fighter2": "/".join(f"{num(s2[i]):.0f}" for i in zones),
                          "edge": None, "difference": None, "lower_is_better": False})
            continue
        i = BASE_COLS.index(base)
        v1 = None if s1[i] is None else float(s1[i])
        v2 = None if s2[i] is None else float(s2[i])
        edge, difference = None, None
        if v1 is not None and v2 is not None:
            difference = abs(v1 - v2)
            edge = "even" if v1 == v2 else (
                "fighter1" if (v1 > v2) ^ (base in LOWER_IS_BETTER) else "fighter2")
        stats.append({"label": label, "fighter1": v1, "fighter2": v2,
                      "edge": edge, "difference": difference, "lower_is_better": base in LOWER_IS_BETTER})

    return {"fighter1": name1, "fighter2": name2, "rounds": rounds,
            "probabilities": {"fighter1": prob_a, "fighter2": prob_b},
            "method_probabilities": method_probs, "predicted_method": pred_method,
            "confidence": {"score": conf_score, "label": conf_label}, "stats": stats}


def predict_matchup(model, scaler, method_model, method_scaler, fighter1, fighter2, rounds=3):
    """Print a shared prediction in the original terminal format."""
    try:
        result = get_matchup_prediction(model, scaler, method_model, method_scaler, fighter1, fighter2, rounds)
    except ValueError as exc:
        print(f"  {exc}")
        return
    name1, name2 = result["fighter1"], result["fighter2"]
    prob_a, prob_b = result["probabilities"]["fighter1"], result["probabilities"]["fighter2"]
    method_probs, pred_method = result["method_probabilities"], result["predicted_method"]
    conf_score, conf_label = result["confidence"]["score"], result["confidence"]["label"]

    print(f"\n{'='*54}")
    print(f"  {name1} vs {name2}  ({rounds}-round bout)")
    print(f"{'='*54}")
    print(f"  {name1:<30} {prob_a*100:5.1f}%")
    print(f"  {name2:<30} {prob_b*100:5.1f}%")
    print(f"{'='*54}")
    print(f"  Confidence: {conf_score}/10 ({conf_label})")
    print(f"  Predicted method: {pred_method}")
    print(f"    KO/TKO {method_probs.get('KO/TKO',0)*100:4.0f}%   "
          f"Submission {method_probs.get('Submission',0)*100:4.0f}%   "
          f"Decision {method_probs.get('Decision',0)*100:4.0f}%")

    # Stat breakdown
    print(f"\n  Stat breakdown ({name1} vs {name2}):")
    print(f"  {'Metric':<18} {'F1':>10} {'F2':>10} {'Edge':>14}")
    print(f"  {'-'*54}")
    for stat in result["stats"]:
        def display(value):
            return "N/A" if value is None else value if isinstance(value, str) else f"{value:.2f}"
        edge = "N/A"
        if stat["edge"] == "even":
            edge = "Even"
        elif stat["edge"]:
            winner = result[stat["edge"]]
            edge = f"{winner.split()[0]} +{stat['difference']:.2f}"
        elif isinstance(stat["fighter1"], str):
            edge = ""
        print(f"  {stat['label']:<18} {display(stat['fighter1']):>10} {display(stat['fighter2']):>10} {edge:>14}")



# Main

if __name__ == "__main__":
    df = load_fight_data()
    features, labels, meta = build_features(df)
    model, scaler = train_model(features, labels)
    method_model, method_scaler = train_method_model(features, meta)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model,         MODEL_DIR / "ufc_model.pkl")
    joblib.dump(scaler,        MODEL_DIR / "ufc_scaler.pkl")
    joblib.dump(method_model,  MODEL_DIR / "ufc_method_model.pkl")
    joblib.dump(method_scaler, MODEL_DIR / "ufc_method_scaler.pkl")
    print("\nModels saved.")

    predict_matchup(model, scaler, method_model, method_scaler,
                    "Islam Makhachev", "Dustin Poirier", rounds=5)

    # Interactive predictor
    print("\n--- Interactive Fight Predictor ---")
    print("Type 'quit' to exit\n")
    while True:
        f1 = input("Enter Fighter 1 name: ").strip()
        if f1.lower() == "quit":
            break
        f2 = input("Enter Fighter 2 name: ").strip()
        if f2.lower() == "quit":
            break
        r = input("Scheduled rounds (3 or 5) [3]: ").strip()
        rounds = 5 if r == "5" else 3
        predict_matchup(model, scaler, method_model, method_scaler, f1, f2, rounds=rounds)
