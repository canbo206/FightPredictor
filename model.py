import psycopg2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Database connection

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ufc_analytics",
        user="postgres",
        password="ufc123",
        port="5432"
    )

# Load fight data from PostgreSQL into a datafram

def load_fight_data():
    conn = get_connection()

    query = """
        SELECT
            f.fight_id,
            f.fighter1_id,
            f.fighter2_id,
            f.winner_id,
            f.weight_class,
            -- Fighter 1 career averages BEFORE this fight
            f1avg.avg_sig_strikes_per_rd    AS f1_sig,
            f1avg.sig_strike_accuracy_pct   AS f1_acc,
            f1avg.avg_td_per_rd             AS f1_td,
            f1avg.td_accuracy_pct           AS f1_td_acc,
            f1avg.avg_ctrl_sec_per_rd       AS f1_ctrl,
            f1avg.avg_knockdowns_per_rd     AS f1_kd,
            f1avg.avg_sub_attempts_per_rd   AS f1_sub,
            -- Fighter 2 career averages BEFORE this fight
            f2avg.avg_sig_strikes_per_rd    AS f2_sig,
            f2avg.sig_strike_accuracy_pct   AS f2_acc,
            f2avg.avg_td_per_rd             AS f2_td,
            f2avg.td_accuracy_pct           AS f2_td_acc,
            f2avg.avg_ctrl_sec_per_rd       AS f2_ctrl,
            f2avg.avg_knockdowns_per_rd     AS f2_kd,
            f2avg.avg_sub_attempts_per_rd   AS f2_sub
        FROM fights f
        JOIN v_fighter_averages f1avg ON f1avg.fighter_id = f.fighter1_id
        JOIN v_fighter_averages f2avg ON f2avg.fighter_id = f.fighter2_id
        WHERE f.winner_id IS NOT NULL
    """

    from sqlalchemy import create_engine
    engine = create_engine("postgresql+psycopg2://postgres:ufc123@localhost:5432/ufc_analytics")
    df = pd.read_sql(query, engine)
    return df


# compute stat differentials


def build_features(df):
    # Differential = Fighter1 stat minus Fighter2 stat
    # Positive value means Fighter1 has the edge
    features = pd.DataFrame()

    features["sig_diff"]     = df["f1_sig"]    - df["f2_sig"]
    features["acc_diff"]     = df["f1_acc"]    - df["f2_acc"]
    features["td_diff"]      = df["f1_td"]     - df["f2_td"]
    features["td_acc_diff"]  = df["f1_td_acc"] - df["f2_td_acc"]
    features["ctrl_diff"]    = df["f1_ctrl"]   - df["f2_ctrl"]
    features["kd_diff"]      = df["f1_kd"]     - df["f2_kd"]
    features["sub_diff"]     = df["f1_sub"]    - df["f2_sub"]

    # Label: 1 if fighter1 won, 0 if fighter2 won
    labels = (df["winner_id"] == df["fighter1_id"]).astype(int)

    # Drop rows where any feature is null
    mask = features.notnull().all(axis=1)
    features = features[mask].fillna(0)
    labels    = labels[mask]

    print(f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} features")
    print(f"Fighter 1 win rate in dataset: {labels.mean():.1%}")
    return features, labels


#  Train the model

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Scale features so no single stat dominates
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {acc:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Fighter2 wins","Fighter1 wins"]))

    # Show feature importance
    feature_names = features.columns.tolist()
    coefficients  = model.coef_[0]
    importance_df = pd.DataFrame({
        "feature":     feature_names,
        "coefficient": coefficients
    }).sort_values("coefficient", ascending=False)
    print("\nFeature importance (higher = stronger predictor of Fighter1 winning):")
    print(importance_df.to_string(index=False))

    return model, scaler


# Predict a matchup


def predict_matchup(model, scaler, fighter1, fighter2):
    conn = get_connection()
    cur  = conn.cursor()

    def get_stats(name):
        cur.execute("""
            SELECT avg_sig_strikes_per_rd, sig_strike_accuracy_pct,
                   avg_td_per_rd, td_accuracy_pct, avg_ctrl_sec_per_rd,
                   avg_knockdowns_per_rd, avg_sub_attempts_per_rd
            FROM v_fighter_averages
            WHERE name ILIKE %s
        """, (f"%{name}%",))
        return cur.fetchone()

    s1 = get_stats(fighter1)
    s2 = get_stats(fighter2)

    if not s1:
        print(f"  Fighter not found: {fighter1}")
        return
    if not s2:
        print(f"  Fighter not found: {fighter2}")
        return

    cur.close()
    conn.close()

    diffs = [[
        s1[0] - s2[0],
        s1[1] - s2[1],
        s1[2] - s2[2],
        s1[3] - s2[3],
        s1[4] - s2[4],
        s1[5] - s2[5],
        s1[6] - s2[6],
    ]]

    diffs_scaled = scaler.transform(diffs)
    prob         = model.predict_proba(diffs_scaled)[0]

    prob_a = prob[1] * 100
    prob_b = prob[0] * 100

    # Confidence level
    confidence = abs(prob_a - 50)
    if confidence > 30:
        conf_label = "HIGH confidence"
    elif confidence > 15:
        conf_label = "MEDIUM confidence"
    else:
        conf_label = "LOW confidence"

    print(f"\n{'='*50}")
    print(f"  {fighter1} vs {fighter2}")
    print(f"{'='*50}")
    print(f"  {fighter1:<28} {prob_a:.1f}%")
    print(f"  {fighter2:<28} {prob_b:.1f}%")
    print(f"{'='*50}")
    print(f"  Confidence: {conf_label}")
    print(f"\n  Stat breakdown ({fighter1} vs {fighter2}):")
    print(f"  {'Metric':<22} {'F1':>8} {'F2':>8} {'Edge':>10}")
    print(f"  {'-'*50}")

    labels = [
        ("Sig Strikes/Rd",  s1[0], s2[0]),
        ("Strike Acc %",    s1[1], s2[1]),
        ("Takedowns/Rd",    s1[2], s2[2]),
        ("TD Accuracy %",   s1[3], s2[3]),
        ("Ctrl Sec/Rd",     s1[4], s2[4]),
        ("Knockdowns/Rd",   s1[5], s2[5]),
        ("Sub Attempts/Rd", s1[6], s2[6]),
    ]

    for label, v1, v2 in labels:
        if v1 is None or v2 is None:
            edge = "N/A"
        elif v1 > v2:
            edge = f"{fighter1.split()[0]} +{v1-v2:.2f}"
        elif v2 > v1:
            edge = f"{fighter2.split()[0]} +{v2-v1:.2f}"
        else:
            edge = "Even"
        v1_str = f"{v1:.2f}" if v1 is not None else "N/A"
        v2_str = f"{v2:.2f}" if v2 is not None else "N/A"
        print(f"  {label:<22} {v1_str:>8} {v2_str:>8} {edge:>14}")

# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    df               = load_fight_data()
    features, labels = build_features(df)
    model, scaler    = train_model(features, labels)

    # Save model to disk
    joblib.dump(model,  "/Users/canbo/FightAnalyze/ufc_model.pkl")
    joblib.dump(scaler, "/Users/canbo/FightAnalyze/ufc_scaler.pkl")
    print("\nModel saved to ufc_model.pkl")

    # Test a prediction
    predict_matchup(model, scaler, "Islam Makhachev", "Dustin Poirier")

      # Interactive predictor
    print("\n--- Interactive Fight Predictor ---")
    print("Type 'quit' to exit\n")
    while True:
        f1 = input("Enter Fighter 1 name: ").strip()
        if f1.lower() == 'quit':
            break
        f2 = input("Enter Fighter 2 name: ").strip()
        if f2.lower() == 'quit':
            break
        predict_matchup(model, scaler, f1, f2)