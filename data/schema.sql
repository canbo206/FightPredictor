-- =============================================================
-- UFC Fight Analytics Database
-- PostgreSQL
-- Fresh databases only: run this file, then queries/metrics_view.sql.
-- Existing databases need migrations; this file does not upgrade them.
-- =============================================================

-- Tables


CREATE TABLE fighters (
    fighter_id    SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    nickname      TEXT,
    weight_class  TEXT NOT NULL,
    stance        TEXT CHECK(stance IN ('Orthodox','Southpaw','Switch','Open Stance','Sideways')),
    dob           DATE,
    reach_in      NUMERIC(5,2),
    height_in     NUMERIC(5,2),
    wins          INTEGER DEFAULT 0,
    losses        INTEGER DEFAULT 0,
    draws         INTEGER DEFAULT 0,
    win_by_ko     INTEGER DEFAULT 0,
    win_by_sub    INTEGER DEFAULT 0,
    win_by_dec    INTEGER DEFAULT 0
);

CREATE TABLE events (
    event_id    SERIAL PRIMARY KEY,
    event_name  TEXT NOT NULL UNIQUE,
    event_date  DATE NOT NULL,
    location    TEXT,
    card_type   TEXT CHECK(card_type IN ('PPV','Fight Night','UFC 300'))
);

CREATE TABLE fights (
    fight_id         SERIAL PRIMARY KEY,
    event_id         INTEGER NOT NULL REFERENCES events(event_id),
    fighter1_id      INTEGER NOT NULL REFERENCES fighters(fighter_id),
    fighter2_id      INTEGER NOT NULL REFERENCES fighters(fighter_id),
    winner_id        INTEGER REFERENCES fighters(fighter_id),
    weight_class     TEXT NOT NULL,
    scheduled_rounds INTEGER DEFAULT 3,
    actual_rounds    INTEGER,
    win_method       TEXT CHECK(win_method IN ('KO/TKO','Submission','Decision - Unanimous','Decision - Split','Decision - Majority','No Contest','Draw')),
    win_round        INTEGER,
    win_time         TEXT,
    is_title_fight   BOOLEAN DEFAULT FALSE,
    CONSTRAINT fights_event_fighters_key UNIQUE (event_id, fighter1_id, fighter2_id)
);

CREATE TABLE round_stats (
    stat_id                 SERIAL PRIMARY KEY,
    fight_id                INTEGER NOT NULL REFERENCES fights(fight_id),
    fighter_id              INTEGER NOT NULL REFERENCES fighters(fighter_id),
    round_number            INTEGER NOT NULL,
    -- Striking
    sig_strikes_landed      INTEGER DEFAULT 0,
    sig_strikes_attempted   INTEGER DEFAULT 0,
    total_strikes_landed    INTEGER DEFAULT 0,
    total_strikes_attempted INTEGER DEFAULT 0,
    -- Strike zones
    head_strikes_landed     INTEGER DEFAULT 0,
    head_strikes_attempted  INTEGER DEFAULT 0,
    body_strikes_landed     INTEGER DEFAULT 0,
    body_strikes_attempted  INTEGER DEFAULT 0,
    leg_strikes_landed      INTEGER DEFAULT 0,
    leg_strikes_attempted   INTEGER DEFAULT 0,
    -- Grappling
    takedowns_landed        INTEGER DEFAULT 0,
    takedowns_attempted     INTEGER DEFAULT 0,
    submission_attempts     INTEGER DEFAULT 0,
    reversals               INTEGER DEFAULT 0,
    -- Control
    ctrl_time_seconds       INTEGER DEFAULT 0,
    knockdowns              INTEGER DEFAULT 0,
    UNIQUE(fight_id, fighter_id, round_number)
);

-- =============================================================
-- Views
-- =============================================================

-- Fighter career averages per round
CREATE VIEW v_fighter_averages AS
SELECT
    f.fighter_id,
    f.name,
    f.weight_class,
    COUNT(DISTINCT rs.fight_id)                                                     AS total_fights,
    ROUND(AVG(rs.sig_strikes_landed)::NUMERIC, 1)                                   AS avg_sig_strikes_per_rd,
    ROUND((AVG(rs.sig_strikes_landed::NUMERIC
               / NULLIF(rs.sig_strikes_attempted, 0)) * 100)::NUMERIC, 1)          AS sig_strike_accuracy_pct,
    ROUND(AVG(rs.takedowns_landed)::NUMERIC, 2)                                     AS avg_td_per_rd,
    ROUND((AVG(rs.takedowns_landed::NUMERIC
               / NULLIF(rs.takedowns_attempted, 0)) * 100)::NUMERIC, 1)            AS td_accuracy_pct,
    ROUND(AVG(rs.ctrl_time_seconds)::NUMERIC, 0)                                    AS avg_ctrl_sec_per_rd,
    ROUND(AVG(rs.submission_attempts)::NUMERIC, 2)                                  AS avg_sub_attempts_per_rd,
    ROUND(AVG(rs.knockdowns)::NUMERIC, 2)                                           AS avg_knockdowns_per_rd
FROM fighters f
JOIN round_stats rs USING (fighter_id)
GROUP BY f.fighter_id, f.name, f.weight_class;

-- Head-to-head fight comparison
CREATE VIEW v_fight_comparison AS
SELECT
    fi.fight_id,
    e.event_name,
    e.event_date,
    f1.name   AS fighter1,
    f2.name   AS fighter2,
    fw.name   AS winner,
    fi.win_method,
    fi.win_round,
    SUM(CASE WHEN rs.fighter_id = fi.fighter1_id THEN rs.sig_strikes_landed  END) AS f1_total_sig,
    SUM(CASE WHEN rs.fighter_id = fi.fighter1_id THEN rs.takedowns_landed    END) AS f1_total_td,
    SUM(CASE WHEN rs.fighter_id = fi.fighter1_id THEN rs.ctrl_time_seconds   END) AS f1_total_ctrl,
    SUM(CASE WHEN rs.fighter_id = fi.fighter2_id THEN rs.sig_strikes_landed  END) AS f2_total_sig,
    SUM(CASE WHEN rs.fighter_id = fi.fighter2_id THEN rs.takedowns_landed    END) AS f2_total_td,
    SUM(CASE WHEN rs.fighter_id = fi.fighter2_id THEN rs.ctrl_time_seconds   END) AS f2_total_ctrl
FROM fights fi
JOIN events        e  ON fi.event_id    = e.event_id
JOIN fighters      f1 ON fi.fighter1_id = f1.fighter_id
JOIN fighters      f2 ON fi.fighter2_id = f2.fighter_id
LEFT JOIN fighters fw ON fi.winner_id   = fw.fighter_id
LEFT JOIN round_stats rs ON rs.fight_id = fi.fight_id
GROUP BY fi.fight_id, e.event_name, e.event_date, f1.name, f2.name, fw.name, fi.win_method, fi.win_round;

-- Round-by-round momentum
CREATE VIEW v_round_momentum AS
SELECT
    rs1.fight_id,
    rs1.round_number,
    f1.name                AS fighter1,
    f2.name                AS fighter2,
    rs1.sig_strikes_landed AS f1_sig,
    rs2.sig_strikes_landed AS f2_sig,
    rs1.ctrl_time_seconds  AS f1_ctrl,
    rs2.ctrl_time_seconds  AS f2_ctrl,
    rs1.takedowns_landed   AS f1_td,
    rs2.takedowns_landed   AS f2_td,
    (rs1.sig_strikes_landed * 2 + rs1.takedowns_landed * 5
        + rs1.ctrl_time_seconds / 30 + rs1.knockdowns * 10) AS f1_score,
    (rs2.sig_strikes_landed * 2 + rs2.takedowns_landed * 5
        + rs2.ctrl_time_seconds / 30 + rs2.knockdowns * 10) AS f2_score,
    CASE
        WHEN (rs1.sig_strikes_landed * 2 + rs1.takedowns_landed * 5
                + rs1.ctrl_time_seconds / 30 + rs1.knockdowns * 10)
           > (rs2.sig_strikes_landed * 2 + rs2.takedowns_landed * 5
                + rs2.ctrl_time_seconds / 30 + rs2.knockdowns * 10)
        THEN f1.name
        ELSE f2.name
    END AS round_winner
FROM round_stats rs1
JOIN round_stats rs2
    ON  rs1.fight_id     = rs2.fight_id
    AND rs1.round_number = rs2.round_number
    AND rs1.fighter_id   < rs2.fighter_id
JOIN fights   fi ON fi.fight_id    = rs1.fight_id
JOIN fighters f1 ON f1.fighter_id  = rs1.fighter_id
JOIN fighters f2 ON f2.fighter_id  = rs2.fighter_id;

-- Win probability model
CREATE VIEW v_win_probability AS
SELECT
    a.name        AS fighter_a,
    b.name        AS fighter_b,
    a.weight_class,
    ROUND((a.avg_sig_strikes_per_rd / NULLIF(a.avg_sig_strikes_per_rd + b.avg_sig_strikes_per_rd, 0) * 100)::NUMERIC, 1)
                  AS a_striking_edge_pct,
    ROUND((a.avg_td_per_rd / NULLIF(a.avg_td_per_rd + b.avg_td_per_rd, 0) * 100)::NUMERIC, 1)
                  AS a_grappling_edge_pct,
    ROUND((a.avg_ctrl_sec_per_rd / NULLIF(a.avg_ctrl_sec_per_rd + b.avg_ctrl_sec_per_rd, 0) * 100)::NUMERIC, 1)
                  AS a_control_edge_pct,
    ROUND((
        (a.avg_sig_strikes_per_rd  / NULLIF(a.avg_sig_strikes_per_rd  + b.avg_sig_strikes_per_rd,  0)) * 40 +
        (a.avg_td_per_rd           / NULLIF(a.avg_td_per_rd           + b.avg_td_per_rd,           0)) * 30 +
        (a.avg_ctrl_sec_per_rd     / NULLIF(a.avg_ctrl_sec_per_rd     + b.avg_ctrl_sec_per_rd,     0)) * 20 +
        (a.sig_strike_accuracy_pct / NULLIF(a.sig_strike_accuracy_pct + b.sig_strike_accuracy_pct, 0)) * 10
    )::NUMERIC, 1) AS a_win_probability_pct
FROM v_fighter_averages a
JOIN v_fighter_averages b
    ON  a.fighter_id   < b.fighter_id
    AND a.weight_class = b.weight_class;
