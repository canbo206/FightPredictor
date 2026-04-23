
-- UFC Fight Analytics — Core Query Playbook
-- PostgreSQL
-- Run these individually in VS Code against ufc_analytics



-- ---------------------------------------------------------------
-- 1. Fighter stat summary (career per-round averages)
-- ---------------------------------------------------------------
SELECT
    name,
    weight_class,
    total_fights,
    avg_sig_strikes_per_rd  AS "Sig Strikes/Rd",
    sig_strike_accuracy_pct AS "Strike Acc %",
    avg_td_per_rd           AS "TDs/Rd",
    td_accuracy_pct         AS "TD Acc %",
    avg_ctrl_sec_per_rd     AS "Ctrl Sec/Rd",
    avg_sub_attempts_per_rd AS "Sub Att/Rd",
    avg_knockdowns_per_rd   AS "KDs/Rd"
FROM v_fighter_averages
ORDER BY avg_sig_strikes_per_rd DESC;


-- ---------------------------------------------------------------
-- 2. Round-by-round momentum for a specific fight
-- ---------------------------------------------------------------
SELECT
    round_number  AS "Round",
    fighter1      AS "Fighter A",
    fighter2      AS "Fighter B",
    f1_sig        AS "A Sig",
    f2_sig        AS "B Sig",
    f1_td         AS "A TDs",
    f2_td         AS "B TDs",
    f1_ctrl       AS "A Ctrl (s)",
    f2_ctrl       AS "B Ctrl (s)",
    f1_score      AS "A Score",
    f2_score      AS "B Score",
    round_winner  AS "Round Winner"
FROM v_round_momentum
WHERE fight_id = 1    -- change this to inspect a different fight
ORDER BY round_number;


-- ---------------------------------------------------------------
-- 3. Head-to-head fight breakdown
-- ---------------------------------------------------------------
SELECT
    event_name,
    event_date,
    fighter1,
    fighter2,
    winner,
    win_method,
    win_round,
    f1_total_sig  AS "F1 Total Sig",
    f2_total_sig  AS "F2 Total Sig",
    f1_total_td   AS "F1 TDs",
    f2_total_td   AS "F2 TDs",
    f1_total_ctrl AS "F1 Ctrl (s)",
    f2_total_ctrl AS "F2 Ctrl (s)"
FROM v_fight_comparison
ORDER BY event_date DESC;


-- ---------------------------------------------------------------
-- 4. Win probability matchup matrix (same weight class)
-- ---------------------------------------------------------------
SELECT
    fighter_a,
    fighter_b,
    weight_class,
    a_striking_edge_pct   AS "Striking Edge %",
    a_grappling_edge_pct  AS "Grappling Edge %",
    a_control_edge_pct    AS "Control Edge %",
    a_win_probability_pct AS "Win Prob A %",
    ROUND((100 - a_win_probability_pct)::NUMERIC, 1) AS "Win Prob B %"
FROM v_win_probability
ORDER BY weight_class, fighter_a;


-- ---------------------------------------------------------------
-- 5. Striker vs Grappler tendency profile
-- ---------------------------------------------------------------
SELECT
    name,
    weight_class,
    avg_sig_strikes_per_rd,
    avg_td_per_rd,
    avg_ctrl_sec_per_rd,
    CASE
        WHEN avg_sig_strikes_per_rd > 18 AND avg_td_per_rd < 1   THEN 'Striker'
        WHEN avg_td_per_rd > 1.5 AND avg_ctrl_sec_per_rd > 60    THEN 'Grappler'
        WHEN avg_sig_strikes_per_rd > 15 AND avg_td_per_rd > 1   THEN 'Complete'
        ELSE 'Balanced'
    END AS fighter_style
FROM v_fighter_averages
ORDER BY fighter_style, avg_sig_strikes_per_rd DESC;


-- ---------------------------------------------------------------
-- 6. Fight finish predictor
--    Higher finish_score = more likely to end before the final bell
-- ---------------------------------------------------------------
SELECT
    a.name AS fighter_a,
    b.name AS fighter_b,
    ROUND((
        (a.avg_knockdowns_per_rd    + b.avg_knockdowns_per_rd)    * 25 +
        (a.avg_sub_attempts_per_rd  + b.avg_sub_attempts_per_rd)  * 15 +
        (a.avg_sig_strikes_per_rd   + b.avg_sig_strikes_per_rd)   * 0.5
    )::NUMERIC, 1) AS finish_score,
    CASE
        WHEN (a.avg_knockdowns_per_rd + b.avg_knockdowns_per_rd) >
             (a.avg_sub_attempts_per_rd + b.avg_sub_attempts_per_rd)
        THEN 'KO/TKO likely'
        WHEN (a.avg_sub_attempts_per_rd + b.avg_sub_attempts_per_rd) > 0.3
        THEN 'Submission likely'
        ELSE 'Decision likely'
    END AS predicted_method
FROM v_fighter_averages a
JOIN v_fighter_averages b
    ON  a.fighter_id   < b.fighter_id
    AND a.weight_class = b.weight_class
ORDER BY finish_score DESC;


-- ---------------------------------------------------------------
-- 7. Weight-class striking volume benchmark
-- ---------------------------------------------------------------
SELECT
    f.weight_class,
    ROUND(AVG(rs.sig_strikes_landed)::NUMERIC, 1)  AS avg_sig_per_rd,
    ROUND(AVG(rs.takedowns_landed)::NUMERIC, 2)    AS avg_td_per_rd,
    ROUND(AVG(rs.ctrl_time_seconds)::NUMERIC, 0)   AS avg_ctrl_sec,
    COUNT(DISTINCT rs.fight_id)                    AS total_fights_sampled
FROM round_stats rs
JOIN fighters f USING (fighter_id)
GROUP BY f.weight_class
ORDER BY avg_sig_per_rd DESC;


-- ---------------------------------------------------------------
-- 8. Cumulative in-fight momentum shift
--    Running sig strike differential across rounds
-- ---------------------------------------------------------------
WITH diffs AS (
    SELECT
        rs1.fight_id,
        rs1.round_number,
        f1.name AS fighter_a,
        f2.name AS fighter_b,
        rs1.sig_strikes_landed - rs2.sig_strikes_landed AS sig_diff
    FROM round_stats rs1
    JOIN round_stats rs2
        ON  rs1.fight_id     = rs2.fight_id
        AND rs1.round_number = rs2.round_number
        AND rs1.fighter_id   < rs2.fighter_id
    JOIN fights   fi ON fi.fight_id   = rs1.fight_id
    JOIN fighters f1 ON f1.fighter_id = rs1.fighter_id
    JOIN fighters f2 ON f2.fighter_id = rs2.fighter_id
)
SELECT
    fight_id,
    round_number,
    fighter_a,
    fighter_b,
    sig_diff,
    SUM(sig_diff) OVER (PARTITION BY fight_id ORDER BY round_number) AS cumulative_diff
FROM diffs
ORDER BY fight_id, round_number;
