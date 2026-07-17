-- Enhanced per-fighter feature views used by the prediction model.
--
--   v_fighter_metrics — offense, DEFENSE (what opponents did to them), and
--                       physicals (height/reach/stance/dob), per round.
--   v_fighter_record  — wins / losses / win% / finish-rate from fight results.
--
-- Defense metrics come from pairing each round_stats row with the opponent's
-- row in the same fight + round (self-join), then averaging per round.
--
-- Run with:
--   PGPASSWORD=ufc123 /Library/PostgreSQL/18/bin/psql \
--     -h localhost -p 5432 -U postgres -d ufc_analytics \
--     -f queries/metrics_view.sql

-- ── Per-fighter offense + defense + physicals ──────────────────────────────
CREATE OR REPLACE VIEW v_fighter_metrics AS
WITH paired AS (
    SELECT
        me.fighter_id,
        me.fight_id,
        me.round_number,
        -- my offense
        me.sig_strikes_landed,      me.sig_strikes_attempted,
        me.total_strikes_landed,
        me.head_strikes_landed,     me.body_strikes_landed,   me.leg_strikes_landed,
        me.takedowns_landed,        me.takedowns_attempted,
        me.submission_attempts,     me.reversals,
        me.ctrl_time_seconds,       me.knockdowns,
        -- opponent output in the same round = what I absorbed / defended
        opp.sig_strikes_landed      AS opp_sig_landed,
        opp.sig_strikes_attempted   AS opp_sig_attempted,
        opp.takedowns_landed        AS opp_td_landed,
        opp.takedowns_attempted     AS opp_td_attempted,
        opp.knockdowns              AS opp_kd
    FROM round_stats me
    JOIN round_stats opp
      ON opp.fight_id     = me.fight_id
     AND opp.round_number = me.round_number
     AND opp.fighter_id  <> me.fighter_id
)
SELECT
    f.fighter_id,
    f.name,
    f.weight_class,
    f.height_in,
    f.reach_in,
    f.stance,
    f.dob,
    count(DISTINCT p.fight_id)                                                              AS total_fights,
    count(*)                                                                                AS total_rounds,
    -- OFFENSE (per round)
    round(avg(p.sig_strikes_landed), 2)                                                     AS avg_sig_strikes_per_rd,
    round(sum(p.sig_strikes_landed)::numeric   / NULLIF(sum(p.sig_strikes_attempted), 0) * 100, 1) AS sig_strike_accuracy_pct,
    round(avg(p.total_strikes_landed), 2)                                                   AS avg_total_strikes_per_rd,
    round(avg(p.head_strikes_landed), 2)                                                    AS avg_head_strikes_per_rd,
    round(avg(p.body_strikes_landed), 2)                                                    AS avg_body_strikes_per_rd,
    round(avg(p.leg_strikes_landed), 2)                                                     AS avg_leg_strikes_per_rd,
    round(avg(p.takedowns_landed), 2)                                                       AS avg_td_per_rd,
    round(sum(p.takedowns_landed)::numeric     / NULLIF(sum(p.takedowns_attempted), 0) * 100, 1) AS td_accuracy_pct,
    round(avg(p.ctrl_time_seconds), 0)                                                      AS avg_ctrl_sec_per_rd,
    round(avg(p.submission_attempts), 2)                                                    AS avg_sub_attempts_per_rd,
    round(avg(p.reversals), 2)                                                              AS avg_reversals_per_rd,
    round(avg(p.knockdowns), 2)                                                             AS avg_knockdowns_per_rd,
    -- DEFENSE (per round)
    round(avg(p.opp_sig_landed), 2)                                                         AS avg_sig_absorbed_per_rd,
    round((1 - sum(p.opp_sig_landed)::numeric  / NULLIF(sum(p.opp_sig_attempted), 0)) * 100, 1) AS sig_strike_defense_pct,
    round(avg(p.opp_td_landed), 2)                                                          AS avg_td_absorbed_per_rd,
    round((1 - sum(p.opp_td_landed)::numeric   / NULLIF(sum(p.opp_td_attempted), 0)) * 100, 1)  AS td_defense_pct,
    round(avg(p.opp_kd), 2)                                                                 AS avg_kd_absorbed_per_rd
FROM fighters f
JOIN paired p ON p.fighter_id = f.fighter_id
GROUP BY f.fighter_id, f.name, f.weight_class, f.height_in, f.reach_in, f.stance, f.dob;


-- ── Per-fighter record & finish rate ───────────────────────────────────────
CREATE OR REPLACE VIEW v_fighter_record AS
WITH bouts AS (
    SELECT fighter1_id AS fighter_id, winner_id, win_method FROM fights WHERE winner_id IS NOT NULL
    UNION ALL
    SELECT fighter2_id AS fighter_id, winner_id, win_method FROM fights WHERE winner_id IS NOT NULL
)
SELECT
    fighter_id,
    count(*)                                                        AS decided_fights,
    count(*) FILTER (WHERE winner_id = fighter_id)                  AS wins,
    count(*) FILTER (WHERE winner_id <> fighter_id)                 AS losses,
    round(count(*) FILTER (WHERE winner_id = fighter_id)::numeric
          / NULLIF(count(*), 0) * 100, 1)                           AS win_pct,
    round(count(*) FILTER (
              WHERE winner_id = fighter_id
                AND (win_method ILIKE '%KO%' OR win_method ILIKE '%Submission%')
          )::numeric
          / NULLIF(count(*) FILTER (WHERE winner_id = fighter_id), 0) * 100, 1) AS finish_rate_pct
FROM bouts
GROUP BY fighter_id;
