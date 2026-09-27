"""Point-in-time, data quality and symmetry checks for pre-fight features."""

import math
import unittest

import pandas as pd

import feature_engine as features


def bout(fight_id, date, first=1, second=2, winner=1, **changes):
    common = {
        "fight_id": fight_id, "event_id": fight_id, "event_date": date,
        "fighter1_id": first, "fighter2_id": second, "winner_id": winner,
        "win_method": "Decision - Unanimous", "scheduled_rounds": 3,
        "win_round": 3, "win_time": "5:00", "rounds": 3,
        "sig_l": 30, "sig_a": 60, "osig_l": 15, "osig_a": 40,
        "td_l": 2, "td_a": 4, "otd_l": 1, "otd_a": 3,
        "ctrl": 60, "kd": 1, "okd": 0, "sub": 1,
    }
    common.update(changes)
    rows = []
    for fighter_id in (first, second):
        rows.append({
            **common, "fighter_id": fighter_id, "name": f"Fighter {fighter_id}",
            "won": None if winner is None else int(fighter_id == winner),
            "dob": "1990-01-01", "reach_in": 70 + fighter_id, "height_in": 69,
        })
    return rows


class PointInTimeFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.rows = (
            bout(1, "2020-01-01", winner=1)
            + bout(2, "2020-07-01", winner=2, win_method="KO/TKO")
            + bout(3, "2021-01-01", winner=1)
        )

    def test_current_and_future_results_cannot_change_prefight_features(self):
        original = pd.DataFrame(self.rows)
        changed = original.copy()
        mask = changed["fight_id"] >= 2
        changed.loc[mask, "winner_id"] = 1
        changed.loc[mask, "won"] = changed.loc[mask, "fighter_id"].eq(1).astype(int)
        changed.loc[mask, ["sig_l", "sig_a", "kd", "ctrl"]] = [500, 900, 50, 300]
        changed.loc[mask, "win_method"] = "Submission"
        X, _, meta = features.build_training_data(original)
        changed_X, _, changed_meta = features.build_training_data(changed)
        pd.testing.assert_series_equal(X.loc[2], changed_X.loc[2])
        pd.testing.assert_series_equal(
            meta.loc[2, features.METHOD_FEATURE_NAMES],
            changed_meta.loc[2, features.METHOD_FEATURE_NAMES],
        )
        self.assertNotEqual(X.loc[3, "elo_diff"], changed_X.loc[3, "elo_diff"])

    def test_all_same_date_fights_use_one_prior_snapshot(self):
        rows = self.rows[:2] + bout(20, "2020-07-01", first=1, second=3, winner=1)
        rows += bout(21, "2020-07-01", first=1, second=4, winner=1)
        frame = pd.DataFrame(rows)
        X, _, _ = features.build_training_data(frame)
        self.assertEqual(X.loc[20, "elo_diff"], X.loc[21, "elo_diff"])
        self.assertEqual(X.loc[20, "total_fights_diff"], 1)
        self.assertEqual(X.loc[21, "total_fights_diff"], 1)
        # Changing a result earlier in fight_id order cannot affect the next
        # matchup on that same date, even when it reuses the same fighter.
        changed = frame.copy()
        changed.loc[changed["fight_id"] == 20, "winner_id"] = 3
        other_X, _, _ = features.build_training_data(changed)
        pd.testing.assert_series_equal(X.loc[21], other_X.loc[21])

    def test_live_profiles_match_training_snapshot_including_debut(self):
        frame = pd.DataFrame(self.rows)
        X, _, meta = features.build_training_data(frame)
        for fight_id, date in ((1, "2020-01-01"), (2, "2020-07-01"), (3, "2021-01-01")):
            with self.subTest(fight_id=fight_id):
                profiles = features.profiles_as_of(frame, date)
                actual = features.matchup_features(profiles[1], profiles[2])
                self.assertEqual(list(actual), features.FEATURE_NAMES)
                self.assertEqual(actual, X.loc[fight_id].to_dict())
                self.assertEqual(
                    features.method_features(profiles[1], profiles[2], 3),
                    meta.loc[fight_id, features.METHOD_FEATURE_NAMES].to_dict(),
                )

    def test_opponent_strength_and_elo_are_updated_after_result(self):
        frame = pd.DataFrame(self.rows)
        before = features.profiles_as_of(frame, "2020-01-01")
        after = features.profiles_as_of(frame, "2020-01-02")
        self.assertEqual(before[1]["elo"], 1500)
        self.assertEqual(after[1]["elo"], 1516)
        self.assertEqual(after[2]["elo"], 1484)
        self.assertEqual(after[1]["opponent_elo"], 1500)
        later = features.profiles_as_of(frame, "2020-07-02")
        self.assertAlmostEqual(later[1]["opponent_elo"], (1500 * 4 + 1500 + 1484) / 6)
        self.assertAlmostEqual(later[1]["elo"] + later[2]["elo"], 3000)

    def test_fighter_swap_negates_winner_features_and_preserves_method(self):
        profiles = features.profiles_as_of(pd.DataFrame(self.rows), "2021-07-01")
        first, second = profiles[1], profiles[2]
        forward = features.matchup_features(first, second)
        backward = features.matchup_features(second, first)
        for key in features.FEATURE_NAMES:
            self.assertEqual(forward[key], -backward[key])
        method = features.method_features(first, second, 5)
        self.assertEqual(list(method), features.METHOD_FEATURE_NAMES)
        self.assertEqual(method, features.method_features(second, first, 5))
        higher_pace1 = {**first, "sig_landed_per_min": first["sig_landed_per_min"] + 1}
        higher_pace2 = {**second, "sig_landed_per_min": second["sig_landed_per_min"] + 1}
        other = features.method_features(higher_pace1, higher_pace2, 5)
        self.assertAlmostEqual(other["sig_landed_per_min_sum"], method["sig_landed_per_min_sum"] + 2)

    def test_recency_and_inactivity_change_with_prediction_date(self):
        frame = pd.DataFrame(self.rows[:2])
        early = features.profiles_as_of(frame, "2020-01-02")[1]
        late = features.profiles_as_of(frame, "2022-01-01")[1]
        self.assertLess(late["recent_win_pct"], early["recent_win_pct"])
        self.assertEqual(late["win_pct"], early["win_pct"])
        self.assertGreater(late["layoff_days"], early["layoff_days"])
        self.assertGreater(late["age_years"], early["age_years"])
        self.assertEqual(late["last_fight_date"], "2020-01-01")


class DataQualityFeaturesTests(unittest.TestCase):
    def test_partial_final_round_uses_actual_elapsed_minutes(self):
        rows = bout(1, "2020-01-01", win_round=2, rounds=2, win_time="2:30")
        profile = features.profiles_as_of(pd.DataFrame(rows), "2020-01-02")[1]
        self.assertEqual(profile["observed_minutes"], 7.5)
        self.assertAlmostEqual(profile["sig_landed_per_min"], (30 + 15 * 3) / (7.5 + 15))
        self.assertAlmostEqual(profile["sig_net_per_min"], (30 - 15) / (7.5 + 15))
        self.assertAlmostEqual(profile["control_pct"], ((60 + 15 * 12) / (7.5 + 15)) / 60 * 100)
        self.assertEqual(profile["rate_coverage_pct"], 100)

    def test_unknown_or_invalid_duration_excludes_numerator_and_denominator(self):
        valid = bout(1, "2020-01-01")
        baseline = features.profiles_as_of(pd.DataFrame(valid), "2020-03-01")[1]
        for changes in (
            {"win_time": None}, {"win_time": "5:61"}, {"win_time": "6:00"},
            {"rounds": 1}, {"win_round": None}, {"scheduled_rounds": 1},
        ):
            with self.subTest(changes=changes):
                rows = valid + bout(2, "2020-02-01", sig_l=900, sig_a=1000, **changes)
                profile = features.profiles_as_of(pd.DataFrame(rows), "2020-03-01")[1]
                self.assertEqual(profile["sig_landed_per_min"], baseline["sig_landed_per_min"])
                self.assertEqual(profile["observed_minutes"], baseline["observed_minutes"])
                self.assertEqual(profile["rate_coverage_pct"], 50)
                self.assertEqual(profile["sample_size"], 2)

    def test_a_missing_count_does_not_add_exposure_to_that_metric(self):
        rows = bout(1, "2020-01-01") + bout(2, "2020-02-01", sig_l=None)
        profile = features.profiles_as_of(pd.DataFrame(rows), "2020-03-01")[1]
        self.assertEqual(profile["observed_minutes"], 30)
        self.assertEqual(profile["sig_landed_per_min"], (30 + 45) / (15 + 15))
        self.assertEqual(profile["rate_coverage_pct"], 50)

    def test_debut_and_missing_metadata_are_finite_and_explicit(self):
        frame = pd.DataFrame(bout(1, "2020-01-01"))
        frame[["dob", "reach_in", "height_in"]] = None
        profiles = features.profiles_as_of(frame, "2020-01-01")
        for profile in profiles.values():
            self.assertEqual(profile["sample_size"], 0)
            self.assertEqual(profile["rate_coverage_pct"], 0)
            self.assertIsNone(profile["last_fight_date"])
            for key in ("age_missing", "reach_missing", "height_missing", "layoff_missing"):
                self.assertEqual(profile[key], 1)
            self.assertTrue(all(math.isfinite(profile[key]) for key, _ in features.BASE_FEATURES))

    def test_nullable_results_are_not_losses_but_stats_remain_in_history(self):
        rows = bout(1, "2020-01-01", winner=None, win_method="No Contest")
        rows += bout(2, "2020-02-01", winner=1)
        frame = pd.DataFrame(rows)
        frame["winner_id"] = frame["winner_id"].astype("Int64")
        frame["won"] = frame["won"].astype("Int64")
        X, y, _ = features.build_training_data(frame)
        self.assertEqual(list(X.index), [2])
        self.assertEqual(list(y), [1])
        profile = features.profiles_as_of(frame, "2020-01-02")[1]
        self.assertEqual(profile["sample_size"], 1)
        self.assertEqual(profile["decided_fights"], 0)
        self.assertEqual(profile["win_pct"], 50)
        self.assertEqual(profile["elo"], 1500)
        self.assertEqual(profile["observed_minutes"], 15)

    def test_invalid_stat_counts_cannot_create_extreme_rates_or_percentages(self):
        rows = bout(1, "2020-01-01", sig_l=99, sig_a=2, ctrl=9999, kd=-1)
        profile = features.profiles_as_of(pd.DataFrame(rows), "2020-01-02")[1]
        self.assertEqual(profile["sig_accuracy_pct"], 45)
        self.assertEqual(profile["sig_landed_per_min"], 3)
        self.assertEqual(profile["control_pct"], 20)
        self.assertAlmostEqual(profile["kd_per15"], 0.3)

    def test_empty_dataset_and_duplicate_history_rows(self):
        X, y, meta = features.build_training_data(pd.DataFrame())
        self.assertTrue(X.empty and y.empty and meta.empty)
        self.assertEqual(list(X.columns), features.FEATURE_NAMES)
        self.assertEqual(features.profiles_as_of(pd.DataFrame(), "2020-01-01"), {})
        rows = bout(1, "2020-01-01")
        with self.assertRaisesRegex(ValueError, "one history row"):
            features.build_training_data(pd.DataFrame(rows + [rows[0]]))


if __name__ == "__main__":
    unittest.main()
