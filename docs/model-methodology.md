# Model methodology: feature version 2

This implementation follows public sports-forecasting principles. It does not
copy a trading firm's proprietary model, and a feature difference is not evidence
of a profitable bet. The new pipeline remains experimental: improvements in
validation and consistency do not guarantee improved forecasts.

## What changed

The original model used 23 mostly per-round career-stat differences, filled
missing differences with zero, and evaluated random train/test splits. Its live
statistics came from separate SQL views. The website marked statistics as edges
using fixed rules, including younger age being better.

The winner model now uses 32 differences from a shared pre-fight feature engine:

- Significant strikes landed, absorbed, attempted, and net difference per minute.
- Strike accuracy and defense; takedown accuracy and defense.
- Takedowns landed/conceded, knockdowns landed/absorbed, and submission attempts
  per 15 minutes; control as a percentage of elapsed fight time.
- Recent net striking and recent win rate; tracked win and finish win/loss rates.
- Elo and average pre-fight opponent Elo, which add opponent-strength context.
- Age at prediction date, reach, height, time since the last tracked fight,
  tracked fight count, observed minutes, and rate-data coverage.
- Explicit missing-value indicators for age, reach, height, and prior fight date.

These rates describe the history available in this database, not necessarily a
fighter's full UFC or professional career. The inspected database starts in March 2019.
Elo is opponent-adjusted, but the individual striking/grappling rates are not a
full opponent-adjusted latent-skill model. Weight-class interactions, stance
matchups, nonlinear age effects, and additional model families are not implemented.

The finish model uses 46 inputs: 32 absolute differences, 13 sums of relevant
activity/vulnerability and exposure features, and scheduled rounds. The sums
distinguish two high-output fighters from two low-output fighters with the same
relative difference. Its probabilities are for the finish method regardless of winner.

## Exposure, missing data, and assumptions

A round-2 finish at 2:30 contributes 7.5 minutes. Unknown or invalid finish times,
nonstandard schedules, and mismatched captured-round counts are excluded from
timed rates. Available attempt-based percentages can still use valid counts.
Each metric excludes its own missing numerator and exposure together. A numeric
zero already stored by the scraper cannot reliably be distinguished from a missing
value that was replaced with zero upstream.

Sparse observations shrink toward fixed reference values, never averages computed
from the final test period. These are transparent modeling assumptions, not fitted
population estimates or proven optimal constants:

- 15 prior minutes for activity rates; baseline significant strikes landed and
  absorbed are 3/min and attempts 7.5/min.
- 40 prior striking attempts at 45% accuracy; four prior takedown attempts at 35%.
- Four prior decisions for result-rate smoothing: 50% win rate and 25% finish-win
  and finish-loss rates per decided fight.
- Recent observations decay with a 730-day half-life.
- Elo starts at 1500, uses K=32 and a 400-point scale; opponent strength uses
  opponent ratings before those bouts, not the opponents' eventual ratings.
- Missing age defaults to 30 years, reach/height to 70 inches, and layoff to
  180 days, each accompanied by its own missing indicator.

All defaults and other rate priors are visible in `feature_engine.py`.
The source lacks dated snapshots of physical attributes. Current stored
height/reach/DOB are assumed known historically; that limitation cannot be removed
by chronological splitting alone.

## Validation and deployment

An entire event date is frozen before calculating its matchup features. Histories
update only afterward. This avoids using the bout's own statistics or another
bout from the same day. Draw/no-contest statistics can enter later histories, but
unknown outcomes are not coded as losses or winner training targets.

Approximately 60% of fights train the model, the following 20% calibrate it, and
the final 20% evaluate it. Whole dates stay together. Feature scaling is fitted
only on training data. Regularization is fixed at C=0.25; the test set is not used
to select hyperparameters. Results include accuracy, log loss, Brier score,
reliability bins, expected calibration error, and train-prior/Elo benchmarks.

Winner training includes both fighter orders, with half weight for each copy,
no intercept, and uncentered scaling. Scalar temperature calibration preserves
the property P(A beats B) = 1 - P(B beats A). A temperature is fitted on the
calibration period, never the final test period. Calibration can improve one
metric while worsening another, so both raw and calibrated scores are retained.

Saved artifacts retain the evaluated training-window parameters and calibration
temperature. They are not secretly refitted on the final test period. Historical
profiles for later test/live fights can use earlier completed test-period bouts,
as those results would have been known then; model weights remain fixed.

The UI's model lean is the direction of a standardized feature times its fitted,
temperature-adjusted coefficient. It is an explanation of this model's output,
not a causal claim or a bookmaker-price comparison. Correlated features can split
or distort individual contributions. The former 0–10 confidence heuristic has
been replaced with tracked-history counts.

## Recorded evaluation (September 26, 2026)

Training covers 2,254 decided fights through October 14, 2023. Calibration covers
752 through April 12, 2025. The untouched test contains 757 fights from April 26,
2025 through September 19, 2026.

The new winner model scored 60.63% accuracy, 0.67482 log loss, and 0.23813 Brier.
Old features under the same symmetric/calibrated training scored 60.24%, 0.67664,
and 0.24142. That is a small feature-level improvement, not proof of a stable gain.

The original estimator with its original features, retrained on the same earlier
training window, scored 60.50%, 0.65912, and 0.23299. Its probability scores were
better, even though that estimator does not enforce fighter-order symmetry. This
new version therefore has not established overall superiority to the original.

Finish-method accuracy was 46.90%, versus a 45.84% train-prior baseline; calibrated
log loss was 1.01935. Winner calibration reduced log loss from 0.68243 to 0.67482,
but Brier worsened from 0.23640 to 0.23813 and ECE from 0.04495 to 0.04813. That
mixed result is why this project does not label its probabilities as guaranteed
accurate. Do not repeatedly tune to this same test period and keep calling it untouched.

`models/evaluation.json` stores the current run, including sample counts and
reliability bins. `models/baseline_comparison.json` is the fixed comparison above,
with the original Git revision. Retraining refreshes the first file, not the
historical baseline comparison.

## What would establish a betting edge?

The database has no historical offered odds or timestamps. We cannot evaluate
profitability, market-relative calibration, or closing-line value from fight
statistics alone. A future odds dataset should record both sides, bookmaker,
quote time, prediction time, opening/closing prices, and actual available prices.
Injuries, camp changes, short-notice replacements, and weight-cut information
would also require additional timestamped data sources.

Given decimal odds dA and dB, one simple proportional margin removal is
`market_A = (1 / dA) / (1 / dA + 1 / dB)`. The probability gap is
`model_A - market_A`; expected net return per unit at offered odds is
`model_A * dA - 1`. These are estimates conditional on model quality and quoted
prices, not a demonstrated profit. They are documented here, not implemented as
recommendations or wagers.

## Sources

- [Holmes, McHale & Żychaluk: A Markov chain model for forecasting results of mixed
  martial arts contests](https://livrepository.liverpool.ac.uk/3154619/) motivates
  attack/defense skill estimates, shrinkage, future-period validation, and comparison
  with bookmaker prices. This project does not reproduce its Markov simulation.
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
  explains separate calibration data, reliability diagrams, and proper scoring rules.
- [scikit-learn: time-series validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  explains why training on future observations to evaluate earlier observations is
  inappropriate. This implementation groups event dates explicitly because fights
  are not equally spaced observations.
- [Pinnacle: interactive betting tools](https://www.pinnacle.com/en/corporate/press/press-release/pinnacle-launches-interactive-betting-tools)
  describes the bookmaker's public tools for implied probability and margin comparison.

Elo, layoff, recency, and the constants above are testable engineering choices.
These sources do not establish that any particular firm uses this exact feature
set, weights, smoothing constants, or betting strategy.
