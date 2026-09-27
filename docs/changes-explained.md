# Changes explained

Each explanation below is followed by a short excerpt of the changed code.
The excerpts show the relevant lines, not complete files.

## Earlier project cleanup

### Database setup: `data/schema.sql`

Kept the database setup because the scraper and predictor still need PostgreSQL.
Added the birth-date field and uniqueness rules required by the scraper.

```sql
name          TEXT NOT NULL UNIQUE,
dob           DATE,
```

```sql
CONSTRAINT fights_event_fighters_key UNIQUE (event_id, fighter1_id, fighter2_id)
```

### Setup documentation: `README.md`, `requirements.txt`, `.gitignore`

Added installation and usage instructions, declared Python dependencies, and
ignored local virtual environments. References to the SQL files you deleted
were removed from the README.

```gitignore
.venv/
venv/
```

### Local website: `app.py`

Added a local server that serves the website and calls the same predictor as the
terminal. Changing a matchup does not retrain the model or run the scraper.

```python
self.run_api(lambda: model.get_matchup_prediction(
    *load_models(), payload.get("fighter1"), payload.get("fighter2"), payload.get("rounds", 3)))
```

### Beginner-friendly styling: `web/style.css`

Expanded the CSS into readable blocks, added explanatory comments, and grouped
mobile rules at the bottom. Colors and layout rules were preserved.

```css
/*
  3. Main page container
  Center the content and limit its width on large screens.
*/
.shell {
  width: min(1080px, calc(100% - 64px));
  margin: auto;
}
```

### Model output paths: `model.py`

Model files are saved relative to the project, so the code no longer depends on
one person's Mac username or checkout directory.

```python
MODEL_DIR = Path(__file__).resolve().parent / "models"
```

### Restored tests: `tests/`

Recreated the accidentally deleted test file and verified it. Tests are optional
for running the website, but help catch regressions after edits.

```sh
python3 -m unittest discover -s tests -v
```

## Current forecasting update

### 1. Actual fight time: `feature_engine.py`

Rates now account for partial rounds. A round-2 finish at 2:30 is 7.5 minutes,
rather than two complete rounds. Invalid or missing times are not guessed.

```python
elapsed = (final_round - 1) * 300 + minutes * 60 + seconds
return elapsed / 60 if elapsed > 0 else None
```

### 2. More useful feature families: `feature_engine.py`

Replaced the old 23-feature list with 32 inputs, including time-adjusted rates,
opponent strength, recent form, layoff, and data-coverage indicators.

```python
("sig_net_per_min", "Net sig. strikes / min"),
("elo", "Opponent-adjusted Elo rating"),
("opponent_elo", "Prior opponent strength (Elo)"),
("layoff_days", "Days since last tracked fight"),
```

### 3. Small-sample smoothing: `feature_engine.py`

One short fight should not create an extreme skill estimate. Rates blend
observations with fixed reference values, whose influence fades as data grows.

```python
return (numerator + PRIOR_MINUTES * RATE_PRIORS[key]) / (minutes + PRIOR_MINUTES)
```

### 4. Opponent strength and recent form: `feature_engine.py`

Elo increases more for beating a stronger opponent. Recent results decay over
time, so older observations have less influence on the recent-form features.

```python
expected = 1 / (1 + 10 ** ((ratings[second_id] - ratings[first_id]) / 400))
delta = ELO_K * (score - expected)
```

```python
return 0.5 ** ((date - self.last_date).days / RECENT_HALF_LIFE_DAYS)
```

### 5. Past information only: `feature_engine.py`

Historical profiles exclude the fight being predicted and all other fights on
that date. This prevents later results from leaking into earlier predictions.

```python
for date, rows in _dates(records):
    if date >= cutoff:
        break
    _observe_date(rows, histories)
```

### 6. Shared training and website features: `model.py`

Training and live predictions now use the same feature engine, replacing the
separate SQL-view calculations used for live predictions previously.

```python
feature_row = pd.DataFrame([matchup_features(first, second)], columns=FEATURE_NAMES)
```

### 7. Chronological validation: `training.py`, `model.py`

Replaced random splitting with earlier training events, a later calibration
period, and a final future-event test. Whole event dates stay together.

```python
split = chronological_split(meta["event_date"])
winner, scaler, winner_report = train_winner(features, labels, meta["event_date"], split=split)
method, method_scaler, method_report = train_method_model(features, meta, split=split)
```

### 8. Fighter-order consistency: `training.py`

Training includes both fighter orderings. With no intercept and no mean-centering,
swapping the inputs flips the winner probability instead of changing the matchup.

```python
train_features = pd.concat([train_features, -train_features], ignore_index=True)
train_labels = pd.concat([train_labels, 1 - train_labels], ignore_index=True)
```

### 9. Probability calibration and scoring: `training.py`

A separate period adjusts how extreme probabilities are. Final evaluation
reports probability errors and reliability, not just correct winner picks.

```python
temperature = _fit_temperature(base_model, scaled_calibration, labels.iloc[calibration])
model = TemperatureModel(base_model, temperature, features.columns.to_numpy())
```

### 10. Finish-method inputs: `feature_engine.py`

Added shared activity levels as well as differences. Two active finishers should
not look identical to two low-output fighters merely because their differences match.

```python
METHOD_FEATURE_NAMES = (
    ["abs_" + name for name in FEATURE_NAMES]
    + [key + "_sum" for key in METHOD_LEVEL_KEYS]
    + ["scheduled_rounds"]
)
```

### 11. Learned contributions in the interface: `model.py`, `web/app.js`, `web/index.html`

Replaced fixed rules such as “younger is better” with the direction learned by
the model. Removed the heuristic confidence score and show tracked fight counts.
The page explicitly distinguishes a model contribution from a betting edge.

```python
contributions = scaler.transform(feature_row)[0] * model.coef_[0]
```

```javascript
$("tracked-fights").textContent = `${result.tracked_fights.fighter1} / ${result.tracked_fights.fighter2}`;
if (stat.edge && stat.edge !== "even") edge = result[stat.edge];
```

### 12. Compatible model files: `app.py`, `models/`

Retrained the four saved files and added version/name checks so an old model
cannot silently receive a different feature list. Evaluation results are saved
in JSON alongside the models.

```python
if (getattr(artifact, "feature_version_", None) != model.FEATURE_VERSION
        or list(getattr(artifact, "feature_names_in_", [])) != names):
    raise ValueError("Saved model feature definitions do not match the predictor.")
```

### 13. Dependencies, checks, and documentation

Added SciPy as an explicit dependency for calibration. Added tests for time leakage,
partial rounds, missing data, symmetry, serialization, and calibration isolation.
Updated the README and documented public sources and the mixed benchmark results.

```python
pd.testing.assert_series_equal(X.loc[2], changed_X.loc[2])
```

This test changes current/future fight data and checks that the pre-fight
features stay identical. Results and limitations are explained in
[model-methodology.md](model-methodology.md), including where the original model
still outperformed this version. No profitability claim is made.
