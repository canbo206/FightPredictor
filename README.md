# FightAnalyze

A Python and PostgreSQL project for collecting UFC fight statistics, exploring
fighter performance with SQL, and predicting fight winners and finish methods
from the terminal. There is currently no dashboard or web application.

## What it does

- Scrapes event results, round statistics, and fighter profiles from UFCStats.
- Stores fighters, events, fights, and round statistics in PostgreSQL.
- Trains logistic regression models for the winner and finish method
  (KO/TKO, submission, or decision).
- Offers an interactive matchup predictor with probabilities and stat comparisons.

Training features use each fighter's statistics from before the fight. Evaluation
currently uses a random 80/20 train/test split, rather than a chronological holdout.
Reported accuracy is an experiment result, not a guarantee of future performance.

## Project layout

```text
FightAnalyze/
├── data/
│   ├── schema.sql              # Database tables and analysis views
│   └── seed_data.sql           # Legacy filename: a SELECT query, not seed records
├── models/                     # Saved models and scalers (.pkl)
├── queries/
│   ├── analysis_queries.sql    # SQL examples for exploring fight statistics
│   └── metrics_view.sql        # Fighter metrics and record views for prediction
├── model.py                    # Training and interactive matchup prediction
├── scraper.py                  # UFCStats scraper and database inserts
├── requirements.txt            # Python dependencies
└── README.md
```

The `data/` folder is still needed without a dashboard: `schema.sql` defines the
database used by both Python scripts. Actual scraped records live in PostgreSQL,
not in this folder. `seed_data.sql` is optional and only displays fighter averages;
it does not populate the database. The analysis views also support the SQL examples.

Local checkouts may additionally contain ignored maintenance files such as
`backfill.py`, `queries/add_metric_columns.sql`, and
`queries/cleanup_and_constraints.sql`. These are not included in a fresh clone.

## Setup

Install Python 3 and PostgreSQL, start the PostgreSQL server, and make sure `psql`
and `createdb` are on your PATH. Run commands from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are currently unpinned; there is no locked, reproducible environment yet.

### Database connection

The current scripts use database `ufc_analytics` on `localhost:5432`, with user
`postgres` and password `ufc123`. To use your own local credentials, update
`get_connection()` in both `scraper.py` and `model.py`, plus the SQLAlchemy URL
inside `load_fight_data()` in `model.py`. Environment variables are not currently
read by these scripts.

### Create a fresh database

The following commands assume the connection settings above. PostgreSQL will ask
for the role's password when authentication requires it.

```sh
createdb -h localhost -p 5432 -U postgres ufc_analytics
psql -h localhost -p 5432 -U postgres -d ufc_analytics -v ON_ERROR_STOP=1 -1 -f data/schema.sql
psql -h localhost -p 5432 -U postgres -d ufc_analytics -v ON_ERROR_STOP=1 -1 -f queries/metrics_view.sql
```

Run `schema.sql` once against an empty database. It is not an upgrade script for
an existing database. Existing installations need the same `fighters.dob` field,
expanded stance values, and unique constraints on fighter names, event names,
and `(event_id, fighter1_id, fighter2_id)` that the current schema defines.
The local maintenance scripts above may help upgrade an older installation;
review them and back up the database first, as the cleanup script deletes duplicates.

## Collect fight statistics

```sh
python scraper.py
```

The scraper checks up to 300 recent events, skips events already in the database,
and stops after five consecutive existing events. It pauses between requests.
An existing event is skipped even if only partially imported, so rerunning is
not a repair mechanism for incomplete events.

## Train and predict

Before training on a different checkout, change the four `joblib.dump()` output
paths near the bottom of `model.py`: they currently point to
`/Users/canbo/FightAnalyze/models/`. Ensure the destination directory exists.

After collecting enough fights for training and evaluation:

```sh
python model.py
```

Each run trains both models, prints evaluation results, and overwrites the four
saved model/scaler files. It then shows an example matchup and prompts for fighter
names and a three- or five-round bout. Type `quit` at either fighter-name prompt
to exit. Fighters must have statistics in the database to be found.

The saved `.pkl` files are currently tracked in Git. The script retrains on every
run rather than loading them automatically.

## Explore with SQL

Run individual examples from `queries/analysis_queries.sql` in your SQL client.
These cover fighter averages, head-to-head results, round momentum, and weight-class
benchmarks. The SQL `v_win_probability` view is a weighted statistical heuristic,
separate from the trained Python model.

## Development and commits

Keep each commit focused on one finished change. Example messages:

```text
docs: explain database setup and prediction workflow
fix: align fresh database schema with scraper requirements
chore: declare Python dependencies and ignore virtual environments
```

There is currently no automated test suite in the repository.
