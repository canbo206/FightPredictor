# FightPredictor

A command-line UFC analytics project that collects fight statistics, stores them
in PostgreSQL, and trains machine-learning models to predict fight winners and
finish methods.

## Features

- Collects event results, round statistics, and fighter profiles from UFCStats.
- Builds 23 matchup features covering offense, defense, experience, and physical attributes.
- Trains logistic regression models for the winner and finish method: KO/TKO,
  submission, or decision.
- Displays matchup probabilities and fighter stat comparisons in an interactive terminal session.

## Tech stack

Python, PostgreSQL, pandas, NumPy, scikit-learn, SQLAlchemy, Requests, and Beautiful Soup.

## Project structure

```text
FightPredictor/
├── data/
│   └── schema.sql          # Database tables and analysis views
├── models/                 # Saved models and scalers
├── queries/
│   └── metrics_view.sql    # Fighter metrics and record views for prediction
├── model.py                # Model training and interactive prediction
├── scraper.py              # UFCStats scraper and database inserts
├── requirements.txt        # Python dependencies
└── README.md
```

Scraped records live in PostgreSQL. The `data/` directory contains the SQL needed
to create the database structure.

## Installation

Install Python 3 and PostgreSQL. Start the PostgreSQL server and ensure `psql`
and `createdb` are available on your PATH.

```sh
git clone https://github.com/canbo206/FightPredictor.git
cd FightPredictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the remaining commands from the repository root with the virtual environment active.

## Database setup

The scripts currently connect to `ufc_analytics` on `localhost:5432` with user
`postgres` and password `ufc123`. To use your own credentials, update
`get_connection()` in both Python scripts and the SQLAlchemy connection URL in
`model.py` inside `load_fight_data()`.

Create and initialize a fresh database:

```sh
createdb -h localhost -p 5432 -U postgres ufc_analytics
psql -h localhost -p 5432 -U postgres -d ufc_analytics -v ON_ERROR_STOP=1 -1 -f data/schema.sql
psql -h localhost -p 5432 -U postgres -d ufc_analytics -v ON_ERROR_STOP=1 -1 -f queries/metrics_view.sql
```

Run `schema.sql` only against an empty database. Existing installations must
already have the fields and constraints defined in this schema; it is not a
migration script. `metrics_view.sql` creates or updates the views used for predictions.

## Usage

### Collect fight data

```sh
python scraper.py
```

The scraper checks up to 300 recent events, skips existing events, and pauses
between requests. It stops after five consecutive events already in the database.

### Train and predict

Before your first run, update the four `joblib.dump()` destinations near the end
of `model.py` to point to the `models/` directory in your checkout. These paths
are currently specific to the original development machine. Ensure the directory exists.

Once the database contains enough fights to train and evaluate both models:

```sh
python model.py
```

The script trains both models, prints evaluation results, saves the models and
scalers, and displays an example matchup. Enter two fighter names and the scheduled
rounds (3 or 5) to predict another matchup. Type `quit` at either fighter-name prompt
to exit. Both fighters must have statistics in the database.

Each run retrains the models and overwrites the saved `.pkl` files.

## Model methodology

The winner model uses differences between the two fighters across 23 features.
Historical performance features are calculated from fights preceding the bout
being predicted, and age is calculated at the fight date. Missing feature values
are filled with zero.

The finish-method model uses the absolute feature differences plus the scheduled
number of rounds. Both models use standardized features and logistic regression,
with a random 80/20 training and test split. The script reports winner accuracy,
a winner classification report, and finish-method accuracy.

## Limitations

- Evaluation uses a random split; chronological validation is needed to better
  assess predictions on future events.
- Missing or incomplete scraped statistics can affect predictions. Existing
  events are skipped, so rerunning the scraper does not repair partially imported events.
- Database credentials and model output paths currently require manual configuration.
- Dependencies are unpinned, and there is currently no automated test suite.

## License

MIT License. See [LICENSE](https://github.com/canbo206/FightPredictor/blob/main/LICENSE).
