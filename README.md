# FightPredictor

A UFC analytics project that collects fight statistics, stores them
in PostgreSQL, and trains machine-learning models to predict fight winners and
finish methods through a terminal or a local web interface.

## Features

- Collects event results, round statistics, and fighter profiles from UFCStats.
- Builds 23 matchup features covering offense, defense, experience, and physical attributes.
- Trains logistic regression models for the winner and finish method: KO/TKO,
  submission, or decision.
- Displays matchup probabilities and fighter stat comparisons in an interactive terminal session.
- Provides a local website with fighter suggestions, probability bars, finish-method
  predictions, confidence scores, and a side-by-side statistics table.

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
├── web/                    # Website HTML, CSS, JavaScript, and favicon
├── tests/                  # Prediction and local API checks
├── app.py                  # Local website server
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

Once the database contains enough fights to train and evaluate both models:

```sh
python model.py
```

The script trains both models, prints evaluation results, saves the models and
scalers, and displays an example matchup. Enter two fighter names and the scheduled
rounds (3 or 5) to predict another matchup. Type `quit` at either fighter-name prompt
to exit. Both fighters must have statistics in the database.

Each run retrains the models and overwrites the saved `.pkl` files in the project's
`models/` directory, which is created automatically.

### Open the website

With PostgreSQL running and the Python dependencies installed:

```sh
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Choose two
fighters, select three or five rounds, and click **Analyze matchup**. If available
in your database, the page initially shows the same Islam Makhachev vs. Dustin
Poirier example as the terminal.

The website uses the same prediction function as the terminal and loads the four
saved model/scaler files once. It does not scrape or retrain on page load. If the
saved models are missing or incompatible, run `python model.py` first. Restart
the website after retraining to load the updated models.

Use `python app.py --port 8001` if port 8000 is occupied. Press `Ctrl+C` in the
server terminal to stop the website. This server listens only on your computer;
it is a local interface, not a production hosting setup.

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
- Database credentials currently require manual configuration.
- The displayed confidence score measures distance from a 50/50 winner prediction;
  it is not a calibrated measure of model accuracy.
- Dependencies are unpinned.

## Tests

```sh
python -m unittest discover -s tests -v
```

These checks cover prediction output, invalid matchups, resource cleanup, saved
model compatibility, and API responses without requiring a running PostgreSQL database.

## License

MIT License. See [LICENSE](https://github.com/canbo206/FightPredictor/blob/main/LICENSE).
