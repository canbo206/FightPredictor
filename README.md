# FightPredictor

A UFC analytics project that collects fight statistics, stores them
in PostgreSQL, and trains machine-learning models to predict fight winners and
finish methods through a terminal or a local web interface.

## Features

- Collects event results, round statistics, and fighter profiles from UFCStats.
- Builds 32 pre-fight features covering time-adjusted performance, opponent
  strength, recent form, inactivity, physical attributes, and data coverage.
- Trains logistic regression models for the winner and finish method: KO/TKO,
  submission, or decision.
- Displays matchup probabilities and fighter stat comparisons in an interactive terminal session.
- Provides a local website with fighter suggestions, probability bars, finish-method
  predictions, tracked fight counts, and learned feature contributions.
- Evaluates later fights with chronological splits, probability calibration,
  log loss, Brier score, accuracy, and reliability bins.

## Tech stack

Python, PostgreSQL, pandas, NumPy, SciPy, scikit-learn, Requests, and Beautiful Soup.

## Project structure

```text
FightPredictor/
├── data/
│   └── schema.sql          # Database tables and analysis views
├── models/                 # Saved models, scalers, and evaluation.json
├── docs/                   # Feature methodology and code-change explanations
├── queries/
│   └── metrics_view.sql    # Optional SQL exploration views
├── web/                    # Website HTML, CSS, JavaScript, and favicon
├── tests/                  # Prediction and local API checks
├── app.py                  # Local website server
├── feature_engine.py       # Shared historical and live feature calculations
├── model.py                # Model training and interactive prediction
├── training.py             # Chronological evaluation and calibration
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

The default connection is `localhost:5432`, database `ufc_analytics`, user
`postgres`, and password `ufc123`. Override it with `DATABASE_URL` or the standard
`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` environment variables.
`database.py` shares these settings across the scraper and model.

Create and initialize a fresh database:

```sh
createdb -h localhost -p 5432 -U postgres ufc_analytics
psql -h localhost -p 5432 -U postgres -d ufc_analytics -v ON_ERROR_STOP=1 -1 -f data/schema.sql
psql -h localhost -p 5432 -U postgres -d ufc_analytics -v ON_ERROR_STOP=1 -1 -f queries/metrics_view.sql
```

Run `schema.sql` only against an empty database. Existing installations must
already have the fields and constraints defined in this schema; it is not a
migration script. `metrics_view.sql` is optional for SQL exploration; the current
predictor calculates its features directly from historical fight records.

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

To train, evaluate, and exit without starting the interactive prompts:

```sh
python model.py --train-only
```

The version-2 feature definitions require retraining older model files. Training
also writes `models/evaluation.json` with date boundaries, feature lists, baseline
scores, and calibration results. The saved models retain the evaluated training
and calibration windows; final test outcomes are not used to fit parameters.

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
the website after retraining or scraping to refresh saved models and cached fighter histories.

Use `python app.py --port 8001` if port 8000 is occupied. Press `Ctrl+C` in the
server terminal to stop the website. This server listens only on your computer;
it is a local interface, not a production hosting setup.

## Model methodology

The winner model uses 32 differences between fighter profiles. Striking rates
use actual elapsed minutes; takedowns, knockdowns, and submission attempts use
15-minute rates. Elo and prior opponent ratings add competition context. Recent
form, layoff, age, reach, and explicit missing-data indicators add context beyond
career averages. Sparse statistics shrink toward fixed, documented reference values.

Every historical profile uses only earlier event dates. Fights on the same day
are updated together after predictions, and live inference uses the same feature
engine. The database's tracked history may be shorter than a fighter's career.

Whole event dates are split into approximately 60% training, 20% calibration,
and 20% testing in chronological order. Logistic regression is fitted only on
the training window; a scalar temperature adjusts probabilities on the calibration
window. Mirrored training and an uncentered scaler make swapping fighters invert
winner probabilities exactly. The finish model uses 46 order-invariant inputs:
absolute differences, shared activity/vulnerability levels, and scheduled rounds.

The UI's **model lean** follows fitted coefficients, rather than assuming that
larger stats or younger age always help. It describes the model, not a causal
effect or a betting recommendation. See [feature methodology](docs/model-methodology.md)
for assumptions, source links, results, and limitations, and
[code-change explanations](docs/changes-explained.md) for short annotated snippets.

## Limitations

- This is an experimental, research-informed model, not a reproduction of a
  betting firm's proprietary system. No bookmaker odds are stored, so profitability,
  market superiority, and closing-line value have not been evaluated.
- More features do not establish better forecasts: the original model scored
  better on some probability metrics in the recorded comparison.
- Missing or incomplete scraped statistics can affect predictions. Existing
  events are skipped, so rerunning the scraper does not repair partially imported events.
- Database credentials can be configured with environment variables.
- Historical height/reach/DOB snapshots, injuries, camp changes, short-notice
  bookings, and weigh-in information are not available. Static physicals are
  assumed known; missing measurements use defaults plus indicator features.
- Calibration is evaluated, not guaranteed. One held-out period is not proof of
  stable performance; repeatedly tuning against it would invalidate that claim.
- `scikit-learn` is pinned to 1.8.0 to match the saved model artifacts; upgrade it together with retraining.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

These checks cover temporal leakage, feature parity, partial-round duration,
fighter-swap symmetry, missing data, calibration, artifact compatibility, prediction
output, and API errors without requiring a running PostgreSQL database.

## License

MIT License. See [LICENSE](https://github.com/canbo206/FightPredictor/blob/main/LICENSE).

## API, containers, and automation

The original `python app.py` command still works. A new **FastAPI** adapter serves
that same website and prediction engine, with validated requests and generated
OpenAPI documentation at `/docs`:

```bash
bash scripts/dev.sh setup
bash scripts/dev.sh test
bash scripts/dev.sh serve
# In a second terminal:
bash scripts/dev.sh health
```

Windows / PowerShell equivalent:

```powershell
./scripts/dev.ps1 -Action setup
./scripts/dev.ps1 -Action test
./scripts/dev.ps1 -Action serve -Port 8000
# In a second terminal:
./scripts/dev.ps1 -Action health -Port 8000
```

The scripts create a virtual environment, install dependencies, run tests, start
the API, and check its health endpoint. Bash uses `PORT=8001` to change the port;
PowerShell uses `-Port 8001`. `PROJECT_PYTHON` can point at an existing Python
interpreter. `/health` checks the web process only; predictions also need the
PostgreSQL data and compatible trained models. No request retrains the model.

Small skills statement: **Wrote Bash and PowerShell scripts to automate project
setup, tests, local startup, and HTTP health checks.** Run both before describing
this as hands-on cross-platform experience.

### Docker Compose (local development)

`Dockerfile` packages the FastAPI server; `compose.yaml` defines it and PostgreSQL
with a persistent database volume. Both published ports bind to your computer.
The API container runs as a non-root user and mounts saved models read-only.

1. Install Docker with Compose. Copy `.env.example` to `.env` and choose a local
   database password. Do not commit `.env`.
2. Run `docker compose config --quiet`, then `docker compose up --build -d`.
3. Open `http://127.0.0.1:8000/docs`. The database on port **5433** starts empty,
   separate from your original PostgreSQL database on port 5432. The schema is
   initialized only on first volume creation; existing data is not imported.
4. To populate the new database, install the Python dependencies locally and run
   the existing scraper and trainer against it. In Bash:

   ```bash
   export PGHOST=127.0.0.1 PGPORT=5433 PGDATABASE=ufc_analytics PGUSER=postgres
   read -r -s -p 'Compose database password: ' PGPASSWORD; echo
   export PGPASSWORD
   .venv/bin/python scraper.py
   .venv/bin/python model.py --train-only
   docker compose restart api
   ```

   In PowerShell, set the same variables using `$env:PGHOST = "127.0.0.1"`,
   `$env:PGPORT = "5433"`, etc., then use `.venv/Scripts/python.exe`.
   An already-set `DATABASE_URL` takes precedence over PG variables: unset it
   before using this PG-based example. Scraping accesses UFCStats and training
   replaces saved artifacts, so neither is part of setup, tests, or CI.
5. `docker compose down` stops services and preserves the database volume.
   Avoid `down -v` unless you intend to delete that volume.

The **YAML** workflow `.github/workflows/checks.yml` is configured to test the
Bash scripts on Linux, PowerShell scripts on Windows, and the image on Linux when
pushed to GitHub. It runs tests and container liveness, not scraping, retraining,
or deployment. Cloud hosting is not configured.

Learn: [FastAPI](https://fastapi.tiangolo.com/tutorial/),
[Docker Compose](https://docs.docker.com/compose/gettingstarted/),
[GitHub Actions](https://docs.github.com/en/actions/get-started/quickstart),
[Bash](https://www.gnu.org/s/bash/manual/html_node/Shell-Scripts.html),
[PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/00-introduction).
