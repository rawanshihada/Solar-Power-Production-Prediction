# Solar Power Production Prediction

**Recommendation: use the tuned Random Forest (`v2`), which predicts hourly PV output to a mean absolute error of 462.7 Wh under rolling-origin cross-validation — 59% better than a mean baseline (1120.8 Wh) and 3% better than the first version (476.9 Wh).** It is the promoted model, served at `POST /predict`, which returns `model_version` with every prediction.

The gain came from data cleaning and honest evaluation, not modelling: 21 consecutive days of undeclared plant outage were being learned as "bright sun → zero output", and a single winter test window was making a usable model look broken. No model was changed to fix either.

## Overview

This project focuses on predicting solar power production using weather and environmental data. The goal is to build a machine learning model that can estimate solar power generation based on factors such as solar radiation, sunshine, temperature, humidity, wind speed, and air pressure.

The project was developed as part of the **PSSAR Applied Research · Data & ML Bootcamp Capstone Project**.

## Problem Statement

Solar power production varies significantly depending on weather and environmental conditions. Accurate production prediction can help improve solar energy planning, monitoring, and operational decision-making.

This project aims to answer:

> **Can solar power production be predicted accurately using available weather and environmental features?**

**Framing caveat:** the model reads *same-hour* radiation and sunshine, so it is a **nowcast, not a forecast**. Predicting a future hour requires the caller to supply forecast weather for that hour, whose own error would compound with the 462.7 Wh reported here.

## Dataset

The dataset contains 8,760 hourly observations from a single solar power plant, covering 1 January – 31 December 2017.

### Features

| Feature               | Description                                     |
| --------------------- | ----------------------------------------------- |
| `Date-Hour(NMT)`      | Date and time of the observation                |
| `WindSpeed`           | Wind speed                                      |
| `Sunshine`            | Sunshine duration/intensity indicator           |
| `AirPressure`         | Atmospheric air pressure                        |
| `Radiation`           | Solar radiation                                 |
| `AirTemperature`      | Air temperature                                 |
| `RelativeAirHumidity` | Relative humidity                               |
| `SystemProduction`    | Solar power system production — target variable |

The target variable is:

```text
SystemProduction
```

It ranges from 0 to 7,701 Wh, and roughly 55% of hours are zero because half the readings are night-time.

## Project Workflow

The project follows a complete machine learning workflow:

1. Data loading and inspection
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature and target selection
5. Rolling-origin cross-validation setup
6. Baseline evaluation
7. Model training
8. Model evaluation and comparison
9. Hyperparameter tuning
10. Model selection and versioning
11. Model saving
12. API development

## Exploratory Data Analysis

The EDA stage was used to understand:

* Data types and structure
* Missing values
* Statistical distributions
* Relationships between weather variables and solar production
* Correlations between features
* Outliers and unusual observations

The analysis showed that **solar radiation has the strongest relationship with production**, and that **relative humidity is the strongest negative predictor** — it acts as a proxy for cloud cover. Air pressure carries no usable signal.

Correlations were computed twice, on all hours and on daylight hours only, because roughly half the dataset is night-time structural zeros that inflate any correlation with a daytime-driven variable.

### Data quality issues found and handled

| Issue | Evidence | Fix |
|---|---|---|
| Plant outage recorded as zero production | 3–23 May: production exactly 0 for every hour of 21 straight days, while mean May radiation was 187 W/m² — higher than March or April | Outage days detected by rule and removed (504 rows, 8,760 → 8,256) |
| Negative radiation readings | 4,464 rows below zero, minimum −9.3 W/m², a night-time sensor offset | Clipped at 0 during cleaning and again at serving time |
| Negative predictions | Tree ensembles can output small negative values | Clipped at 0 in the API |

`EDA.ipynb` writes the cleaned dataset to `Data/solar_clean.csv`, which `Modeling.ipynb` reads. Run EDA first if the cleaning logic changes.

## Machine Learning Models

The project evaluates different regression approaches for predicting `SystemProduction`.

### Models

* Mean baseline (predicts the training mean for every hour)
* Linear Regression
* Random Forest Regressor

## Model Evaluation

**MAE is the selection metric.** It is expressed in the target's own units, so "off by 462.7 Wh in a typical hour" is directly readable, and it is not dominated by a handful of large midday misses the way RMSE is.

R² is reported but not selected on. The original single chronological 80/20 split landed the entire test set in winter, where the target barely varies: a usable model scored R² = 0.223 there while the mean baseline scored −6.03. A metric that swings that hard should not decide anything.

**Validation uses `TimeSeriesSplit(n_splits=5)`, not a random split.** Consecutive hours are strongly autocorrelated, so a shuffled split places near-copies of test rows into training and reports an inflated score that measures leakage rather than skill. No fold ever trains on data that postdates its test window.

### Results

Mean across the five folds:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean baseline | 1120.8 | 1629.8 | −2.824 |
| Linear Regression | 515.4 | 891.4 | 0.493 |
| Random Forest (untuned) | 470.7 | 886.4 | 0.524 |
| **Random Forest (tuned) — promoted as v2** | **462.7** | — | — |

### Hyperparameter tuning

A randomised search over 20 combinations, scored on MAE across the same rolling-origin folds, selected `n_estimators=200`, `max_depth=16`, `min_samples_leaf=5`, `max_features=0.8`.

Every selected parameter *constrains* the model rather than enlarging it, which indicates the original configuration was overfitting, not underpowered. This is consistent with an earlier experiment where raising trees from 100 to 200 made performance worse — extra capacity was never the missing piece. The depth limit also shrank the saved artefact from 29 MB to 4.8 MB.

The improvement from v1 to v2 is modest (476.9 → 462.7, about 3%), and is reported as measured.

## Model Versions

| Version | Data | Features | Tuned | CV MAE |
|---|---|---|---|---|
| v1 | raw, 8,760 rows | 6 weather + Hour + Month | no | 476.9 |
| **v2** (promoted) | cleaned, 8,256 rows | 6 weather | yes | **462.7** |

`models/registry.json` records which version is live. The API reads each version's feature list from its own `metadata.json`, so both are served by the same code despite requiring different inputs.

## Leakage Checks

* No feature is derived from the target or from a future timestamp.
* All reported numbers use `TimeSeriesSplit`; no random split is used for any claim.
* Cleaning thresholds are fixed constants, not fitted on the target.
* Feature order is read from `metadata.json` at prediction time rather than hard-coded, so training and serving cannot drift apart.

## Project Structure

```text
Solar-Power-Production-Prediction/
│
├── Data/
│   ├── Solar Power Plant Data.csv     raw dataset
│   └── solar_clean.csv                cleaned export written by EDA.ipynb
│
├── Notebooks/
│   ├── EDA.ipynb                      exploration, data-quality findings, cleaning
│   └── Modeling.ipynb                 baselines, comparison, tuning, versioned export
│
├── models/
│   ├── registry.json                  which versions exist, which is promoted
│   ├── v1/model.pkl + metadata.json
│   └── v2/model.pkl + metadata.json
│
├── docs/
│   └── DEBUGGING_LOG.md               Symptom → Diagnosis → Fix entries
│
├── main.py                            FastAPI service
├── requirements.txt
├── .gitignore
└── README.md
```

Code and model artefacts are kept separate: nothing under `models/` is Python, and `main.py` hard-codes no version.

## API

The trained models are served through FastAPI in `main.py`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and model status |
| GET | `/versions` | All trained versions and which one is promoted |
| GET | `/metadata?model_version=` | Feature contract, CV score, environment |
| POST | `/predict?model_version=` | Predict production for one hour |

Inputs are validated against physical ranges (humidity 0–100%, pressure 850–1100 hPa, and so on). Out-of-range or missing fields return **422**; an unknown `model_version` returns **404**. Omitting `model_version` uses the promoted version.

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2017-07-15T12:00:00","WindSpeed":2.4,"Sunshine":60,
       "AirPressure":1013.2,"Radiation":780.5,"AirTemperature":24.1,
       "RelativeAirHumidity":42}'
```

```json
{"prediction": 3971.09, "unit": "Wh", "model_version": "v2"}
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rawanshihada/Solar-Power-Production-Prediction.git
cd Solar-Power-Production-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start the API:

```bash
uvicorn main:app --reload
```

Interactive documentation: <http://127.0.0.1:8000/docs>

The trained models are committed under `models/`, so the API returns predictions immediately — no training run is required.

For the complete analysis and model development process, open the notebooks inside the `Notebooks/` directory.

## Reproducibility

* Random seed fixed at 42 throughout.
* `requirements.txt` is pinned; `scikit-learn` is pinned exactly because the saved artefacts are pickles and loading them under a different version is not guaranteed.
* Each version's `metadata.json` records its feature contract, CV score, row count, seed and library version.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* FastAPI
* Git & GitHub

## Key Outcome

The project demonstrates an end-to-end machine learning workflow for solar power production prediction, from raw data analysis through model development, honest evaluation, versioning, and deployment behind a validated API.

The headline finding is methodological rather than algorithmic: cleaning the data and fixing the evaluation moved the results far more than any change to the model itself.

## Known Limitations

* One plant, one year of data; no cross-site generalisation is claimed.
* February shows 49% zero-production daylight hours. Unlike May, the zeros are scattered within days rather than spanning whole days, so the outage rule does not flag them. It may be a partial outage or snow cover. **Unresolved.**
* `v2` uses no time features at all, relying on radiation to carry the daily and seasonal signal. Adding `Hour` and `Month` back on the cleaned data is untested and may recover part of the gap.
* No lag or rolling-average features. Previous-hour production would likely improve accuracy but would change the API contract to require history.

## Future Improvements

* Additional weather and temporal features
* Testing additional regression algorithms
* Model monitoring after deployment
* Deploying the API to a public URL

## Author

**Rawan Jihad Shihada**

GitHub: [rawanshihada](https://github.com/rawanshihada)
