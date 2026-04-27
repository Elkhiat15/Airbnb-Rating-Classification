# Classifying Airbnb Listing Ratings Using Property Features

---

## Team

| Name | ID |
|---|---|
| Mohamed ElKhayat | 9211015 |
| Abdallah Salah | 9220478 |
| Mohamed Sayed | 9220695 |
| Abdelrahman Samy | 9220430 |

---

## Project Description

This project develops a supervised machine learning system that predicts the
rating category of an Airbnb listing based on its structural property
features — such as room type, price, amenities, accommodation capacity, and
location. The dataset combines **74,111 records from Kaggle** with **1,000
listings collected via web scraping**, resulting in a final dataset of
**75,111 rows** across 16 features.

The target variable (`review_scores_rating`) is binned into three categories:
**Medium Rating**, **High Rating**, and **Very High Rating**. Six classifiers
were trained and evaluated — CatBoost, Random Forest, XGBoost, Histogram
Gradient Boosting, Logistic Regression, and KNN — alongside a dummy baseline,
under four class-balancing strategies.

---

**Project structure `Cookie-Cutter folder structure.`**

```
Airbnb-Rating-Classification
├── cleaning
│   ├── __init__.py
│   ├── cleaning.ipynb
│   └── cleaning.py
├── data
│   ├── merged
│   │   └── merged_airbnb_data.csv
│   ├── processed
│   │   ├── cleaned.csv
│   │   ├── featured.csv
│   │   ├── ready_features.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw
│       ├── Airbnb_Data.csv
│       └── scraped_airbnb_listings.csv
├── eda
│   ├── dashboard.py
│   └── visualize.py
├── eda.ipynb
├── feature_engineering
│   ├── __init__.py
│   ├── engineering.py
│   ├── features.ipynb
│   ├── pipeline.py
│   ├── selection.py
│   └── transformations.py
├── Makefile
├── modelling
│   ├── __init__.py
│   ├── baseline.py
│   ├── class_balancing.py
│   ├── config.py
│   ├── evaluate.py
│   ├── train_enhanced.py
│   ├── train_single.py
│   └── train.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── scraper
│   ├── merge.ipynb
│   ├── merge.py
│   └── scraper.ipynb
├── tests
│   ├── __init__.py
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_merge.py
│   ├── test_modelling.py
│   └── test_training.py
├── validation
│   ├── __init__.py
│   ├── validation.ipynb
│   └── validation.py
└── validation_report
    └── validation_report_full.json
```

---

## Data Setup

Before running the pipeline, the two raw data files must be placed in `data/raw/`.

### 1. Kaggle Dataset

Download the [Airbnb Price Dataset](https://www.kaggle.com/datasets/rupindersinghrana/airbnb-price-dataset) from Kaggle and place it in `data/raw/`

### 2. Scraped Listings

Either download the [pre-collected scraped listings](https://drive.google.com/file/d/1kh0C16r-HJFlctZ6lZYyzaefNgfVj7k1/view?usp=sharing) and place the file in `data/raw/` also

Or run the scraper notebook yourself to collect your own listings:

```
scraper/scraper.ipynb
```

---

## Setup

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)

### Install dependencies

```bash
pip install poetry
poetry install
```

for MacOS it is better to install via `brew`

```bash
brew install poetry
```

## Running the Pipeline

Each stage can be run individually or the full pipeline can be executed with
a single command.

### Full pipeline

```bash
make all
# merge-> clean-> features-> train-> test
```

### Individual stages

| Command | Description |
|---|---|
| `make merge` | Merge Kaggle and scraped datasets |
| `make clean` | Run the data cleaning pipeline |
| `make features` | Run feature engineering and produce train/test splits |
| `make train` | Train all models with full hyperparameter grid search |
| `make train-smote-tomek` | Train with SMOTE-Tomek balancing |
| `make train-smote` | Train with Mild SMOTE balancing |
| `make train-borderline` | Train with Borderline SMOTE balancing |
| `make train-single` | Train all models with fixed hyperparameters (fast) |
| `make test` | Run the full test suite with coverage report |
| `make lint` | Run static analysis with `ruff` |
| `make format` | Auto-format all source files |
| `make mlflow-ui` | Launch the MLflow experiment tracking UI |
| `make eda` | Launch the Streamlit EDA dashboard |

---

## Testing

```bash
make test
```

Runs 123 unit and integration tests across four modules with line coverage
reporting. All tests are expected to pass.

---

## Experiment Tracking

MLflow is used to log all training runs, hyperparameters, and metrics.
After training, launch the tracking UI with:

```bash
make mlflow-ui
```

Then open [http://127.0.0.1:5000/#/experiments](http://127.0.0.1:5000/#/experiments) in your browser.

---

## CI Pipeline

A GitHub Actions workflow runs automatically on every push to `main` or
`develop` and on every pull request targeting `main`. The pipeline:

1. Sets up Python 3.12 and installs all dependencies via Poetry
2. Runs the full test suite (`make test`)
3. Verifies code formatting (`make format-check`)

> **Skipping CI:** To bypass the workflow for a commit (e.g. documentation
> or config-only changes), include `[skip ci]` anywhere in the commit
> message.

```bash
git commit -m "update README [skip ci]"
```


