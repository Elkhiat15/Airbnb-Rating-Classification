.PHONY: all data scrape merge validate clean features train train-smote-tomek train-smote train-borderline train-single train-single-smote-tomek train-single-smote train-single-borderline compare-models mlflow-ui test lint check format eda install-ml

#  Default 
all: data validate clean features train test

#  Data Acquisition & Merge 
data: merge

merge:
	poetry run python3 scraper/merge.py

#  Validation 
# validate:
# 	python3 validation/validation.py --data data/merged/merged_airbnb_data.csv

#  Cleaning 
clean:
	poetry run python3 cleaning/cleaning.py

#  Feature Engineering 
features:
	poetry run python3 feature_engineering/pipeline.py

#  Model Training 
# Train all models (full hyperparameter grids)
train:
	poetry run python3 modelling/train.py

train-smote-tomek:
	poetry run python3 modelling/train_enhanced.py --balance smote_tomek

train-smote:
	poetry run python3 modelling/train_enhanced.py --balance mild_smote

train-borderline:
	poetry run python3 modelling/train_enhanced.py --balance borderline

# Train all models (single parameter sets - fastest)
train-single:
	poetry run python3 modelling/train_single.py

train-single-smote-tomek:
	poetry run python3 modelling/train_single.py --balance smote_tomek

train-single-smote:
	poetry run python3 modelling/train_single.py --balance mild_smote

train-single-borderline:
	poetry run python3 modelling/train_single.py --balance borderline

compare-models:
	poetry run python3 modelling/compare_models.py

mlflow-ui:
	poetry run mlflow ui


#  EDA Dashboard 
eda:
	poetry run python3 -m streamlit run eda/dashboard.py

#  Testing 
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

#  Linting 
lint:
	poetry run ruff check .

format-check:
	poetry run ruff format --check .

format:
	poetry run ruff format .