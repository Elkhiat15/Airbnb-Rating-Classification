.PHONY: all data scrape merge validate clean features train train-smote-tomek train-smote train-borderline train-single train-single-smote-tomek train-single-smote train-single-borderline compare-models mlflow-ui test lint eda install-ml

#  Default 
all: data validate clean features train test

#  Data Acquisition & Merge 
data: scrape merge

scrape:
	python3 scraper/scraper.py

merge:
	python3 scraper/merge.py

#  Validation 
validate:
	python3 validation/validation.py --data data/merged/merged_airbnb_data.csv

#  Cleaning 
clean:
	python3 -m poetry run python3 cleaning/cleaning.py

#  Feature Engineering 
features:
	python3 -m poetry run python3 feature_engineering/pipeline.py

#  Model Training 
# Train all models (full hyperparameter grids)
train:
	python3 -m poetry run python3 modelling/train.py

train-smote-tomek:
	python3 -m poetry run python3 modelling/train_enhanced.py --balance smote_tomek

train-smote:
	python3 -m poetry run python3 modelling/train_enhanced.py --balance mild_smote

train-borderline:
	python3 -m poetry run python3 modelling/train_enhanced.py --balance borderline

# Train all models (single parameter sets - fastest)
train-single:
	python3 -m poetry run python3 modelling/train_single.py

train-single-smote-tomek:
	python3 -m poetry run python3 modelling/train_single.py --balance smote_tomek

train-single-smote:
	python3 -m poetry run python3 modelling/train_single.py --balance mild_smote

train-single-borderline:
	python3 -m poetry run python3 modelling/train_single.py --balance borderline

compare-models:
	python3 -m poetry run python3 modelling/compare_models.py

mlflow-ui:
	python3 -m poetry run mlflow ui


#  EDA Dashboard 
eda:
	python3 -m poetry run python3 -m streamlit run eda/dashboard.py

#  Testing 
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

#  Linting 
lint:
	ruff check .
	ruff format --check .
