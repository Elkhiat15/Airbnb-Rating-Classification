#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.preprocessing import FunctionTransformer
from modelling.train import load_splits, prepare_data, train_and_log
from modelling.config import SINGLE_CONFIGS, TRAKING_URI
from modelling.class_balancing import recommended_balancing, IMBLEARN_AVAILABLE
import mlflow
import joblib
import logging

import warnings
warnings.filterwarnings('ignore', message='Found unknown categories')
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_dashboard_model(model_pipeline, model_name: str) -> None:
    """Persist a trained pipeline for the dashboard to load locally."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    explicit_path = models_dir / f"{model_name}_dashboard.pkl"
    dashboard_path = models_dir / "best_model.pkl"

    joblib.dump(model_pipeline, explicit_path)
    joblib.dump(model_pipeline, dashboard_path)

    logger.info(f"Saved dashboard model to {explicit_path}")
    logger.info(f"Updated dashboard default model at {dashboard_path}")

def train_single_params(
    data_dir: str = "data/processed/",
    models_to_train: list = None,
    balance_method: str = 'none'
):
    """Train models with single fixed parameter sets (no grid search)."""

    logger.info("="*70)
    logger.info("SINGLE-PARAMETER TRAINING MODE")
    logger.info("Training with fixed parameters - NO hyperparameter search")
    logger.info("="*70)
    logger.info(f"Balance method: {balance_method}")
    
    # Set MLflow tracking URI to use MLflow server
    mlflow.set_tracking_uri(TRAKING_URI)
    
    mlflow.set_experiment("single-test")
    # Load data
    logger.info(f"\nLoading data from {data_dir}...")
    train_df, test_df = load_splits(data_dir)
    
    # Prepare data
    logger.info("\nPreparing data...")
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = \
        prepare_data(train_df, test_df)
    
    # Apply preprocessing for balancing if needed
    if balance_method != 'none':
        logger.info("\nApplying preprocessing for balancing...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        if not IMBLEARN_AVAILABLE:
            logger.warning("   imbalanced-learn not installed, skipping balancing")
            logger.warning("   Install: pip install imbalanced-learn")
            balance_method = 'none'
        else:
            X_train_processed, y_train = recommended_balancing(
                X_train_processed, 
                y_train, 
                method=balance_method
            )
            # Use passthrough preprocessor since data is already processed
            preprocessor = FunctionTransformer()
            X_train = X_train_processed
            X_test = X_test_processed
    
    # Determine which models to train
    if models_to_train is None:
        models_to_train = list(SINGLE_CONFIGS.keys())
    else:
        # Validate model names
        invalid = [m for m in models_to_train if m not in SINGLE_CONFIGS]
        if invalid:
            logger.error(f"Invalid model names: {invalid}")
            logger.error(f"Available models: {list(SINGLE_CONFIGS.keys())}")
            sys.exit(1)
    
    logger.info(f"\nTraining {len(models_to_train)} models: {models_to_train}")
    logger.info("="*70)
    
    # Train each model
    results = {}
    for model_name in models_to_train:
        config = SINGLE_CONFIGS[model_name]
        
        try:
            model_run_name = f"{model_name}_single" if balance_method == 'none' else f"{model_name}_single_balanced_{balance_method}"
            best_pipeline = train_and_log(
                model_name=model_run_name,
                model=config["model"],
                param_grid=config["params"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                preprocessor=preprocessor,
                label_encoder=label_encoder,
                use_grid_search=True,  # Will be fast since only 1 param set
                cv_folds=3
            )

            if model_name == "xgboost":
                save_dashboard_model(best_pipeline, model_name)

            results[model_name] = "SUCCESS"
            logger.info(f"{model_name} completed successfully")
            
        except Exception as e:
            logger.error(f"{model_name} failed: {str(e)}")
            results[model_name] = f"FAILED: {str(e)}"
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    
    successes = [m for m, r in results.items() if r == "SUCCESS"]
    failures = [m for m, r in results.items() if r != "SUCCESS"]
    
    logger.info(f"\nSuccessful: {len(successes)}/{len(models_to_train)}")
    for model in successes:
        logger.info(f"  - {model}")
    
    if failures:
        logger.info(f"\nFailed: {len(failures)}/{len(models_to_train)}")
        for model in failures:
            logger.info(f"  - {model}: {results[model]}")
    
    logger.info("\n" + "="*70)
    logger.info("SINGLE-PARAMETER TRAINING COMPLETE")
    logger.info("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models with single fixed parameter sets (no grid search)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to train (default: all available). "
             f"Options: {list(SINGLE_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/",
        help="Directory with train/test splits (default: data/processed/)"
    )
    parser.add_argument(
        "--balance",
        choices=['none', 'mild_smote', 'smote_tomek', 'borderline'],
        default='none',
        help="Class balancing method (default: none)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.balance != 'none' and not IMBLEARN_AVAILABLE:
        print("\n   ERROR: imbalanced-learn not installed")
        print("Install it to use SMOTE balancing:")
        print("  pip install imbalanced-learn")
        sys.exit(1)
    
    # Run training
    results = train_single_params(
        data_dir=args.data_dir,
        models_to_train=args.models,
        balance_method=args.balance
    )
    
    # Exit with error if any model failed
    if any(r != "SUCCESS" for r in results.values()):
        sys.exit(1)

# Usage:
#     python modelling/train_single.py
#     python modelling/train_single.py --models xgboost catboost knn