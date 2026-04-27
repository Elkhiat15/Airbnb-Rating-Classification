import argparse
import sys
from pathlib import Path
import warnings
import joblib
from sklearn.preprocessing import FunctionTransformer
import mlflow
import logging

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

sys.path.insert(0, str(Path(__file__).parent.parent))

from modelling.train import load_splits, prepare_data, train_and_log
from modelling.config import (
    MODEL_CONFIGS,
    CATBOOST_AVAILABLE,
    TRAKING_URI,
    ENHANCED_EXP_RUN,
    MODELS_DIR,
)
from modelling.class_balancing import recommended_balancing, IMBLEARN_AVAILABLE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_balancing(
    data_dir: str = "data/processed/",
    models_to_train: list = None,
    balance_method: str = "none",
    cv_folds: int = 3,
):
    """Train models with optional class balancing."""

    logger.info(f"Balance method: {balance_method}")

    # Set MLflow tracking URI to use MLflow server
    mlflow.set_tracking_uri(TRAKING_URI)
    mlflow.set_experiment(ENHANCED_EXP_RUN)
    train_df, test_df = load_splits(data_dir)

    X_train, X_test, y_train, y_test, preprocessor, label_encoder = prepare_data(
        train_df, test_df
    )

    # Apply preprocessing to get numeric arrays for SMOTE
    # (SMOTE requires numeric features, not DataFrames)
    logger.info("\nApplying preprocessing for balancing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Apply class balancing if requested
    if balance_method != "none":
        if not IMBLEARN_AVAILABLE:
            logger.warning("⚠️  imbalanced-learn not installed, skipping balancing")
            logger.warning("   Install: poetry add imbalanced-learn")
        else:
            X_train_processed, y_train = recommended_balancing(
                X_train_processed, y_train, method=balance_method
            )

    configs = MODEL_CONFIGS

    if models_to_train is None:
        # Use all models by default
        models_to_train = list(configs.keys())
        logger.info(f"\nTraining all models: {models_to_train}")
    else:
        # Validate specified models
        invalid = [m for m in models_to_train if m not in configs]
        if invalid:
            logger.error(f"Invalid model names: {invalid}")
            logger.error(f"Available models: {list(configs.keys())}")
            sys.exit(1)

    configs = {k: v for k, v in configs.items() if k in models_to_train}

    logger.info(f"\nTraining {len(configs)} models with balancing={balance_method}")

    # Train models
    # We pass processed data directly, so we use a simple "passthrough" preprocessor
    passthrough_preprocessor = FunctionTransformer()

    trained_models = {}
    best_model = None
    best_model_name = None
    best_f1 = -1  # Track best F1

    for model_name, config in configs.items():
        try:
            trained_pipeline, test_f1 = train_and_log(
                model_name=f"{model_name}_balanced_{balance_method}",
                model=config["model"],
                param_grid=config["params"],
                X_train=X_train_processed,
                y_train=y_train,
                X_test=X_test_processed,
                y_test=y_test,
                preprocessor=passthrough_preprocessor,
                label_encoder=label_encoder,
                cv_folds=cv_folds,
            )

            trained_models[model_name] = trained_pipeline

            # Track best model
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_model = trained_pipeline
                best_model_name = model_name


        except Exception as e:
            logger.error(f" Error training {model_name}: {str(e)}")
            continue

    # Save best model
    if best_model is not None:
        models_path = Path(MODELS_DIR)
        best_model_path = models_path / "best_balanced_model.pkl"
        joblib.dump(best_model, best_model_path)
        logger.info(f"\nBest Balanced model: {best_model_name} (F1: {best_f1:.4f})")
        logger.info(f"Best Balanced model saved to: {best_model_path}")

    logger.info(f"Successfully trained {len(trained_models)} models")

    return trained_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced training with class balancing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with no balancing
  python modelling/train_enhanced.py
  
  # Train all models with SMOTE+Tomek
  python modelling/train_enhanced.py --balance smote_tomek
  
  # Train specific models with balancing
  python modelling/train_enhanced.py --balance mild_smote --models xgboost catboost
        """,
    )

    parser.add_argument(
        "--data-dir", default="data/processed/", help="Directory with train/test splits"
    )
    parser.add_argument(
        "--models", nargs="+", help="Models to train (default: all models)"
    )
    parser.add_argument(
        "--balance",
        choices=["none", "mild_smote", "smote_tomek", "borderline"],
        default="none",
        help="Class balancing method (default: none)",
    )
    parser.add_argument("--cv-folds", type=int, default=3, help="CV folds (default: 3)")

    args = parser.parse_args()

    # Check dependencies
    if args.balance != "none" and not IMBLEARN_AVAILABLE:
        print("\n   ERROR: imbalanced-learn not installed")
        print("Install it to use SMOTE balancing:")
        print("  poetry add imbalanced-learn")
        sys.exit(1)

    if "catboost" in (args.models or []) and not CATBOOST_AVAILABLE:
        print("\n   WARNING: CatBoost not installed")
        print("Install it for best performance:")
        print("  poetry add catboost")

    # Run training
    train_with_balancing(
        data_dir=args.data_dir,
        models_to_train=args.models,
        balance_method=args.balance,
        cv_folds=args.cv_folds,
    )
