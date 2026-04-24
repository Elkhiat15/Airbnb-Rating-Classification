import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from modelling.config import (
    MODEL_CONFIGS, SINGLE_CONFIGS,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_FEATURES
)
from modelling.evaluate import (
    standard_metrics,
    business_metrics,
    error_analysis,
    print_evaluation_summary
)

from modelling.train import *


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def train_test_dfs():
    """Minimal train/test DataFrames matching project schema."""
    np.random.seed(42)

    def _make(n):
        cats = ['Medium Rating', 'High Rating', 'Very High Rating']
        return pd.DataFrame({
            'property_type': np.random.choice(['Apartment', 'House'], n),
            'room_type': np.random.choice(['Entire home/apt', 'Private room'], n),
            'city': np.random.choice(['NYC', 'LA'], n),
            'neighbourhood': np.random.choice(['Downtown', 'Uptown'], n),
            'accommodates': np.random.randint(1, 6, n),
            'bathrooms': np.random.choice([1.0, 2.0], n),
            'bedrooms': np.random.choice([1, 2, 3], n).astype(float),
            'host_response_rate': np.random.uniform(50, 100, n),
            'amenity_count': np.random.randint(1, 20, n),
            'price_per_bed': np.random.uniform(20, 200, n),
            'listing_density': np.random.randint(1, 50, n),
            'price_relative_to_room_type': np.random.uniform(-2, 2, n),
            'log_price': np.random.uniform(3, 6, n),
            'log_number_of_reviews': np.random.uniform(0, 5, n),
            'rating_category': np.random.choice(cats, n, p=[0.2, 0.45, 0.35]),
        })

    return _make(80), _make(20)


def test_model_configs_structure():
    """Test that MODEL_CONFIGS has correct structure."""
    assert len(MODEL_CONFIGS) >= 5, "Should have at least 5 models"
    
    for model_name, config in MODEL_CONFIGS.items():
        assert 'model' in config, f"{model_name} missing 'model' key"
        assert 'params' in config, f"{model_name} missing 'params' key"


def test_single_configs_structure():
    """SINGLE_CONFIGS must have same keys and single-value param lists."""
    for name in SINGLE_CONFIGS:
        assert name in MODEL_CONFIGS, f"{name} not in MODEL_CONFIGS"
    for name, cfg in SINGLE_CONFIGS.items():
        for param, vals in cfg['params'].items():
            assert len(vals) == 1, f"SINGLE_CONFIGS[{name}].{param} has {len(vals)} values"


def test_feature_lists():
    """Test that feature lists are properly defined."""
    assert len(CATEGORICAL_FEATURES) > 0, "Should have categorical features"
    assert len(NUMERIC_FEATURES) > 0, "Should have numeric features"
    
    # Check no overlap
    overlap = set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)
    assert len(overlap) == 0, f"Feature lists overlap: {overlap}"


def test_target_feature():
    """Target column must be defined."""
    assert 'rating_category' in TARGET_FEATURES


def test_standard_metrics():
    """Test standard metrics computation."""
    # Create simple test data
    y_true = ['High Rating', 'Medium Rating', 'Very High Rating', 'High Rating'] * 5
    y_pred = ['High Rating', 'High Rating', 'Very High Rating', 'Medium Rating'] * 5
    
    metrics = standard_metrics(y_true, y_pred)
    
    # Check all expected metrics are present
    expected_metrics = [
        'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
        'precision_macro', 'recall_macro'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert 0 <= metrics[metric] <= 1, f"{metric} out of bounds"


def test_business_metrics():
    """Test business metrics computation."""
    # Test case: predictions all correct
    y_true = ['Medium Rating', 'High Rating', 'Very High Rating'] * 3
    y_pred = ['Medium Rating', 'High Rating', 'Very High Rating'] * 3
    
    metrics = business_metrics(y_true, y_pred)
    
    # Check all expected business metrics
    expected_metrics = [
        'over_promise_rate', 'undersell_rate', 'high_quality_recall',
        'severe_misclassification_rate',
        'host_confidence_score'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing business metric: {metric}"
    
    # For perfect predictions
    assert metrics['over_promise_rate'] == 0.0, "Should have no over-promises"
    assert metrics['undersell_rate'] == 0.0, "Should have no undersells"
    assert metrics['high_quality_recall'] == 1.0, "Should identify all high quality"


def test_business_metrics_over_promise():
    """Test over-promise detection."""
    # All predictions too high
    y_true = ['Medium Rating'] * 10
    y_pred = ['Very High Rating'] * 10
    
    metrics = business_metrics(y_true, y_pred)
    
    assert metrics['over_promise_rate'] == 1.0, "Should detect 100% over-promise"
    assert metrics['severe_misclassification_rate'] == 1.0, "All are severe errors"


def test_business_metrics_undersell():
    """Test undersell detection."""
    # All predictions too low
    y_true = ['Very High Rating'] * 10
    y_pred = ['Medium Rating'] * 10
    
    metrics = business_metrics(y_true, y_pred)
    
    assert metrics['undersell_rate'] == 1.0, "Should detect 100% undersell"
    assert metrics['host_confidence_score'] == 0.0, "Host confidence should be 0"


def test_error_analysis():
    """Test error analysis function."""
    y_true = ['High Rating', 'Medium Rating', 'Very High Rating'] * 3
    y_pred = ['High Rating', 'High Rating', 'Very High Rating'] * 3
    
    analysis = error_analysis(y_true, y_pred)
    
    assert 'confusion_matrix' in analysis
    assert 'classification_report' in analysis
    assert 'total_errors' in analysis
    assert 'error_rate' in analysis
    
    # Check confusion matrix shape
    cm = analysis['confusion_matrix']
    assert cm.shape[0] == cm.shape[1], "Confusion matrix should be square"


def test_print_evaluation_summary():
    """Test that evaluation summary prints without errors."""
    # Combine standard and business metrics
    y_true = ['High Rating', 'Medium Rating', 'Very High Rating'] * 3
    y_pred = ['High Rating', 'High Rating', 'Very High Rating'] * 3
    
    std_metrics = standard_metrics(y_true, y_pred)
    bus_metrics = business_metrics(y_true, y_pred)
    
    all_metrics = {**std_metrics, **bus_metrics}
    
    # Should not raise any errors
    print_evaluation_summary(all_metrics, "Test Model")


def test_baseline_model_in_config():
    """Test that baseline model is properly configured."""
    assert 'baseline' in MODEL_CONFIGS, "Baseline model missing"
    
    baseline_config = MODEL_CONFIGS['baseline']
    model = baseline_config['model']
    
    assert isinstance(model, DummyClassifier), "Baseline should be DummyClassifier"
    assert model.strategy == 'most_frequent', "Baseline should use most_frequent strategy"


# ── Load & Prepare ───────────────────────────────────────

def test_load_splits(train_test_dfs, tmp_path):
    """load_splits should reload CSVs with identical shape."""
    train_df, test_df = train_test_dfs
    train_df.to_csv(tmp_path / "train.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)
    t, te = load_splits(str(tmp_path))
    assert t.shape == train_df.shape
    assert te.shape == test_df.shape


def test_load_splits_missing_dir(tmp_path):
    """load_splits raises on missing directory."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_splits(str(tmp_path / "no_such_dir"))


def test_prepare_data_returns_correct_types(train_test_dfs):
    """prepare_data returns arrays, ColumnTransformer, and LabelEncoder."""
    train_df, test_df = train_test_dfs
    X_train, X_test, y_train, y_test, preprocessor, le = prepare_data(train_df, test_df)
    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(le, LabelEncoder)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)


def test_prepare_data_target_excluded(train_test_dfs):
    """Target column must not appear in features."""
    train_df, test_df = train_test_dfs
    X_train, *_ = prepare_data(train_df, test_df)
    assert 'rating_category' not in X_train.columns


def test_prepare_data_label_encoding_roundtrip(train_test_dfs):
    """LabelEncoder must round-trip cleanly."""
    train_df, test_df = train_test_dfs
    _, _, y_train, _, _, le = prepare_data(train_df, test_df)
    decoded = le.inverse_transform(y_train)
    re_encoded = le.transform(decoded)
    np.testing.assert_array_equal(y_train, re_encoded)


# ── train_and_log (MLflow mocked) ────────────────────────
@patch("modelling.train.mlflow")
def test_train_and_log_baseline(mock_mlflow, train_test_dfs):
    """train_and_log should return a fitted Pipeline for baseline."""

    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    train_df, test_df = train_test_dfs
    X_train, X_test, y_train, y_test, preprocessor, le = prepare_data(train_df, test_df)

    pipeline = train_and_log(
        model_name="baseline_test",
        model=DummyClassifier(strategy="most_frequent"),
        param_grid={},
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        preprocessor=preprocessor,
        label_encoder=le,
    )
    assert isinstance(pipeline, Pipeline)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)


@patch("modelling.train.mlflow")
def test_train_and_log_with_grid_search(mock_mlflow, train_test_dfs):
    """train_and_log with a small param grid should fit and return pipeline."""

    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    train_df, test_df = train_test_dfs
    X_train, X_test, y_train, y_test, preprocessor, le = prepare_data(train_df, test_df)

    pipeline = train_and_log(
        model_name="logreg_test",
        model=LogisticRegression(max_iter=200, random_state=42),
        param_grid={"C": [0.1, 1.0]},
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        preprocessor=preprocessor,
        label_encoder=le,
        use_grid_search=True,
        cv_folds=2,
    )
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.predict(X_test)) == len(y_test)


@patch("modelling.train.mlflow")
def test_train_and_log_logs_metrics(mock_mlflow, train_test_dfs):
    """MLflow log_metric should be called with standard and business metrics."""

    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    train_df, test_df = train_test_dfs
    X_train, X_test, y_train, y_test, preprocessor, le = prepare_data(train_df, test_df)

    train_and_log(
        model_name="baseline_metrics_test",
        model=DummyClassifier(strategy="most_frequent"),
        param_grid={},
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        preprocessor=preprocessor,
        label_encoder=le,
    )

    logged_names = [call.args[0] for call in mock_mlflow.log_metric.call_args_list]
    assert any("accuracy" in n for n in logged_names)
    assert any("f1_macro" in n for n in logged_names)
    assert any("over_promise_rate" in n for n in logged_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
