import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from collections import Counter

from modelling.evaluate import standard_metrics, business_metrics, error_analysis, print_evaluation_summary
from modelling.baseline import train_baseline
from modelling.class_balancing import compute_sample_weights, get_balanced_strategy, IMBLEARN_AVAILABLE
from modelling.config import MODEL_CONFIGS, SINGLE_CONFIGS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_FEATURES

from modelling.class_balancing import recommended_balancing

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from modelling.train import *

# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def binary_labels():
    """Simple binary labels for quick metric checks."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1])
    return y_true, y_pred


@pytest.fixture
def rating_labels():
    """String labels matching the project's 3-class target."""
    y_true = ['Medium Rating', 'High Rating', 'Very High Rating',
              'High Rating', 'Medium Rating', 'Very High Rating',
              'High Rating', 'High Rating', 'Very High Rating']
    y_pred = ['Medium Rating', 'High Rating', 'High Rating',
              'High Rating', 'Medium Rating', 'Very High Rating',
              'Medium Rating', 'High Rating', 'Very High Rating']
    return y_true, y_pred


@pytest.fixture
def imbalanced_labels():
    """Numeric labels with class imbalance matching real distribution."""
    np.random.seed(42)
    y = np.array([0] * 455 + [1] * 208 + [2] * 337)
    return y


@pytest.fixture
def synthetic_features(imbalanced_labels):
    """Synthetic feature matrix matching imbalanced labels."""
    np.random.seed(42)
    X = np.random.randn(len(imbalanced_labels), 10)
    return X


@pytest.fixture
def train_test_dfs():
    """Minimal train/test DataFrames with required columns."""
    np.random.seed(42)
    n_train, n_test = 80, 20

    def _make_df(n):
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
            'rating_category': np.random.choice(
                ['Medium Rating', 'High Rating', 'Very High Rating'], n,
                p=[0.2, 0.45, 0.35]
            ),
        })

    return _make_df(n_train), _make_df(n_test)


# ── StandardMetrics ──────────────────────────────────────

class TestStandardMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1])
        metrics = standard_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_returns_all_keys(self, binary_labels):
        y_true, y_pred = binary_labels
        metrics = standard_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

    def test_metrics_in_valid_range(self, rating_labels):
        y_true, y_pred = rating_labels
        metrics = standard_metrics(y_true, y_pred)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"

    def test_per_class_metrics_present(self, rating_labels):
        y_true, y_pred = rating_labels
        metrics = standard_metrics(y_true, y_pred)
        for cls in ['High Rating', 'Medium Rating', 'Very High Rating']:
            assert f"f1_{cls}" in metrics
            assert f"precision_{cls}" in metrics
            assert f"recall_{cls}" in metrics

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        metrics = standard_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_balanced_accuracy_differs_from_accuracy(self):
        """Balanced accuracy penalises majority-class bias."""
        y_true = ['High Rating'] * 8 + ['Medium Rating'] * 2
        y_pred = ['High Rating'] * 10  # always majority
        metrics = standard_metrics(y_true, y_pred)
        assert metrics["accuracy"] > metrics["balanced_accuracy"]


# ── BusinessMetrics ──────────────────────────────────────

class TestBusinessMetrics:
    def test_perfect_predictions(self):
        y = ['Medium Rating', 'High Rating', 'Very High Rating'] * 5
        metrics = business_metrics(y, y)
        assert metrics['over_promise_rate'] == 0.0
        assert metrics['undersell_rate'] == 0.0
        assert metrics['high_quality_recall'] == 1.0
        assert metrics['host_confidence_score'] == 1.0
        assert metrics['severe_misclassification_rate'] == 0.0

    def test_all_over_promised(self):
        y_true = ['Medium Rating'] * 10
        y_pred = ['Very High Rating'] * 10
        metrics = business_metrics(y_true, y_pred)
        assert metrics['over_promise_rate'] == 1.0
        assert metrics['severe_misclassification_rate'] == 1.0

    def test_all_undersold(self):
        y_true = ['Very High Rating'] * 10
        y_pred = ['Medium Rating'] * 10
        metrics = business_metrics(y_true, y_pred)
        assert metrics['undersell_rate'] == 1.0
        assert metrics['host_confidence_score'] == 0.0

    def test_mixed_errors(self, rating_labels):
        y_true, y_pred = rating_labels
        metrics = business_metrics(y_true, y_pred)
        # Sum of over_promise + undersell + correct should equal 1
        correct_rate = 1 - metrics['over_promise_rate'] - metrics['undersell_rate']
        assert abs(correct_rate - np.mean(np.array(y_true) == np.array(y_pred))) < 1e-9

    def test_no_very_high_in_true(self):
        """high_quality_recall should be 0 when no Very High in ground truth."""
        y_true = ['Medium Rating', 'High Rating'] * 5
        y_pred = ['Medium Rating', 'High Rating'] * 5
        metrics = business_metrics(y_true, y_pred)
        assert metrics['high_quality_recall'] == 0.0


    def test_returns_all_keys(self, rating_labels):
        y_true, y_pred = rating_labels
        metrics = business_metrics(y_true, y_pred)
        expected = [
            'over_promise_rate', 'undersell_rate', 'high_quality_recall',
            'severe_misclassification_rate',
            'host_confidence_score',
        ]
        for key in expected:
            assert key in metrics


# ── ErrorAnalysis ────────────────────────────────────────

class TestErrorAnalysis:
    def test_returns_required_keys(self, rating_labels):
        y_true, y_pred = rating_labels
        result = error_analysis(y_true, y_pred)
        assert 'confusion_matrix' in result
        assert 'classification_report' in result
        assert 'total_errors' in result
        assert 'error_rate' in result

    def test_perfect_predictions(self):
        y = ['A', 'B', 'C'] * 5
        result = error_analysis(y, y)
        assert result['total_errors'] == 0
        assert result['error_rate'] == 0.0

    def test_confusion_matrix_square(self, rating_labels):
        y_true, y_pred = rating_labels
        cm = error_analysis(y_true, y_pred)['confusion_matrix']
        assert cm.shape[0] == cm.shape[1]

    def test_error_rate_consistency(self, rating_labels):
        y_true, y_pred = rating_labels
        result = error_analysis(y_true, y_pred)
        expected_rate = result['total_errors'] / len(y_true)
        assert abs(result['error_rate'] - expected_rate) < 1e-9

    def test_with_feature_data(self):
        """error_patterns populated when X_test is provided."""
        y_true = ['A', 'B', 'A', 'B']
        y_pred = ['A', 'A', 'A', 'B']
        X = np.random.randn(4, 3)
        result = error_analysis(y_true, y_pred, X_test=X, feature_names=['f1', 'f2', 'f3'])
        assert 'error_patterns' in result
        assert result['error_patterns']['total_errors'] == 1


# ── PrintEvaluationSummary ───────────────────────────────

class TestPrintSummary:
    def test_runs_without_error(self, rating_labels, capsys):
        y_true, y_pred = rating_labels
        metrics = {**standard_metrics(y_true, y_pred), **business_metrics(y_true, y_pred)}
        print_evaluation_summary(metrics, "TestModel")
        captured = capsys.readouterr()
        assert "TestModel" in captured.out
        assert "STANDARD METRICS" in captured.out
        assert "BUSINESS METRICS" in captured.out


# ── Baseline ─────────────────────────────────────────────

class TestBaseline:
    def test_train_baseline_returns_model_and_preds(self):
        X_train = np.random.randn(50, 5)
        y_train = np.array([0] * 25 + [1] * 25)
        X_test = np.random.randn(10, 5)
        y_test = np.array([0] * 5 + [1] * 5)
        model, preds = train_baseline(X_train, y_train, X_test, y_test)
        assert len(preds) == len(y_test)
        # Most-frequent baseline predicts only one class
        assert len(set(preds)) == 1

    def test_baseline_config_is_dummy(self):
        assert isinstance(MODEL_CONFIGS['baseline']['model'], DummyClassifier)


# ── ComputeSampleWeights ─────────────────────────────────

class TestComputeSampleWeights:
    def test_balanced_weights_length(self, imbalanced_labels):
        weights = compute_sample_weights(imbalanced_labels, strategy='balanced')
        assert len(weights) == len(imbalanced_labels)

    def test_balanced_minority_gets_higher_weight(self, imbalanced_labels):
        weights = compute_sample_weights(imbalanced_labels, strategy='balanced')
        # Class 1 (208 samples) is minority → should have highest weight
        minority_weight = weights[imbalanced_labels == 1][0]
        majority_weight = weights[imbalanced_labels == 0][0]
        assert minority_weight > majority_weight

    def test_sqrt_strategy(self, imbalanced_labels):
        weights = compute_sample_weights(imbalanced_labels, strategy='sqrt')
        assert len(weights) == len(imbalanced_labels)
        assert all(w > 0 for w in weights)

    def test_custom_dict_strategy(self):
        y = np.array([0, 0, 1, 1, 2])
        weights = compute_sample_weights(y, strategy={0: 1.0, 1: 2.0, 2: 3.0})
        assert weights[0] == 1.0
        assert weights[2] == 2.0
        assert weights[4] == 3.0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_sample_weights(np.array([0, 1]), strategy='bad')


# ── GetBalancedStrategy ──────────────────────────────────

class TestGetBalancedStrategy:
    def test_returns_dict(self, imbalanced_labels):
        strategy = get_balanced_strategy(imbalanced_labels, target_ratio=0.7)
        assert isinstance(strategy, dict)

    def test_boosts_minority_only(self, imbalanced_labels):
        strategy = get_balanced_strategy(imbalanced_labels, target_ratio=0.6)
        # Majority class (0, 455 samples) should NOT appear in strategy
        majority_count = Counter(imbalanced_labels).most_common(1)[0][1]
        target = int(majority_count * 0.6)
        for cls, count in strategy.items():
            assert count >= target

    def test_full_balance(self, imbalanced_labels):
        strategy = get_balanced_strategy(imbalanced_labels, target_ratio=1.0)
        majority_count = Counter(imbalanced_labels).most_common(1)[0][1]
        for cls, count in strategy.items():
            assert count >= majority_count


# ── RecommendedBalancing ─────────────────────────────────

class TestRecommendedBalancingNone:
    def test_none_returns_original(self, synthetic_features, imbalanced_labels):
        X_out, y_out = recommended_balancing(
            synthetic_features, imbalanced_labels, method='none'
        )
        np.testing.assert_array_equal(X_out, synthetic_features)
        np.testing.assert_array_equal(y_out, imbalanced_labels)


@pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
class TestRecommendedBalancingSmote:
    def test_smote_tomek_resamples(self, synthetic_features, imbalanced_labels):
        X_out, y_out = recommended_balancing(
            synthetic_features, imbalanced_labels, method='smote_tomek'
        )
        # SMOTE-Tomek is hybrid: SMOTE oversamples then Tomek removes boundary
        # samples, so output may be smaller than input. Just verify it ran.
        assert len(y_out) > 0
        assert X_out.shape[0] == len(y_out)

    def test_borderline_increases_samples(self, synthetic_features, imbalanced_labels):
        X_out, y_out = recommended_balancing(
            synthetic_features, imbalanced_labels, method='borderline'
        )
        assert len(y_out) >= len(imbalanced_labels)

    def test_mild_smote_increases_samples(self, synthetic_features, imbalanced_labels):
        X_out, y_out = recommended_balancing(
            synthetic_features, imbalanced_labels, method='mild_smote'
        )
        assert len(y_out) >= len(imbalanced_labels)


# ── Config ───────────────────────────────────────────────

class TestConfig:
    def test_model_configs_minimum_count(self):
        assert len(MODEL_CONFIGS) >= 5

    def test_single_configs_match_model_configs(self):
        for name in SINGLE_CONFIGS:
            assert name in MODEL_CONFIGS, f"{name} in SINGLE but not MODEL"

    def test_each_config_has_required_keys(self):
        for name, cfg in MODEL_CONFIGS.items():
            assert 'model' in cfg, f"{name} missing 'model'"
            assert 'params' in cfg, f"{name} missing 'params'"

    def test_single_configs_have_single_values(self):
        """Single configs should have exactly 1 value per param (fast training)."""
        for name, cfg in SINGLE_CONFIGS.items():
            for param, values in cfg['params'].items():
                assert len(values) == 1, f"{name}.{param} has {len(values)} values, expected 1"

    def test_no_feature_overlap(self):
        overlap = set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)
        assert len(overlap) == 0, f"Feature overlap: {overlap}"

    def test_target_feature_defined(self):
        assert 'rating_category' in TARGET_FEATURES


# ── PrepareData ──────────────────────────────────────────

class TestPrepareData:
    def test_prepare_data_shapes(self, train_test_dfs):
        
        train_df, test_df = train_test_dfs
        X_train, X_test, y_train, y_test, preprocessor, le = prepare_data(train_df, test_df)
        assert X_train.shape[0] == len(train_df)
        assert X_test.shape[0] == len(test_df)
        assert len(y_train) == len(train_df)
        assert len(y_test) == len(test_df)

    def test_label_encoder_classes(self, train_test_dfs):
        from modelling.train import prepare_data
        train_df, test_df = train_test_dfs
        _, _, _, _, _, le = prepare_data(train_df, test_df)
        expected = {'High Rating', 'Medium Rating', 'Very High Rating'}
        assert set(le.classes_) == expected

    def test_preprocessor_is_column_transformer(self, train_test_dfs):
        train_df, test_df = train_test_dfs
        _, _, _, _, preprocessor, _ = prepare_data(train_df, test_df)
        assert isinstance(preprocessor, ColumnTransformer)


# ── LoadSplits ───────────────────────────────────────────

class TestLoadSplits:
    def test_load_splits_from_csvs(self, train_test_dfs, tmp_path):
        train_df, test_df = train_test_dfs
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)
        loaded_train, loaded_test = load_splits(str(tmp_path))
        assert loaded_train.shape == train_df.shape
        assert loaded_test.shape == test_df.shape

    def test_load_splits_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_splits(str(tmp_path / "nonexistent"))


# ── Integration ──────────────────────────────────────────

class TestIntegration:
    def test_prepare_then_baseline(self, train_test_dfs):
        """End-to-end: prepare data → train baseline → get predictions."""
        train_df, test_df = train_test_dfs
        X_train, X_test, y_train, y_test, preprocessor, le = prepare_data(train_df, test_df)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        model, preds = train_baseline(X_train_t, y_train, X_test_t, y_test)
        assert len(preds) == len(y_test)

    def test_metrics_pipeline(self, rating_labels):
        """standard_metrics → business_metrics → print_evaluation_summary."""
        y_true, y_pred = rating_labels
        std = standard_metrics(y_true, y_pred)
        bus = business_metrics(y_true, y_pred)
        all_metrics = {**std, **bus}
        # Should not raise
        print_evaluation_summary(all_metrics, "IntegrationTest")
        err = error_analysis(y_true, y_pred)
        assert err['total_errors'] + np.trace(err['confusion_matrix']) == len(y_true)
