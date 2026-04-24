import pandas as pd
import numpy as np
import pytest

from feature_engineering.transformations import encode_categoricals, scale_numerics, log_transform
from feature_engineering.engineering import *
from feature_engineering.selection import correlation_filter, mutual_information_ranking

class TestTransformations:
    """Tests for encoding and scaling."""

    def test_encode_categoricals_creates_dummies(self):
        """Test that one-hot encoding creates expected dummy columns."""
        df = pd.DataFrame({
            "city": ["New York", "Los Angeles", "New York"],
            "room_type": ["Entire home/apt", "Private room", "Entire home/apt"],
            "price": [100, 200, 150]
        })
        result = encode_categoricals(df, columns=["city", "room_type"])
        
        # Original columns should be removed
        assert "city" not in result.columns
        assert "room_type" not in result.columns
        
        # Dummy columns should be created
        assert "city_New York" in result.columns
        assert "city_Los Angeles" in result.columns
        assert "room_type_Entire home/apt" in result.columns
        assert "room_type_Private room" in result.columns
        
        # Price should remain unchanged
        assert "price" in result.columns
        assert list(result["price"]) == [100, 200, 150]

    def test_encode_categoricals_drop_first(self):
        """Test that drop_first parameter works correctly."""
        df = pd.DataFrame({
            "city": ["New York", "Los Angeles", "New York"],
            "price": [100, 200, 150]
        })
        result = encode_categoricals(df, columns=["city"], drop_first=True)
        
        # Should only have n-1 dummy columns
        city_cols = [col for col in result.columns if col.startswith("city_")]
        assert len(city_cols) == 1  # 2 cities - 1 = 1 column

    def test_scale_numerics_standard_scaler(self):
        """Test StandardScaler on numeric columns."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "accommodates": [2, 4, 6],
            "city": ["NYC", "LA", "SF"]
        })
        result = scale_numerics(df, columns=["price", "accommodates"], method="standard")
        
        # Check that scaling happened (mean ~0, std close to 1 using ddof=0)
        assert abs(result["price"].mean()) < 0.01
        assert abs(result["price"].std(ddof=0) - 1.0) < 0.01
        
        # Non-numeric column should remain
        assert list(result["city"]) == ["NYC", "LA", "SF"]

    def test_scale_numerics_minmax_scaler(self):
        """Test MinMaxScaler on numeric columns."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "accommodates": [2, 4, 6]
        })
        result = scale_numerics(df, columns=["price"], method="minmax")
        
        # MinMax should scale to [0, 1]
        assert result["price"].min() == 0.0
        assert result["price"].max() == 1.0

    def test_scale_numerics_invalid_method(self):
        """Test that invalid scaling method raises error."""
        df = pd.DataFrame({"price": [100, 200, 300]})
        with pytest.raises(ValueError, match="Unknown scaling method"):
            scale_numerics(df, columns=["price"], method="invalid")

    def test_log_transform_applies_correctly(self):
        """Test that log transformation works correctly."""
        df = pd.DataFrame({
            "price": [100, 1000, 10000],
            "number_of_reviews": [1, 10, 100]
        })
        result = log_transform(df, columns=["price", "number_of_reviews"])
        
        # Check log transformation (log1p)
        expected_price = np.log1p([100, 1000, 10000])
        np.testing.assert_array_almost_equal(result["price"], expected_price)

    def test_log_transform_handles_zeros(self):
        """Test that log1p handles zero values correctly."""
        df = pd.DataFrame({"price": [0, 1, 10, 100]})
        result = log_transform(df, columns=["price"])
        
        # log1p(0) should be 0
        assert result["price"].iloc[0] == 0.0
        assert result["price"].iloc[1] == np.log1p(1)

    def test_transformations_do_not_modify_original(self):
        """Test that transformations don't modify the original DataFrame."""
        df = pd. DataFrame({
            "city": ["New York", "Los Angeles"],
            "price": [100, 200]
        })
        original_df = df.copy()
        
        _ = encode_categoricals(df, columns=["city"])
        pd.testing.assert_frame_equal(df, original_df)
        
        _ = scale_numerics(df, columns=["price"])
        pd.testing.assert_frame_equal(df, original_df)
        
        _ = log_transform(df, columns=["price"])
        pd.testing.assert_frame_equal(df, original_df)


class TestEngineering:
    """Tests for derived features."""

    def test_amenity_count_is_non_negative(self):
        """Test that amenity_count is always non-negative."""
        df = pd.DataFrame({
            "amenities": [
                '{"WiFi","Kitchen","TV"}',
                '{"WiFi"}',
                '{}',
                None
            ]
        })
        result = add_amenity_count(df)
        
        assert "amenity_count" in result.columns
        assert all(result["amenity_count"] >= 0)

    def test_amenity_count_parses_correctly(self):
        """Test that amenity count parses amenities string correctly."""
        df = pd.DataFrame({
            "amenities": [
                '{"WiFi","Kitchen","TV"}',
                '{"WiFi"}',
                '{}',
                ''
            ]
        })
        result = add_amenity_count(df)
        
        assert result["amenity_count"].iloc[0] == 3
        assert result["amenity_count"].iloc[1] == 1
        assert result["amenity_count"].iloc[2] == 0
        assert result["amenity_count"].iloc[3] == 0

    def test_price_per_bed_handles_zero_beds(self):
        """Test that price_per_bed handles zero beds gracefully."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "beds": [2, 0, 4]  # One zero bed
        })
        result = add_price_ratios(df)
        
        assert "price_per_bed" in result.columns
        # When beds=0, should return price itself
        assert result["price_per_bed"].iloc[0] == 50.0  # 100/2
        assert result["price_per_bed"].iloc[1] == 200.0  # 200 (no division)
        assert result["price_per_bed"].iloc[2] == 75.0  # 300/4

    def test_price_per_guest_handles_zero_accommodates(self):
        """Test that price_per_guest handles zero accommodates gracefully."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "accommodates": [2, 0, 5]
        })
        result = add_price_ratios(df)
        
        assert "price_per_guest" in result.columns
        assert result["price_per_guest"].iloc[0] == 50.0
        assert result["price_per_guest"].iloc[1] == 200.0
        assert result["price_per_guest"].iloc[2] == 60.0

    def test_price_ratios_creates_both_features(self):
        """Test that add_price_ratios creates both ratio features."""
        df = pd.DataFrame({
            "price": [100, 200],
            "beds": [2, 4],
            "accommodates": [4, 8]
        })
        result = add_price_ratios(df)
        
        assert "price_per_bed" in result.columns
        assert "price_per_guest" in result.columns

    def test_is_entire_home_flag_created(self):
        """Test that is_entire_home binary flag is created correctly."""
        df = pd.DataFrame({
            "room_type": ["Entire home/apt", "Private room", "Entire home/apt", "Shared room"]
        })
        result = add_categorical_flags(df)
        
        assert "is_entire_home" in result.columns
        assert list(result["is_entire_home"]) == [1, 0, 1, 0]

    def test_listing_density_counts_correctly(self):
        """Test that listing_density counts listings per neighbourhood."""
        df = pd.DataFrame({
            "neighbourhood": ["Brooklyn", "Manhattan", "Brooklyn", "Brooklyn", "Manhattan"]
        })
        result = add_listing_density(df)
        
        assert "listing_density" in result.columns
        assert result["listing_density"].iloc[0] == 3  # Brooklyn appears 3 times
        assert result["listing_density"].iloc[1] == 2  # Manhattan appears 2 times

    def test_engineering_does_not_modify_original(self):
        """Test that feature engineering doesn't modify original DataFrame."""
        df = pd.DataFrame({
            "amenities": ['{"WiFi","Kitchen"}'],
            "price": [100],
            "beds": [2],
            "accommodates": [4]
        })
        original_df = df.copy()
        
        _ = add_amenity_count(df)
        pd.testing.assert_frame_equal(df, original_df)
        
        _ = add_price_ratios(df)
        pd.testing.assert_frame_equal(df, original_df)


class TestSelection:
    """Tests for feature selection."""

    def test_correlation_filter_removes_high_correlation(self):
        """Test that correlation filter removes highly correlated features."""
        # Create perfectly correlated features
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [1, 2, 3, 4, 5],  # Perfect correlation with feature1
            "feature3": [5, 4, 3, 2, 1]   # Perfect negative correlation with feature1
        })
        
        result = correlation_filter(df, threshold=0.95)
        
        # Should keep only one of the highly correlated features
        assert len(result) < len(df.columns)
        assert isinstance(result, list)

    def test_correlation_filter_keeps_low_correlation(self):
        """Test that correlation filter keeps features with low correlation."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 3, 1, 4, 2],  # Low correlation
            "feature3": [2, 4, 1, 5, 3]   # Low correlation
        })
        
        result = correlation_filter(df, threshold=0.95)
        
        # All features should be kept
        assert len(result) == len(df.columns)

    def test_mutual_information_ranking_returns_top_k(self):
        """Test that MI ranking returns exactly top_k features."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.randint(0, 3, 100))
        
        result = mutual_information_ranking(X, y, top_k=5)
        
        assert len(result) == 5
        assert isinstance(result, list)
        assert all(f in X.columns for f in result)

class TestFeaturePipeline:
    """Integration tests for the full feature engineering pipeline."""

    def test_full_pipeline_runs_without_error(self):
        """Test that the full feature pipeline runs end-to-end."""
        df = pd.DataFrame({
            "amenities": ['{"WiFi","Kitchen"}', '{"WiFi"}'],
            "price": [100, 200],
            "beds": [2, 4],
            "accommodates": [4, 8],
            "room_type": ["Entire home/apt", "Private room"],
            "neighbourhood": ["Brooklyn", "Manhattan"],
            "review_scores_rating": [4.5, 4.0]
        })
        
        
        df_engineered = engineer_features(df)
        
        # Check new features were added
        assert "amenity_count" in df_engineered.columns
        assert "price_per_bed" in df_engineered.columns
        assert "price_per_guest" in df_engineered.columns
        assert "is_entire_home" in df_engineered.columns
        assert "listing_density" in df_engineered.columns

    def test_pipeline_preserves_target(self):
        """Test that feature engineering preserves the target variable."""
        df = pd.DataFrame({
            "amenities": ['{"WiFi"}'],
            "price": [100],
            "beds": [2],
            "accommodates": [4],
            "room_type": ["Entire home/apt"],
            "neighbourhood": ["Brooklyn"],
            "review_scores_rating": [4.5]
        })
        
        result = engineer_features(df)
        
        assert "review_scores_rating" in result.columns
        assert result["review_scores_rating"].iloc[0] == 4.5
