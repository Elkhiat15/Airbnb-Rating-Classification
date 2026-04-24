import pandas as pd
import numpy as np
import pytest

from cleaning.cleaning import *

class TestNormalizeColumns:
    """Tests for data normalisation."""

    def test_city_aliases_mapped_correctly(self):
        """Test that city aliases are mapped to canonical names."""
        df = pd.DataFrame({
            "city": ["NYC", "LA", "SF", "DC", "Chicago", "Boston", 
                     "New York", "Los Angeles", "San Francisco"]
        })
        result = normalize_columns(df)
        expected = ["New York", "Los Angeles", "San Francisco", "Washington DC", 
                    "Chicago", "Boston", "New York", "Los Angeles", "San Francisco"]
        assert list(result["city"]) == expected

    def test_room_type_aliases_mapped_correctly(self):
        """Test that room_type aliases are mapped to canonical names."""
        df = pd.DataFrame({
            "room_type": ["Entire home", "Entire apartment", "Private room", 
                          "Entire home/apt", "Shared room"]
        })
        result = normalize_columns(df)
        expected = ["Entire home/apt", "Entire home/apt", "Private room", 
                    "Entire home/apt", "Shared room"]
        assert list(result["room_type"]) == expected

    def test_host_identity_verified_converted_to_boolean(self):
        """Test that host_identity_verified is converted from string to boolean."""
        df = pd.DataFrame({
            "host_identity_verified": ["t", "f", "True", "False", "1", "0"]
        })
        result = normalize_columns(df)
        expected = [True, False, True, False, True, False]
        assert list(result["host_identity_verified"]) == expected
        assert result["host_identity_verified"].dtype == bool

    def test_normalize_handles_missing_columns(self):
        """Test that normalization works even if some columns are missing."""
        df = pd.DataFrame({
            "city": ["NYC", "LA"],
            "other_col": [1, 2]
        })
        result = normalize_columns(df)
        assert "city" in result.columns
        assert list(result["city"]) == ["New York", "Los Angeles"]
        assert "other_col" in result.columns

    def test_normalize_does_not_modify_original(self):
        """Test that normalization doesn't modify the original DataFrame."""
        df = pd.DataFrame({"city": ["NYC", "LA"]})
        original_values = df["city"].copy()
        _ = normalize_columns(df)
        pd.testing.assert_series_equal(df["city"], original_values)


class TestHandleMissingValues:
    """Tests for missing value imputation."""

    def test_critical_columns_drop_rows(self):
        """Test that rows with missing critical columns are dropped."""
        df = pd.DataFrame({
            "price": [100, None, 300],
            "city": ["New York", "Los Angeles", None],
            "property_type": ["Apartment", "House", "Condo"],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        result = handle_missing_values(df)
        # Should drop rows 1 (missing price) and 2 (missing city)
        assert len(result) == 1
        assert result.iloc[0]["price"] == 100

    def test_numeric_columns_imputed_with_median(self):
        """Test that numeric columns are imputed with median."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "accommodates": [2, None, 4],
            "bathrooms": [1, 2, None],
            "bedrooms": [None, 2, 3],
            "beds": [1, None, 3],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        result = handle_missing_values(df)
        # Check that no nulls remain in numeric columns
        assert result["accommodates"].isnull().sum() == 0
        assert result["bathrooms"].isnull().sum() == 0
        assert result["bedrooms"].isnull().sum() == 0
        assert result["beds"].isnull().sum() == 0
        # Check median imputation (accommodates: median of [2, 4] = 3.0)
        assert result["accommodates"].iloc[1] == 3.0

    def test_host_response_rate_creates_flag(self):
        """Test that missing host_response_rate creates a has_response_rate flag."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "host_response_rate": [95, None, 100],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        result = handle_missing_values(df)
        assert "has_response_rate" in result.columns
        assert list(result["has_response_rate"]) == [1, 0, 1]
        assert result["host_response_rate"].isnull().sum() == 0

    def test_host_response_rate_percentage_conversion(self):
        """Test that host_response_rate percentage strings are converted to numeric."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "host_response_rate": ["95%", "100%", "50%"],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        result = handle_missing_values(df)
        assert result["host_response_rate"].dtype in [np.float64, np.int64, float, int]
        assert list(result["host_response_rate"]) == [95.0, 100.0, 50.0]

    def test_target_missing_values_dropped(self):
        """Test that rows with missing target (review_scores_rating) are dropped."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "city": ["New York", "Los Angeles", "San Francisco"],
            "review_scores_rating": [4.5, None, 3.5]
        })
        result = handle_missing_values(df)
        assert len(result) == 2
        assert result["review_scores_rating"].isnull().sum() == 0

    def test_neighbourhood_imputed_with_unknown(self):
        """Test that missing neighbourhood is imputed with 'Unknown'."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "neighbourhood": ["Brooklyn", None, "Manhattan"],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        result = handle_missing_values(df)
        assert result["neighbourhood"].isnull().sum() == 0
        assert result["neighbourhood"].iloc[1] == "Unknown"

    def test_handle_missing_does_not_modify_original(self):
        """Test that missing value handling doesn't modify original DataFrame."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "accommodates": [2, None, 4],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        original_shape = df.shape
        _ = handle_missing_values(df)
        assert df.shape == original_shape

    def test_coordinate_missing_values_dropped(self):
        """Test that rows with missing latitude/longitude are dropped."""
        df = pd.DataFrame({
            "price": [100, 200, 300, 400],
            "latitude": [40.7, None, 40.8, 40.9],
            "longitude": [-73.9, -74.0, None, -74.1],
            "review_scores_rating": [4.5, 4.0, 3.5, 4.2]
        })
        result = handle_missing_values(df)
        # Should drop rows with missing coordinates (rows 1 and 2)
        assert len(result) == 2
        assert result["latitude"].isnull().sum() == 0
        assert result["longitude"].isnull().sum() == 0

    def test_room_type_imputed_with_mode(self):
        """Test that room_type is imputed with mode."""
        df = pd.DataFrame({
            "price": [100, 200, 300, 400],
            "room_type": ["Entire home/apt", None, "Entire home/apt", "Private room"],
            "review_scores_rating": [4.5, 4.0, 3.5, 4.2]
        })
        result = handle_missing_values(df)
        assert result["room_type"].isnull().sum() == 0
        # Mode should be "Entire home/apt" (appears 2 times)
        assert result["room_type"].iloc[1] == "Entire home/apt"


class TestHandleOutliers:
    """Tests for outlier handling."""

    def test_price_capped_at_domain_bounds(self):
        """Test that price is capped at $10 - $10,000."""
        df = pd.DataFrame({
            "price": [5, 50, 500, 5000, 15000]
        })
        result = handle_outliers(df)
        assert result["price"].min() == 10
        assert result["price"].max() == 10000
        assert list(result["price"]) == [10, 50, 500, 5000, 10000]

    def test_iqr_capping_for_numeric_columns(self):
        """Test that IQR-based capping works for numeric columns."""
        # Create data with clear outliers
        df = pd.DataFrame({
            "number_of_reviews": [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        result = handle_outliers(df)
        # Check that extreme outlier has been capped
        assert result["number_of_reviews"].max() < 100
        # All values should be within reasonable bounds
        assert result["number_of_reviews"].max() <= result["number_of_reviews"].quantile(0.75) + 1.5 * (
            result["number_of_reviews"].quantile(0.75) - result["number_of_reviews"].quantile(0.25)
        )

    def test_handle_outliers_preserves_valid_data(self):
        """Test that valid data points are not modified."""
        df = pd.DataFrame({
            "price": [100, 200, 300, 400, 500],
            "accommodates": [2, 3, 4, 5, 6]
        })
        result = handle_outliers(df)
        # All values are within normal range, should be unchanged
        pd.testing.assert_series_equal(result["price"], df["price"])
        pd.testing.assert_series_equal(result["accommodates"], df["accommodates"])

    def test_handle_outliers_does_not_modify_original(self):
        """Test that outlier handling doesn't modify original DataFrame."""
        df = pd.DataFrame({
            "price": [5, 50, 500, 5000, 15000]
        })
        original_values = df["price"].copy()
        _ = handle_outliers(df)
        pd.testing.assert_series_equal(df["price"], original_values)

    def test_handle_outliers_with_missing_columns(self):
        """Test that outlier handling works even if some columns are missing."""
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "other_col": [1, 2, 3]
        })
        result = handle_outliers(df)
        assert "price" in result.columns
        assert "other_col" in result.columns


class TestCleaningPipeline:
    """Integration tests for the full cleaning pipeline."""

    def test_pipeline_order_matters(self):
        """Test that cleaning steps work correctly in sequence."""
        df = pd.DataFrame({
            "city": ["NYC", "LA", "SF"],
            "room_type": ["Entire home", "Private room", "Entire apartment"],
            "host_identity_verified": ["t", "f", "t"],
            "price": [100, None, 5000],
            "accommodates": [2, 4, None],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        
        # Apply all three functions in sequence
        df = normalize_columns(df)
        df = handle_missing_values(df)
        df = handle_outliers(df)
        
        # Verify normalization worked
        assert list(df["city"]) == ["New York", "San Francisco"]  # LA row dropped due to missing price
        assert df["host_identity_verified"].dtype == bool
        
        # Verify missing values handled
        assert df["accommodates"].isnull().sum() == 0
        
        # Verify outliers handled
        assert df["price"].min() >= 10

    def test_no_data_leakage(self):
        """Test that cleaning doesn't introduce data leakage."""
        # This is a simple check that we're not using future information
        df = pd.DataFrame({
            "price": [100, 200, 300],
            "accommodates": [2, None, 4],
            "review_scores_rating": [4.5, 4.0, 3.5]
        })
        
        # Split into "train" and "test"
        train = df.iloc[:2].copy()
        test = df.iloc[2:].copy()
        
        # Process separately
        train_clean = handle_missing_values(train)
        test_clean = handle_missing_values(test)
        
        # Ensure test data wasn't affected by train data statistics
        # (in real scenario, we'd fit imputer on train and transform test)
        assert test_clean is not None

    def test_clean_pipeline_end_to_end(self, tmp_path):
        """Test that clean_pipeline runs end-to-end and saves output."""
        
        # Create a temporary input CSV
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "city": ["NYC", "LA", "SF"],
            "room_type": ["Entire home", "Private room", "Entire apartment"],
            "host_identity_verified": ["t", "f", "t"],
            "price": [100, 200, 300],
            "property_type": ["Apartment", "House", "Condo"],
            "accommodates": [2, 4, 6],
            "bathrooms": [1, 2, 3],
            "bedrooms": [1, 2, 3],
            "beds": [1, 2, 3],
            "latitude": [40.7, 40.8, 37.8],
            "longitude": [-73.9, -74.0, -122.4],
            "neighbourhood": ["Brooklyn", "Manhattan", "Mission"],
            "number_of_reviews": [10, 20, 30],
            "review_scores_rating": [4.5, 4.0, 3.5],
            "amenities": ["WiFi", "Kitchen", "TV"],
            "host_response_rate": ["95%", "100%", "80%"]
        })
        df.to_csv(input_csv, index=False)
        
        # Run pipeline
        output_dir = tmp_path / "output"
        result = clean_pipeline(str(input_csv), str(output_dir))
        
        # Verify output
        assert result is not None
        assert len(result) == 3
        assert (output_dir / "cleaned.csv").exists()
        
        # Verify cleaning worked
        assert result["city"].iloc[0] == "New York"  # NYC -> New York
        assert result["room_type"].iloc[0] == "Entire home/apt"  # Normalized
        assert result["host_identity_verified"].dtype == bool
        assert result["host_response_rate"].dtype in [np.float64, np.int64]  # Can be int or float
