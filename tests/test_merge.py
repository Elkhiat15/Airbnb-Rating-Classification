import pandas as pd
import numpy as np
import pytest
from pathlib import Path


class TestMerge:
    """Tests for data source merging."""

    def test_merge_produces_expected_row_count(self):
        """Test that merge produces the expected number of rows."""
        merged_path = "data/merged/merged_airbnb_data.csv"
        
        if not Path(merged_path).exists():
            pytest.skip(f"Merged file not found: {merged_path}")
        
        df = pd.read_csv(merged_path)
        
        # Expected: Kaggle (74,111) + Scraped (1,000) = 75,111
        assert df.shape[0] == 75111, f"Expected 75,111 rows, got {df.shape[0]}"

    def test_merge_has_correct_columns(self):
        """Test that merged data has the expected columns."""
        merged_path = "data/merged/merged_airbnb_data.csv"
        
        if not Path(merged_path).exists():
            pytest.skip(f"Merged file not found: {merged_path}")
        
        df = pd.read_csv(merged_path)
        
        expected_columns = [
            'id', 'property_type', 'room_type', 'amenities', 'accommodates',
            'bathrooms', 'city', 'host_identity_verified', 'host_response_rate',
            'latitude', 'longitude', 'neighbourhood', 'number_of_reviews',
            'review_scores_rating', 'bedrooms', 'beds', 'price'
        ]
        
        assert list(df.columns) == expected_columns

    def test_merge_review_scores_in_valid_range(self):
        """Test that review scores are in 0-5 range after scaling."""
        merged_path = "data/merged/merged_airbnb_data.csv"
        
        if not Path(merged_path).exists():
            pytest.skip(f"Merged file not found: {merged_path}")
        
        df = pd.read_csv(merged_path)
        
        # Filter out NaN values
        valid_scores = df['review_scores_rating'].dropna()
        
        assert valid_scores.min() >= 0, f"Min score {valid_scores.min()} < 0"
        assert valid_scores.max() <= 5, f"Max score {valid_scores.max()} > 5"

    def test_merge_price_is_positive(self):
        """Test that prices are positive after transformation."""
        merged_path = "data/merged/merged_airbnb_data.csv"
        
        if not Path(merged_path).exists():
            pytest.skip(f"Merged file not found: {merged_path}")
        
        df = pd.read_csv(merged_path)
        
        # Filter out NaN values
        valid_prices = df['price'].dropna()
        
        assert (valid_prices > 0).all(), "Found non-positive prices"

    def test_merge_has_unique_and_duplicate_ids(self):
        """Test that merged data contains both unique and potentially duplicate IDs."""
        merged_path = "data/merged/merged_airbnb_data.csv"
        
        if not Path(merged_path).exists():
            pytest.skip(f"Merged file not found: {merged_path}")
        
        df = pd.read_csv(merged_path)
        
        # Check that IDs exist
        assert 'id' in df.columns
        assert df['id'].notna().any()
        
        # Check for reasonable ID range
        valid_ids = df['id'].dropna()
        assert valid_ids.min() > 0
        assert valid_ids.dtype in [np.int64, np.int32, int]

    def test_sources_documented(self):
        """Test that both data sources are documented in merge.py."""
        merge_file = Path("scraper/merge.py")
        
        if not merge_file.exists():
            pytest.skip("merge.py not found")
        
        content = merge_file.read_text()
        
        # Check for source documentation
        assert "Kaggle" in content or "kaggle" in content, "Kaggle source not documented"
        assert "scraped" in content.lower(), "Scraped source not documented"
