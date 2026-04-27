import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from scraper.merge import merge_sources


class TestMerge:
    """Tests for data source merging."""

    def test_merge_sources_executes_correctly(self, tmp_path):
        """Test that merge_sources successfully transforms and combines datasets."""

        # 1. dummy Kaggle data
        kaggle_data = pd.DataFrame(
            {
                "id": [1],
                "log_price": [np.log(150)],  # Will become price = 150
                "property_type": ["Apartment"],
                "room_type": ["Entire home/apt"],
                "amenities": ['{"WiFi"}'],
                "accommodates": [2],
                "bathrooms": [1.0],
                "bed_type": ["Real Bed"],  # Should be dropped
                "cancellation_policy": ["strict"],  # Should be dropped
                "cleaning_fee": [True],  # Should be dropped
                "city": ["NYC"],
                "description": ["Nice place"],  # Should be dropped
                "first_review": ["2020-01-01"],  # Should be dropped
                "host_has_profile_pic": ["t"],  # Should be dropped
                "host_identity_verified": ["t"],
                "host_response_rate": ["100%"],
                "host_since": ["2015-01-01"],  # Should be dropped
                "instant_bookable": ["t"],  # Should be dropped
                "last_review": ["2021-01-01"],  # Should be dropped
                "latitude": [40.7128],
                "longitude": [-74.0060],
                "name": ["Cozy Apt"],  # Should be dropped
                "neighbourhood": ["Brooklyn"],
                "number_of_reviews": [10],
                "review_scores_rating": [90],  # Will be scaled by / 20 -> 4.5
                "thumbnail_url": ["url"],  # Should be dropped
                "zipcode": ["10001"],  # Should be dropped
                "bedrooms": [1],
                "beds": [1],
            }
        )
        kaggle_path = tmp_path / "kaggle.csv"
        kaggle_data.to_csv(kaggle_path, index=False)

        # 2. dummy Scraped data
        scraped_data = pd.DataFrame(
            {
                "listing_url": [
                    "https://airbnb.com/rooms/2",
                    "https://airbnb.com/rooms/3",
                ],
                "is_superhost": [True, False],  # Should be dropped
                "free_cancellation": [True, False],  # Should be dropped
                "property_type": ["House", "Condo"],
                "room_type": ["Private room", "Entire home/apt"],
                "amenities": ['{"TV"}', '{"WiFi","Pool"}'],
                "guests": [1, 4],  # Renamed to accommodates
                "bathrooms": [1.0, 2.0],
                "city": ["NYC", "LA"],
                "host_identity_verified": ["f", "t"],
                "host_response_rate": ["90%", "100%"],
                "latitude": [40.75, 34.05],
                "longitude": [-73.98, -118.24],
                "neighbourhood": ["Manhattan", "Downtown"],
                "review_count": [5, 20],  # Renamed to number_of_reviews
                "rating": [4.0, 5.0],  # Renamed to review_scores_rating
                "bedrooms": [1, 2],
                "beds": [1, 2],
                "price_per_night": [50, 200],  # Renamed to price
            }
        )
        scraped_path = tmp_path / "scraped.csv"
        scraped_data.to_csv(scraped_path, index=False)

        # 3. output path
        output_path = tmp_path / "merged_output.csv"

        # 4. Run the merge function using our temporary mock files
        result_df = merge_sources(
            scraped_path=str(scraped_path),
            kaggle_path=str(kaggle_path),
            output_path=str(output_path),
        )

        # 5. Assertions
        assert len(result_df) == 3, "Expected 1 Kaggle row + 2 scraped rows"

        # Check that unnecessary columns were dropped
        assert "cancellation_policy" not in result_df.columns
        assert "zipcode" not in result_df.columns
        assert "is_superhost" not in result_df.columns

        # Check ID extraction from URL
        assert "id" in result_df.columns
        assert list(result_df["id"]) == [1, 2, 3]

        # Check Price Transformations (np.exp and renaming)
        assert "price" in result_df.columns
        np.testing.assert_array_almost_equal(
            result_df["price"].values, [150.0, 50.0, 200.0]
        )

        # Check Review Rating Transformations (/20 and renaming)
        assert "review_scores_rating" in result_df.columns
        np.testing.assert_array_almost_equal(
            result_df["review_scores_rating"].values, [4.5, 4.0, 5.0]
        )

        # Verify output file was correctly saved to disk
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 3

    def test_sources_documented(self):
        """Test that both data sources are documented in merge.py."""
        # Update path to merge.py if necessary depending on folder structure
        merge_file = Path("merge.py")

        if not merge_file.exists():
            # Try alternate path if root fails
            merge_file = Path("scraper/merge.py")
            if not merge_file.exists():
                pytest.skip("merge.py not found")

        content = merge_file.read_text()

        assert "Kaggle" in content or "kaggle" in content, (
            "Kaggle source not documented"
        )
        assert "scraped" in content.lower(), "Scraped source not documented"
