import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_df():
    """A small synthetic DataFrame mimicking the merged Airbnb dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "property_type": np.random.choice(["Apartment", "House", "Condo"], n),
        "room_type": np.random.choice(["Entire home/apt", "Private room", "Shared room"], n),
        "amenities": ["{TV,Wifi,Kitchen}"] * n,
        "accommodates": np.random.randint(1, 10, n),
        "bathrooms": np.random.choice([1.0, 1.5, 2.0], n),
        "city": np.random.choice(["New York", "Los Angeles", "Chicago"], n),
        "host_identity_verified": np.random.choice([True, False], n),
        "host_response_rate": np.random.uniform(50, 100, n),
        "latitude": np.random.uniform(25, 48, n),
        "longitude": np.random.uniform(-122, -71, n),
        "neighbourhood": np.random.choice(["Downtown", "Midtown", "Uptown", None], n),
        "number_of_reviews": np.random.randint(0, 500, n),
        "review_scores_rating": np.random.uniform(3.0, 5.0, n),
        "bedrooms": np.random.choice([0, 1, 2, 3], n).astype(float),
        "beds": np.random.choice([1, 2, 3, 4], n).astype(float),
        "price": np.random.uniform(30, 500, n),
    })


@pytest.fixture
def tmp_csv(sample_df, tmp_path):
    """Write sample_df to a temporary CSV and return its path."""
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)
