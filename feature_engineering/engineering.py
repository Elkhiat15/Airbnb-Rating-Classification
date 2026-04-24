import pandas as pd
import numpy as np
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def add_amenity_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of amenities per listing"""

    df = df.copy()
    
    if 'amenities' not in df.columns:
        logger.warning("amenities column not found, skipping amenity_count")
        return df
    
    # Amenities are stored as JSON-like strings: '{"WiFi","Kitchen","TV"}'
    def count_amenities(amenities_str):
        if pd.isna(amenities_str) or amenities_str == '':
            return 0
        # Remove curly braces and split by comma
        amenities_str = str(amenities_str).strip('{}')
        if not amenities_str:
            return 0

        items = [item.strip() for item in amenities_str.split(',') if item.strip()]
        return len(items)
    
    df['amenity_count'] = df['amenities'].apply(count_amenities)
        
    return df


def add_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Create price_per_bed and price_per_guest features."""

    df = df.copy()
    
    # Price per bed
    if 'price' in df.columns and 'beds' in df.columns:
        df['price_per_bed'] = df.apply(
            lambda row: row['price'] / row['beds'] if row['beds'] > 0 else row['price'],
            axis=1
        )
        logger.info(f"  Created price_per_bed feature (mean: ${df['price_per_bed'].mean():.2f})")
    
    # Price per guest (handle division by zero)
    if 'price' in df.columns and 'accommodates' in df.columns:
        df['price_per_guest'] = df.apply(
            lambda row: row['price'] / row['accommodates'] if row['accommodates'] > 0 else row['price'],
            axis=1
        )
    
    return df


def add_categorical_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # is_entire_home flag
    if 'room_type' in df.columns:
        df['is_entire_home'] = (df['room_type'] == 'Entire home/apt').astype(int)
    
    return df


def add_listing_density(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    if 'neighbourhood' not in df.columns:
        logger.warning("neighbourhood column not found, skipping listing_density")
        return df
    
    # Count listings per neighbourhood
    density_map = df['neighbourhood'].value_counts().to_dict()
    df['listing_density'] = df['neighbourhood'].map(density_map)
    
    
    return df


def add_price_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price relative to room_type and city medians.
    These features capture whether a listing is expensive/cheap for its category,
    which is critical since price gaps far exceed rating gaps.
    """

    df = df.copy()
    
    # Price relative to room type median 
    if 'price' in df.columns and 'room_type' in df.columns:
        room_type_stats = df.groupby('room_type')['price'].agg(['median', 'std'])
        
        df['price_relative_to_room_type'] = df.apply(
            lambda row: (row['price'] - room_type_stats.loc[row['room_type'], 'median']) / 
                       (room_type_stats.loc[row['room_type'], 'std'] if room_type_stats.loc[row['room_type'], 'std'] > 0 else 1),
            axis=1
        )
    
    # Price relative to city median
    if 'price' in df.columns and 'city' in df.columns:
        city_stats = df.groupby('city')['price'].agg(['median', 'std'])
        
        df['price_relative_to_city'] = df.apply(
            lambda row: (row['price'] - city_stats.loc[row['city'], 'median']) / 
                       (city_stats.loc[row['city'], 'std'] if city_stats.loc[row['city'], 'std'] > 0 else 1),
            axis=1
        )
    
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transforms to right-skewed features."""

    df = df.copy()
    
    # Log transform price (right-skewed)
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
        logger.info(f"  Created log_price feature")
    
    # Log transform number_of_reviews (right-skewed)
    if 'number_of_reviews' in df.columns:
        df['log_number_of_reviews'] = np.log1p(df['number_of_reviews'])
        logger.info(f"  Created log_number_of_reviews feature")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    
    logger.info("Starting feature engineering...")
    initial_cols = df.shape[1]
    
    # Existing features
    df = add_amenity_count(df)
    df = add_price_ratios(df)
    df = add_categorical_flags(df)
    df = add_listing_density(df)
    
    # New priority features from EDA
    df = add_price_relative_features(df)
    df = add_log_transforms(df)
    
    new_cols = df.shape[1] - initial_cols
    logger.info(f"Feature engineering complete: added {new_cols} new features")
    logger.info(f"  Final shape: {df.shape}")
    
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run feature engineering")
    parser.add_argument("--data", default="data/processed/cleaned.csv")
    parser.add_argument("--output", default="data/processed/featured.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = engineer_features(df)
    df.to_csv(args.output, index=False)
    print(f"Saved {df.shape} to {args.output}")
