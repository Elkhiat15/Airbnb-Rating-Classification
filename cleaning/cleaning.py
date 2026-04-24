import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Alias / normalisation maps ──────────────────────────────
CITY_ALIASES = {
    "NYC": "New York", "New York City": "New York", "New York, NY": "New York",
    "New York, NY, USA": "New York",
    "LA": "Los Angeles", "Los Angeles, CA": "Los Angeles",
    "Los Angeles, CA, USA": "Los Angeles",
    "SF": "San Francisco", "San Francisco, CA": "San Francisco",
    "San Francisco, CA, USA": "San Francisco",
    "DC": "Washington DC", "Washington": "Washington DC",
    "Washington, DC": "Washington DC", "Washington, DC, USA": "Washington DC",
    "Washington DC, USA": "Washington DC",
    "Chicago, IL": "Chicago", "Chicago, IL, USA": "Chicago",
    "Boston, MA": "Boston", "Boston, MA, USA": "Boston",
}

ROOM_TYPE_ALIASES = {
    "Entire home": "Entire home/apt", "Entire apartment": "Entire home/apt",
    "Entire place": "Entire home/apt", "Entire house": "Entire home/apt",
    "Entire bungalow": "Entire home/apt", "Entire cabin": "Entire home/apt",
    "Entire condo": "Entire home/apt", "Entire cottage": "Entire home/apt",
    "Entire guest suite": "Entire home/apt", "Entire guesthouse": "Entire home/apt",
    "Entire loft": "Entire home/apt", "Entire rental unit": "Entire home/apt",
    "Entire serviced apartment": "Entire home/apt",
    "Entire townhouse": "Entire home/apt", "Entire villa": "Entire home/apt",
    "Entire vacation home": "Entire home/apt",
}


def standardize_room_type(room_type_str):
    """
    Standardize room_type using pattern matching for robustness.
    Handles malformed entries, case sensitivity, and variations.
    """
    if pd.isna(room_type_str):
        return room_type_str
    
    # Convert to string and clean
    s = str(room_type_str).strip()
    s_lower = s.lower()
    
    # Check for "entire" patterns
    if s_lower.startswith("entire"):
        return "Entire home/apt"
    
    # Check for "private" patterns
    if "private" in s_lower:
        return "Private room"
    
    # Check for "shared" patterns
    if "shared" in s_lower:
        return "Shared room"
    
    # Handle generic "Room" 
    if s_lower == "room":
        return "Private room"
    
    return s

HOST_VERIFIED_MAP = {
    "t": True, "f": False, "True": True, "False": False, "1": True, "0": False,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise known aliases to canonical values."""
    df = df.copy()
    
    # Normalize city names
    if "city" in df.columns:
        df["city"] = df["city"].replace(CITY_ALIASES)
        logger.info(f"  Normalized city values: {df['city'].nunique()} unique cities")
    
    if "room_type" in df.columns:
        # First apply dictionary-based aliases
        df["room_type"] = df["room_type"].replace(ROOM_TYPE_ALIASES)
        # Then apply pattern-based standardization for variant entries
        df["room_type"] = df["room_type"].apply(standardize_room_type)
        logger.info(f"  Normalized room_type: {df['room_type'].nunique()} unique types")
    
    if "host_identity_verified" in df.columns:
        df["host_identity_verified"] = df["host_identity_verified"].replace(HOST_VERIFIED_MAP)

        df["host_identity_verified"] = df["host_identity_verified"].astype(bool)
        logger.info(f"  Normalized host_identity_verified to boolean")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values with per-column strategy."""
    df = df.copy()
    initial_rows = len(df)
    
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = pd.to_numeric(
            df["host_response_rate"].astype(str).str.replace("%", ""), 
            errors="coerce"
        )
    
    # drop rows with missing values (< 0.01% null)
    critical_cols = ["price", "city", "property_type"]
    for col in critical_cols:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"  Dropped {dropped} rows with missing {col}")
    
    if "room_type" in df.columns and df["room_type"].isnull().any():
        missing_count = df["room_type"].isnull().sum()
        df["room_type"] = df["room_type"].fillna(df["room_type"].mode()[0] if not df["room_type"].mode().empty else "Unknown")
        logger.info(f"  Imputed {missing_count} missing room_type with mode")
    
    # Numeric columns: impute with median (grouped by room_type when available)
    numeric_impute_cols = ["accommodates", "bathrooms", "bedrooms", "beds"]
    for col in numeric_impute_cols:
        if col in df.columns and df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            if "room_type" in df.columns and df["room_type"].notna().all():

                df[col] = df.groupby("room_type")[col].transform(
                    lambda x: x.fillna(x.median())
                )

                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            else:

                df[col] = df[col].fillna(df[col].median())
            logger.info(f"  Imputed {missing_count} missing values in {col} with median")
    
    if "host_response_rate" in df.columns:
        if df["host_response_rate"].isnull().any():
            missing_count = df["host_response_rate"].isnull().sum()

            df["has_response_rate"] = (~df["host_response_rate"].isnull()).astype(int)
            # Impute with median
            df["host_response_rate"] = df["host_response_rate"].fillna(df["host_response_rate"].median())
            logger.info(f"  Imputed {missing_count} missing host_response_rate with median; added has_response_rate flag")
    
    # review_scores_rating: drop rows with missing target
    if "review_scores_rating" in df.columns:
        before = len(df)
        df = df.dropna(subset=["review_scores_rating"])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"  Dropped {dropped} rows with missing target (review_scores_rating)")
    
    # neighbourhood: impute with "Unknown"
    if "neighbourhood" in df.columns and df["neighbourhood"].isnull().any():
        missing_count = df["neighbourhood"].isnull().sum()
        df["neighbourhood"] = df["neighbourhood"].fillna("Unknown")
        logger.info(f"  Imputed {missing_count} missing neighbourhood with 'Unknown'")
    
    coordinate_cols = ["latitude", "longitude"]
    for col in coordinate_cols:
        if col in df.columns and df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            # For coordinates, dropping rows is better than imputing
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"  Dropped {dropped} rows with missing {col}")
    
    logger.info(f"  Missing value handling: {initial_rows} : {len(df)} rows ({initial_rows - len(df)} dropped)")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap or remove outliers based on domain knowledge."""
    df = df.copy()

    if "price" in df.columns:
        before_min = df["price"].min()
        before_max = df["price"].max()
        df["price"] = df["price"].clip(lower=10, upper=10000)
        after_min = df["price"].min()
        after_max = df["price"].max()
        logger.info(f"  Capped price: [{before_min:.2f}, {before_max:.2f}] → [{after_min:.2f}, {after_max:.2f}]")
    
    # IQR-based capping for numeric columns
    numeric_cols = ["accommodates", "bathrooms", "bedrooms", "beds", 
                    "number_of_reviews"]
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Skip if IQR is too small (prevents collapsing to single value)
            if IQR < 0.01:
                logger.info(f"  Skipped IQR capping for {col} (IQR={IQR:.4f} too small)")
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"  Capped {outliers_count} outliers in {col} using IQR method (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
    
    return df


def clean_pipeline(input_path: str, output_dir: str) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"  Raw shape: {df.shape}")

    df = normalize_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)

    logger.info(f"  Cleaned shape: {df.shape}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cleaned.csv", index=False)
    logger.info(f"  Saved to {out / 'cleaned.csv'}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data cleaning pipeline")
    parser.add_argument("--data", default="data/merged/merged_airbnb_data.csv")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    clean_pipeline(args.data, args.output)

# Usage:
#  python cleaning/cleaning.py [--data path/to/merged.csv] [--output data/processed/]