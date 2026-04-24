import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Remove highly correlated features (keeps one of each pair)."""
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find columns with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Columns to keep
    to_keep = [col for col in df.columns if col not in to_drop]
    
    logger.info(f"  Correlation filter: removed {len(to_drop)} highly correlated features (threshold={threshold})")
    if to_drop:
        logger.info(f"    Dropped: {to_drop}")
    
    return to_keep


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series, top_k: int = 15) -> list[str]:
    """Rank features by mutual information with the target."""

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    # Sort by score and get top k
    top_features = mi_series.nlargest(top_k).index.tolist()
    
    logger.info(f"  Mutual information: selected top {top_k} features")
    logger.info(f"    Top 5: {top_features[:5]}")
    
    return top_features


def select_features(df: pd.DataFrame, target_col: str = "review_scores_rating", 
                   use_correlation_filter: bool = True,
                   use_mi_ranking: bool = False,
                   mi_top_k: int = 15) -> pd.DataFrame:
    """Run feature selection pipeline."""

    logger.info("Starting feature selection...")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Select only numeric features for correlation analysis
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    features_to_keep = numeric_features.copy()
    
    # Apply correlation filter
    if use_correlation_filter and len(numeric_features) > 0:
        X_numeric = X[numeric_features]
        features_to_keep = correlation_filter(X_numeric, threshold=0.95)
    
    # Apply mutual information ranking
    if use_mi_ranking and len(features_to_keep) > 0:
        X_selected = X[features_to_keep]
        features_to_keep = mutual_information_ranking(X_selected, y, top_k=mi_top_k)
    
    # Create final DataFrame with selected features and target
    result = pd.concat([X[features_to_keep], y], axis=1)
    
    logger.info(f"Feature selection complete: {len(X.columns)} -> {len(features_to_keep)} features")
    
    return result


def categorize_rating(rating: float, thresholds: tuple = (3.0, 4.51, 4.91)) -> str:
    """Categorize review scores into discrete rating classes."""

    low_thresh, med_thresh, high_thresh = thresholds
    
    if 0 <= rating < low_thresh:
        return 'Low Rating'
    elif low_thresh <= rating < med_thresh:
        return 'Medium Rating'
    elif med_thresh <= rating < high_thresh:
        return 'High Rating'
    elif high_thresh <= rating <= 5.0:
        return 'Very High Rating'
    else:
        return None


def bin_target_variable(df: pd.DataFrame, target_col: str = "review_scores_rating",
                       thresholds: tuple = (3.0, 4.51, 4.91),
                       new_col_name: str = "rating_category",
                       drop_low_ratings: bool = True) -> pd.DataFrame:
    """Apply rating categorization to the target variable and optionally drop Low ratings."""

    df = df.copy()
    df[new_col_name] = df[target_col].apply(lambda x: categorize_rating(x, thresholds))
    
    # Log the distribution before dropping
    logger.info(f"  Binned target variable '{target_col}' -> '{new_col_name}'")
    logger.info(f"  Class distribution (before dropping Low):")
    for category, count in df[new_col_name].value_counts().sort_index().items():
        pct = (count / len(df)) * 100
        logger.info(f"    {category}: {count} ({pct:.2f}%)")
    
    # Drop Low Rating category if requested (based on EDA: only 0.43% of data)
    if drop_low_ratings:
        initial_rows = len(df)
        df = df[df[new_col_name] != 'Low Rating'].copy()
        dropped_rows = initial_rows - len(df)
        logger.info(f"  Dropped {dropped_rows} Low Rating rows ({dropped_rows/initial_rows*100:.2f}%)")
        
        logger.info(f"  Final class distribution:")
        for category, count in df[new_col_name].value_counts().sort_index().items():
            pct = (count / len(df)) * 100
            logger.info(f"    {category}: {count} ({pct:.2f}%)")
    
    return df


def drop_unwanted_features(df: pd.DataFrame, features_to_drop: list = None) -> pd.DataFrame:
    """Drop features identified as redundant or weak predictors from EDA."""

    df = df.copy()
    
    if features_to_drop is None:
        # Default from EDA: drop beds (redundant), and weak predictors
        features_to_drop = ['beds', 'number_of_reviews']
    
    # Only drop features that actually exist
    features_to_drop = [f for f in features_to_drop if f in df.columns]
    
    if features_to_drop:
        df = df.drop(columns=features_to_drop)
        logger.info(f"  Dropped {len(features_to_drop)} unwanted features: {features_to_drop}")
    else:
        logger.info(f"  No features to drop (all specified features already removed or don't exist)")
    
    return df


def prepare_ready_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling by removing high-cardinality and highly-correlated features."""
    
    df = df.copy()
    
    # Features to drop based on analysis
    features_to_drop = [
        'review_scores_rating',
        # High cardinality
        'id',
        'amenities',
        'latitude',
        'longitude',
        'price',  # Redundant with log_price and price_relative features
        'price_relative_to_city',  # Highly correlated with price (r=0.98) and log_price (r=0.84)
        'price_per_guest',  # Highly correlated with price_per_bed (r=0.81), and lower target correlation
        
        # Low cardinality
        'host_identity_verified',  
        'has_response_rate',  
        'is_entire_home' 
    ]
    
    # Only drop features that actually exist
    features_to_drop = [f for f in features_to_drop if f in df.columns]
    
    if features_to_drop:
        df = df.drop(columns=features_to_drop)
        logger.info(f"  Dropped {len(features_to_drop)} features for modeling readiness:")
        logger.info(f"    High cardinality: id, amenities, latitude, longitude")
        logger.info(f"    High correlation: price, price_relative_to_city, price_per_guest")
        logger.info(f"  Retained features: {len(df.columns) - 2} (excluding review_scores_rating, rating_category)")
    
    return df


def create_train_test_split(df: pd.DataFrame, 
                           target_col: str = "rating_category",
                           stratify_cols: list = None,
                           train_ratio: float = 0.8,
                           test_ratio: float = 0.2, 
                           random_state: int = 42) -> tuple:
    """Split data into train and test sets with stratification."""

    assert abs(train_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    if stratify_cols is None:
        stratify_cols = [target_col]
    

    existing_stratify_cols = [col for col in stratify_cols if col in df.columns]
    
    if not existing_stratify_cols:
        logger.warning(f"None of the stratify columns {stratify_cols} found in DataFrame. Using target only.")
        stratify_key = df[target_col]
    else:
        stratify_key = df[existing_stratify_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    train_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=random_state, stratify=stratify_key
    )
    
    logger.info(f"  Train split: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test split:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Log stratification verification
    logger.info(f"  Stratification verification:")
    for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
        target_dist = split_df[target_col].value_counts(normalize=True).sort_index()
        logger.info(f"    {split_name} - {target_col} distribution: {target_dist.to_dict()}")
        if 'room_type' in split_df.columns:
            room_dist = split_df['room_type'].value_counts(normalize=True).sort_index()
            logger.info(f"    {split_name} - room_type distribution: {room_dist.to_dict()}")
    
    return train_df, test_df
