import argparse
import logging
from pathlib import Path
import pandas as pd

from engineering import engineer_features
from selection import (
    create_train_test_split, 
    bin_target_variable,
    drop_unwanted_features,
    prepare_ready_features
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(input_path: str, output_dir: str = "data/processed",
                train_ratio: float = 0.8, test_ratio: float = 0.2,
                drop_low_ratings: bool = True):
    """Run the complete feature engineering pipeline based on EDA insights."""

    logger.info("="*70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*70)
    
    # Load cleaned data
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Engineer features
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Feature Engineering")
    logger.info("="*70)
    df_featured = engineer_features(df)
    
    # Bin target variable and optionally drop Low ratings
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Target Variable Binning")
    logger.info("="*70)
    
    
    df_featured = bin_target_variable(
        df_featured,
        target_col="review_scores_rating",
        new_col_name="rating_category",
        drop_low_ratings=drop_low_ratings
    )
    
    # Drop unwanted features (beds, number_of_reviews from EDA)
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Feature Selection (Drop Unwanted Features)")
    logger.info("="*70)
    df_featured = drop_unwanted_features(df_featured)
    
    # Save featured data (all features for analysis)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    featured_path = output_path / "featured.csv"
    df_featured.to_csv(featured_path, index=False)
    logger.info(f"\nSaved featured data (for analysis) to {featured_path}")
    logger.info(f"  Featured dataset: {df_featured.shape[0]} rows × {df_featured.shape[1]} columns")
    
    # Prepare ready features for modeling (drop high-cardinality and high-correlation)
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Prepare Ready Features (Correlation & Cardinality Cleanup)")
    logger.info("="*70)
    df_ready = prepare_ready_features(df_featured)

    # Save ready features data (for modeling)
    ready_path = output_path / "ready_features.csv"
    df_ready.to_csv(ready_path, index=False)
    logger.info(f"\nSaved ready features (for modeling) to {ready_path}")
    logger.info(f"  Ready dataset: {df_ready.shape[0]} rows × {df_ready.shape[1]} columns")
    
    # Create train/test splits with stratification (using ready features)
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Train/Test Split (80/20 with Stratification)")
    logger.info("="*70)
    logger.info("  Using ready_features.csv for splits")
    logger.info("  Validation will be handled by k-fold CV during training")
    train_df, test_df = create_train_test_split(
        df_ready,
        target_col="rating_category",
        stratify_cols=["rating_category", "room_type"],
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        random_state=42
    )
    
    # Save splits
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"\nSaved train set to {train_path}")
    logger.info(f"Saved test set to {test_path}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Featured data (analysis):  {df_featured.shape}")
    logger.info(f"Ready features (modeling): {df_ready.shape}")
    logger.info(f"Train set (80%):           {train_df.shape}")
    logger.info(f"Test set (20%):            {test_df.shape}")
    logger.info(f"\nTarget classes: {sorted(df_ready['rating_category'].unique())}")
    logger.info(f"All files saved to {output_path}/")
    
    # Final feature list (ready for modeling)
    logger.info("\n" + "="*70)
    logger.info("FEATURES FOR MODELING (ready_features.csv)")
    logger.info("="*70)
    feature_cols = [col for col in df_ready.columns if col not in ['rating_category']]
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Feature list: {feature_cols}")
    
    # Analysis feature list (for reference)
    logger.info("\n" + "="*70)
    logger.info("FEATURES FOR ANALYSIS (featured.csv)")
    logger.info("="*70)
    analysis_feature_cols = [col for col in df_featured.columns if col not in ['review_scores_rating', 'rating_category']]
    logger.info(f"Total features: {len(analysis_feature_cols)}")
    dropped_for_modeling = set(analysis_feature_cols) - set(feature_cols)
    if dropped_for_modeling:
        logger.info(f"Features in analysis but not modeling: {sorted(dropped_for_modeling)}")
    
    return df_featured, df_ready, train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete feature engineering pipeline")
    parser.add_argument("--input", default="data/processed/cleaned.csv", 
                       help="Path to cleaned CSV file")
    parser.add_argument("--output", default="data/processed",
                       help="Directory to save output files")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Proportion of data for training (default: 0.8)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--keep-low-ratings", action="store_true",
                       help="Keep Low Rating category (default: drop it based on EDA)")
    
    args = parser.parse_args()
    
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        drop_low_ratings=not args.keep_low_ratings
    )

# Usage:
#     python feature_engineering/pipeline.py [--input data/processed/cleaned.csv]