import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def merge_sources(
    scraped_path: str = "data/raw/scraped_airbnb_listings.csv",
    kaggle_path: str = "data/raw/Airbnb_Data.csv",
    output_path: str = "data/merged/merged_airbnb_data.csv",
) -> pd.DataFrame:
    
    logger.info("="*70)
    logger.info("MERGE PIPELINE - Combining Airbnb Data Sources")
    logger.info("="*70)
    
    # Source 1: Scraped data
    logger.info(f"\nLoading Source 1 (Scraped): {scraped_path}")
    scraped_df = pd.read_csv(scraped_path)
    logger.info(f"   Loaded: {scraped_df.shape[0]:,} rows × {scraped_df.shape[1]} cols")
    
    # Source 2: Kaggle data
    logger.info(f"\nLoading Source 2 (Kaggle): {kaggle_path}")
    kaggle_df = pd.read_csv(kaggle_path)
    logger.info(f"   Loaded: {kaggle_df.shape[0]:,} rows × {kaggle_df.shape[1]} cols")
    
    logger.info("\nTransforming scraped data...")
    
    # Extract ID from URL
    logger.info("   Extracting listing IDs from URLs")
    scraped_df['id'] = scraped_df['listing_url'].str.extract(r'/rooms/(\d+)')[0].astype(np.int64)
    
    # Drop unnecessary columns
    logger.info("   Dropping unnecessary columns from scraped data")
    scraped_df = scraped_df.drop(columns=['listing_url', 'is_superhost', 'free_cancellation'])
    
    # Rename columns to match Kaggle
    logger.info("   Renaming columns to match Kaggle schema")
    scraped_df = scraped_df.rename(columns={
        'guests': 'accommodates',
        'rating': 'review_scores_rating',
        'review_count': 'number_of_reviews',
        'price_per_night': 'price'
    })
    
    logger.info("\nTransforming Kaggle data...")
    
    # Drop unnecessary columns
    logger.info("   Dropping unnecessary columns from Kaggle data")
    kaggle_df = kaggle_df.drop(columns=[
        'bed_type', 'cancellation_policy', 'cleaning_fee', 'description', 'first_review',
        'host_has_profile_pic', 'host_since', 'instant_bookable', 'last_review',
        'name', 'thumbnail_url', 'zipcode'
    ])
    
    # Transform log_price back to linear scale
    logger.info("   Converting log_price to linear scale")
    kaggle_df['price'] = np.exp(kaggle_df['log_price'])
    kaggle_df = kaggle_df.drop(columns=['log_price'])
    
    # Scale review_scores_rating from 0-100 to 0-5
    logger.info("   Scaling review_scores_rating from 0-100 to 0-5")
    kaggle_df['review_scores_rating'] = kaggle_df['review_scores_rating'] / 20
    
    logger.info("\nAligning schemas...")
    
    columns_order = kaggle_df.columns.tolist()
    logger.info(f"   Target schema: {len(columns_order)} columns")
    
    # Reorder scraped_df to match Kaggle
    scraped_df = scraped_df[columns_order]
    logger.info(f"   Scraped data: {scraped_df.shape[0]:,} rows × {scraped_df.shape[1]} cols")
    logger.info(f"   Kaggle data:  {kaggle_df.shape[0]:,} rows × {kaggle_df.shape[1]} cols")
    
    logger.info("\nConcatenating datasets...")
    merged_df = pd.concat([kaggle_df, scraped_df], ignore_index=True)
    logger.info(f"   Merged: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} cols")
    
    logger.info(f"\nSaving merged data to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"   File saved successfully")
    
    logger.info("\n" + "="*70)
    logger.info("MERGE COMPLETE")
    logger.info("="*70)
    logger.info(f"Source 1 (Scraped): {scraped_df.shape[0]:,} rows")
    logger.info(f"Source 2 (Kaggle):  {kaggle_df.shape[0]:,} rows")
    logger.info(f"Merged Dataset:     {merged_df.shape[0]:,} rows × {merged_df.shape[1]} cols")
    logger.info(f"Output: {output_path}")
    logger.info("="*70)
    
    return merged_df


if __name__ == "__main__":
    merge_sources()
