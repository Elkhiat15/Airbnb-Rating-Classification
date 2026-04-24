import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import logging

logger = logging.getLogger(__name__)


def encode_categoricals(df: pd.DataFrame, columns: list[str], drop_first: bool = False) -> pd.DataFrame:
    """One-hot encode the given categorical columns."""

    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
            
        # Get dummies and add to dataframe
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        
        # Drop original column
        df = df.drop(columns=[col])
        
        logger.info(f"  Encoded {col}: created {len(dummies.columns)} dummy variables")
    
    return df


def scale_numerics(df: pd.DataFrame, columns: list[str], method: str = "standard") -> pd.DataFrame:
    """Scale numeric columns using StandardScaler or MinMaxScaler."""

    df = df.copy()
    
    # Select scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'minmax'.")
    
    existing_cols = [col for col in columns if col in df.columns]
    
    if len(existing_cols) == 0:
        logger.warning(f"None of the specified columns found in DataFrame")
        return df
    
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    
    logger.info(f"  Scaled {len(existing_cols)} columns using {method} scaler")
    
    return df


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        # Apply log1p (log(1 + x)) to handle zeros
        df[col] = np.log1p(df[col])
        
        logger.info(f"  Log-transformed {col}")
    
    return df
