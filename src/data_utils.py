# File: src/utils/data_utils.py

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any

def load_complaints_data(filepath: Path) -> pd.DataFrame:
    """
    Load CFPB complaints data with optimized data types.
    Returns the loaded DataFrame.
    """
    dtypes = {
        'Complaint ID': 'str',
        'Product': 'category',
        'Sub-product': 'category',
        'Issue': 'category',
        'Sub-issue': 'category',
        'Consumer complaint narrative': 'str',
        'Company': 'category',
        'State': 'category',
        'ZIP code': 'str',
        'Consumer consent provided?': 'category',
        'Submitted via': 'category',
        'Company response to consumer': 'category',
        'Timely response?': 'category',
        'Consumer disputed?': 'category'
    }
    
    date_columns = ['Date received', 'Date sent to company']
    
    try:
        return pd.read_csv(
            filepath,
            dtype=dtypes,
            parse_dates=date_columns,
            infer_datetime_format=True,
            low_memory=False
        )
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic statistics about the DataFrame."""
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sort_values(ascending=False),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2)
    }

def filter_products(df: pd.DataFrame, product_list: list) -> pd.DataFrame:
    """Filter DataFrame to include only specified products."""
    return df[df['Product'].isin(product_list)].copy()

def clean_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess narrative text."""
    df = df.copy()
    df['narrative_clean'] = df['Consumer complaint narrative'].str.lower().str.strip()
    df['narrative_length'] = df['narrative_clean'].str.len()
    return df