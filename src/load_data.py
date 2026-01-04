# File: src/data/load_data.py

import pandas as pd
from pathlib import Path
import time
from typing import Tuple, Dict, Any
import logging

def load_complaints_data(data_path: Path) -> pd.DataFrame:
    """
    Load the CFPB complaints dataset with optimized data types.
    
    Args:
        data_path: Path to the complaints CSV file
        
    Returns:
        pd.DataFrame: Loaded complaints data
    """
    # Define data types for memory optimization
    dtypes = {
        'Complaint ID': 'str',
        'Product': 'category',
        'Sub-product': 'category',
        'Issue': 'category',
        'Sub-issue': 'category',
        'Consumer complaint narrative': 'str',
        'Company public response': 'category',
        'Company': 'category',
        'State': 'category',
        'ZIP code': 'str',
        'Consumer consent provided?': 'category',
        'Submitted via': 'category',
        'Company response to consumer': 'category',
        'Timely response?': 'category',
        'Consumer disputed?': 'category'
    }
    
    # Columns to parse as dates
    date_columns = ['Date received', 'Date sent to company']
    
    try:
        logging.info(f"Loading data from {data_path}...")
        start_time = time.time()
        
        df = pd.read_csv(
            data_path,
            dtype=dtypes,
            parse_dates=date_columns,
            infer_datetime_format=True,
            low_memory=False
        )
        
        logging.info(f"Data loaded successfully in {time.time() - start_time:.2f} seconds")
        logging.info(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the loaded dataset.
    
    Args:
        df: Loaded DataFrame
        
    Returns:
        Dict containing summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().mean() * 100).round(2).to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2)  # in MB
    }
    return summary