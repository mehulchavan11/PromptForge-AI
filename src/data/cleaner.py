import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard data scrubbing techniques to prepare data for SQL ingestion.
    """
    clean_df = df.copy()
    
    # 1. Remove duplicate rows
    clean_df = clean_df.drop_duplicates()
    
    # 2. Drop columns/rows that are entirely null
    clean_df = clean_df.dropna(axis=0, how='all')
    clean_df = clean_df.dropna(axis=1, how='all')
    
    # 3. Trim whitespace in string columns to prevent SQL syntax errors
    for col in clean_df.select_dtypes(include=['object', 'string']).columns:
        clean_df[col] = clean_df[col].astype(str).str.strip()
        
    return clean_df