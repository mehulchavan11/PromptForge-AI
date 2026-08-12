import pandas as pd
from typing import Tuple, Dict, Any, Optional

def load_data(file, filename: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Ingests CSV, Excel (.xlsx), or JSON files into a Pandas DataFrame.
    Performs initial data validation and automated cleaning.
    
    Returns:
        Tuple[Optional[pd.DataFrame], Dict[str, Any]]: Cleaned DataFrame and execution status report.
    """
    status = {
        "success": False,
        "filename": filename,
        "error": None,
        "rows": 0,
        "columns": 0,
        "column_names": []
    }
    
    try:
        # 1. Multi-Format Ingestion
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif filename.endswith(".json"):
            df = pd.read_json(file)
        else:
            status["error"] = f"Unsupported file type for '{filename}'. Allowed: CSV, XLSX, JSON."
            return None, status

        # 2. Basic Validation Checks
        if df is None or df.empty:
            status["error"] = "Uploaded file is empty or could not be parsed."
            return None, status

        # 3. Automated Cleaning & Normalization
        # Strip leading/trailing whitespace and normalize column names (lowercase, underscores)
        df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Drop completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Update metadata
        status["success"] = True
        status["rows"] = len(df)
        status["columns"] = len(df.columns)
        status["column_names"] = list(df.columns)
        
        return df, status

    except Exception as e:
        status["error"] = f"Failed to process file: {str(e)}"
        return None, status