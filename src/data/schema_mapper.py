import pandas as pd
import re

def detect_schema_roles(df: pd.DataFrame) -> str:
    """
    Scans dataframe headers and data types to auto-detect semantic roles.
    Returns a formatted schema string for the AI Orchestrator.
    """
    schema_parts = []
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        col_lower = str(col).lower()
        role = "attribute"
        
        # 🛡️ Safe Pandas 2.x dtype checking (Replaces np.issubdtype)
        is_date = pd.api.types.is_datetime64_any_dtype(df[col])
        is_num = pd.api.types.is_numeric_dtype(df[col])
        
        # Regex scoring for semantic roles
        if re.search(r'(date|time|year|month|day)', col_lower) or is_date:
            role = "temporal"
        elif re.search(r'(price|cost|revenue|profit|sales|amount|total|money|cash)', col_lower) and is_num:
            role = "financial_metric"
        elif re.search(r'(id|uuid|guid|key)', col_lower):
            role = "identifier"
        elif re.search(r'(qty|quantity|count)', col_lower) and is_num:
            role = "quantitative_metric"
        
        schema_parts.append(f"{col} ({dtype_str} -> {role})")
        
    return ", ".join(schema_parts)