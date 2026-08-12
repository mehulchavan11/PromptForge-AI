import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_engine():
    """Creates and returns a secure database engine connection."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is missing from environment variables.")
    # Fix for some cloud DB providers requiring a specific dialect prefix
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

def save_dataframe_to_postgres(df: pd.DataFrame, table_name: str) -> dict:
    """
    Saves a cleaned DataFrame directly into a PostgreSQL table.
    """
    status = {"success": False, "error": None}
    try:
        engine = get_engine()
        # Automatically creates the table and infer data types
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        status["success"] = True
    except Exception as e:
        status["error"] = str(e)
        
    return status

def execute_safe_query(query: str) -> pd.DataFrame:
    """
    Executes a SELECT query and returns the results as a DataFrame.
    (Detailed safety validation will be handled by the AI module).
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # text() safely wraps the SQL string
            result_df = pd.read_sql(text(query), connection)
        return result_df
    except Exception as e:
        raise Exception(f"Database execution failed: {str(e)}")