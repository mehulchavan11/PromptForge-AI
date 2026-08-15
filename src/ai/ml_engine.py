import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

def run_anomaly_detection(df: pd.DataFrame, target_col: str) -> dict:
    """Flags anomalies in a numeric column using Isolation Forest."""
    try:
        data = df[[target_col]].dropna()
        if len(data) < 10:
            return {"error": "Need at least 10 data points for anomaly detection."}
            
        # Standardize features for accurate isolation
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Train Isolation Forest with auto-contamination
        model = IsolationForest(contamination='auto', random_state=42)
        predictions = model.fit_predict(scaled_data)
        
        # Identify anomalies (-1 signifies an anomaly in scikit-learn)
        anomalies = data[predictions == -1]
        
        return {
            "success": True,
            "pipeline": "Isolation Forest Anomaly Detection",
            "target_column": target_col,
            "total_rows_analyzed": len(data),
            "anomalies_found": len(anomalies),
            "anomaly_values": anomalies[target_col].tolist()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_time_series_forecast(df: pd.DataFrame, date_col: str, target_col: str, periods: int = 7) -> dict:
    """Projects future values using Ridge Regression. Autonomously handles both Dates and Numeric Time."""
    try:
        data = df[[date_col, target_col]].dropna().copy()
        
        # 1. Smart Date/Time Check (The 1970 Bug Fix)
        is_numeric_time = pd.api.types.is_numeric_dtype(data[date_col])
        
        if not is_numeric_time:
            data[date_col] = pd.to_datetime(data[date_col])
            
        data = data.sort_values(date_col)
        
        # 2. Convert time/date to a numerical feature for Scikit-Learn
        start_time = data[date_col].min()
        
        if is_numeric_time:
            # If it's an integer like 'hour_of_day', just subtract the numbers
            data['time_step'] = data[date_col] - start_time
        else:
            # If it's a real date, calculate the days difference
            data['time_step'] = (data[date_col] - start_time).dt.days
            
        X = data[['time_step']]
        y = data[target_col]
        
        # 3. Train cross-validated Ridge model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        model.fit(X_scaled, y)
        
        # 4. Forecast future periods
        last_step = data['time_step'].max()
        future_steps = np.array([[last_step + i] for i in range(1, periods + 1)])
        future_steps_scaled = scaler.transform(future_steps)
        forecasts = model.predict(future_steps_scaled)
        
        # 5. Format future dates output appropriately
        if is_numeric_time:
            # Output strings of the future numbers (e.g., hour "22")
            future_dates_formatted = [str(data[date_col].max() + i) for i in range(1, periods + 1)]
        else:
            # Output formatted calendar dates
            future_dates_raw = [data[date_col].max() + pd.Timedelta(days=i) for i in range(1, periods + 1)]
            future_dates_formatted = [d.strftime('%Y-%m-%d') for d in future_dates_raw]
            
        return {
            "success": True,
            "pipeline": "Ridge Regression Forecasting",
            "target_column": target_col,
            "forecast_periods": periods,
            "future_dates": future_dates_formatted,
            "forecast_values": np.round(forecasts, 2).tolist()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}