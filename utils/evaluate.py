import pandas as pd
from sklearn.metrics import mean_squared_error


def get_rmse(forecast_df: pd.DataFrame, dataset: pd.DataFrame) -> float:
    """Helper function to calculate the RMSE of TimeGPT forecast."""
    merged_df = pd.merge(forecast_df, dataset[["Date", "Close"]], on="Date", how="inner")
    rmse = mean_squared_error(merged_df['Close'], merged_df['TimeGPT'])

    return round(rmse, 2)
