import pandas as pd
import yfinance as yf

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

GOLD_TICKER = "GLD"


@step
def data_loader(
    ticker: str = GOLD_TICKER,
    start_date: str = "2000-01-01",
) -> pd.DataFrame:
    """Dataset reader step.

    Args:
        ticker: The ticker symbol of the asset to load.
        start_date: The start date of the data to load.
    Returns:
        A pandas DataFrame containing the historical data.
    """
    dataset = yf.download(ticker, start=start_date)

    if dataset.empty:
        raise ValueError(f"No data found for ticker {ticker}.")

    logger.info(f"Dataset with {len(dataset)} records loaded!")
    return dataset
