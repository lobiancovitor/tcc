import pandas as pd
import pandas_market_calendars as mcal

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


MAPPING = {
    "Date": "ds",
    "Close": "y",
}

COLUMNS = ["ds", "y"]


@step
def data_preprocessor(dataset: pd.DataFrame) -> pd.DataFrame:
    """Data preprocessor step.

    Args:
        dataset: The dataset to preprocess.

    Returns:
        A pandas DataFrame with the preprocessed data.
    """
    dataset = dataset.rename(columns=MAPPING)
    dataset = dataset[COLUMNS]
    logger.info(f"Dataset with {len(dataset)} records preprocessed!")

    return dataset
