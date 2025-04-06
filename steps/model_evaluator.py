import os
import pandas as pd
from dotenv import load_dotenv

from zenml import step
from zenml.logger import get_logger

from utils import get_rmse

logger = get_logger(__name__)

load_dotenv()
API_KEY = os.getenv("API_KEY")


@step(enable_cache=False)
def model_evaluator(
    dataset: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> float:
    """Model evaluator step."""
    # get the model historical rmse
    rmse = get_rmse(
        forecast_df, dataset
    )
    logger.info(f"Model RMSE: {rmse}")

    return rmse
