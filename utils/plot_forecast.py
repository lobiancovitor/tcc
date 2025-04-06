import os
import pandas as pd
from dotenv import load_dotenv

from nixtla import NixtlaClient

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


load_dotenv()
API_KEY = os.getenv("API_KEY")


@step
def visualize_predict(
    dataset: pd.DataFrame,
    forecast_df: pd.DataFrame,
    api_key: str = API_KEY,
    output_path: str = "plots",
) -> None:
    """Visualize predict step.
    
    Args:
        dataset: The dataset containing historical data.
        forecast_df: The forecast dataframe.
        api_key: The Nixtla API key.
        output_path: The folder path to save the plot.
    """
    import matplotlib.pyplot as plt
    import os
    
    nixtla_client = NixtlaClient(api_key=api_key)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate the plot
    fig = nixtla_client.plot(
        dataset,
        forecast_df,
        max_insample_length=50,
    )
    
    # Save the plot to the specified path
    filename = os.path.join(output_path, "forecast_plot.png")
    fig.savefig(filename)
    plt.close(fig)
    
    logger.info(f"Plot saved to {filename}")

