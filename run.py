from steps import (
    data_loader, 
    data_preprocessor, 
    inference_predict,
    model_evaluator
)

from nixtla import NixtlaClient
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


# this pipeline will be moved to the inference.py

# to run the dashboard: zenml login --local


@pipeline
def inference():
    dataset = data_loader()
    df_inference = data_preprocessor(dataset)
    forecast_df = inference_predict(df_inference)
    model_evaluator(dataset, forecast_df)


if __name__ == "__main__":
    inference()