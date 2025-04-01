from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor

from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


# this pipeline will be moved to the inference.py

# to run the dashboard: zenml login --local


@pipeline
def inference():
    dataset = data_loader()
    data_preprocessor(dataset)

if __name__ == "__main__":
    inference()