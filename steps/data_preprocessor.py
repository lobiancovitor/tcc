import pandas as pd
from zenml import step


@step
def data_preprocessor(dataset: pd.DataFrame) -> pd.DataFrame:
    """Data preprocessor step.

    Args:
        dataset: The dataset to preprocess.

    Returns:
        A pandas DataFrame with the preprocessed data.
    """
    print(dataset.head())

    return dataset
