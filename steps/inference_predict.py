import os
import pandas as pd
from dotenv import load_dotenv

from utils import get_custom_business_day

from nixtla import NixtlaClient
from utilsforecast.losses import rmse

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

load_dotenv()
API_KEY = os.getenv("API_KEY")

@step(enable_cache=False)
def inference_predict(
    df_inference: pd.DataFrame,
    api_key: str = API_KEY,
) -> pd.DataFrame:
    """Inference predict step."""
    nixtla_client = NixtlaClient(api_key=api_key)
    
    custom_freq = get_custom_business_day(df_inference)

    predict_df = nixtla_client.forecast(
        df=df_inference,
        h=1,
        freq=custom_freq,
        finetune_steps=50,
        finetune_loss='rmse',
        add_history=True,
    )

    predict_df['ds'] = pd.to_datetime(predict_df['ds'])
    predict_df = predict_df.rename(columns={"ds": "Date"})
    
    return predict_df
