import os
import pandas as pd
from nixtla import NixtlaClient

api ="nixak-sMYr8U9pygGX5e4fRuZxZ2P4W0OhfLEyf7fvY5FWx1IQxWq4N6zJienSJKJ85ZBKB8ahryUNeoFaTzMf"

client = NixtlaClient(
    api_key=api,
)

df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')

df.head()