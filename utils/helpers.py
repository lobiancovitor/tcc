import pandas as pd
import pandas_market_calendars as mcal

def get_custom_business_day(
    dataset: pd.DataFrame, 
    calendar: str = "NYSE"
) -> pd.offsets.CustomBusinessDay:
    """
    For irregular timestamps, TimeGPT requires to specify the frequency of the data directly.
    """
    dates = pd.DatetimeIndex(sorted(dataset['ds'].unique()))
    nyse = mcal.get_calendar(calendar)

    nyse_dates = nyse.valid_days(
        start_date=dates.min(), end_date=dates.max()).tz_localize(None)
    
    weekdays = pd.date_range(start=dates.min(), end=dates.max(), freq='B')

    closed_days = weekdays.difference(nyse_dates)

    custom_bday = pd.offsets.CustomBusinessDay(holidays=closed_days)

    return custom_bday


    