import datetime as dt
import pandas as pd
from typing import Dict, Any, List, Union
import pandas as pd

from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient

from config import load_api_keys, make_logger

logger = make_logger()

KEY, SECRET = load_api_keys()

client = StockHistoricalDataClient(api_key = KEY, secret_key = SECRET)

def fetch_data(symbol: str,
               start_date: tuple[int, int, int] = (2020, 1, 1),
               end_date: tuple[int, int, int] = (2025, 1, 1),
               limit: int | None = None) -> pd.DataFrame:

    if client is None:
        logger.error("Alpaca client not initialized")
        return pd.DataFrame()

    start = dt.date(*start_date)
    end = dt.date(*end_date)

    today = dt.date.today()
    if end > today:
        end = today
    if start > today:
        logger.error("start_date cannot be in the future.")
        return pd.DataFrame()

    request_params = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe = TimeFrame.Day,
        start = start,
        end = end
    )

    bars = client.get_stock_bars(request_params)

    # Convert to DataFrame
    df = bars.df

    if df is None or df.empty:
        return pd.DataFrame()

    # Apply manual limit
    if limit is not None:
        df = df.sort_index().tail(limit)

    return df


def normalize_bars(bars: Union[pd.DataFrame, List[Any]]) -> List[Dict[str, float]]:
    """
    Convert bar data into a consistent format: [{"c": close_price}, ...]

    Args:
        bars (Union[pd.DataFrame, List[Any]]):
            Bar data in various formats:
            - DataFrame with 'c' or 'close'
            - List of dicts
            - List of floats / ints / strings

    Returns:
        List[Dict[str, float]]:
            Normalized close data.
    """
    if bars is None:
        return []

    # If bar data is a DataFrame
    if isinstance(bars, pd.DataFrame):
        if "c" in bars.columns:
            return bars[["c"]].to_dict("records")
        if "close" in bars.columns:
            return bars.rename(columns={"close": "c"})[["c"]].to_dict("records")

        logger.error("DataFrame is missing 'c' or 'close' columns.")
        return []

    # List-based bar data
    if isinstance(bars, list) and len(bars) > 0:
        first = bars[0]

        if isinstance(first, dict):
            if "c" in first:
                return bars
            if "close" in first:
                return [{"c": float(x["close"])} for x in bars]

        if isinstance(first, (float, int, str)):
            try:
                return [{"c": float(x)} for x in bars]
            except ValueError:
                logger.error("Bar list includes non-numeric values.")
                return []

    logger.error("Unrecognized bar format.")
    return []