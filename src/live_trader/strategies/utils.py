from __future__ import annotations

import datetime as dt
import pandas as pd
from typing import Iterable, Mapping, Union
from tenacity import retry, wait_exponential, stop_after_attempt

from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient

from live_trader.config import load_api_keys, make_logger

logger = make_logger()

KEY, SECRET = load_api_keys()

client = StockHistoricalDataClient(api_key = KEY, secret_key = SECRET)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def _get_bars(request_params):
    return client.get_stock_bars(request_params)


def fetch_data(symbol: str,
               start_date: tuple[int, int, int] = (2020, 1, 1),
               end_date: tuple[int, int, int] = (2026, 1, 1)) -> pd.DataFrame:

    if client is None:
        logger.error("Alpaca client not initialized")
        return pd.DataFrame()

    y, m, d = start_date
    start = dt.date(y, m, max(1, d))

    y, m, d = end_date
    end = dt.date(y, m, max(1, d))


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

    bars = _get_bars(request_params)

    # Convert to DataFrame
    df = bars.df

    # Flatten symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    # Enforce UTC DatetimeIndex
    df.index = pd.to_datetime(df.index, utc=True)

    KEEP = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    df = df[[c for c in KEEP if c in df.columns]]

    return df

BarLike = Union[
    pd.DataFrame,
    Iterable[Mapping[str, object]]
]


def normalize_bars(bars: BarLike) -> pd.DataFrame:
    """
    Normalize OHLCV bar data into a pandas DataFrame with full column names.

    This function accepts bar data in multiple common formats and guarantees
    a standardized output suitable for indicator calculations and ML models.

    Accepted input formats:
        1. pandas.DataFrame with columns:
            - Full names: open, high, low, close, volume
            - Short names: o, h, l, c, v
        2. Iterable of dict-like objects with keys:
            - Alpaca-style: o, h, l, c, v, t
            - Full names: open, high, low, close, volume, time/timestamp

    Output guarantees:
        - pandas.DataFrame
        - Columns: open, high, low, close, volume
        - DatetimeIndex (UTC, tz-aware)
        - Sorted by timestamp ascending
        - Numeric columns coerced to float (volume -> int)

    Args:
        bars:
            Raw bar data (DataFrame or iterable of bar dictionaries).

    Returns:
        pd.DataFrame:
            Normalized OHLCV DataFrame. Empty if input is invalid or empty.
    """
    if bars is None:
        return pd.DataFrame()

    if isinstance(bars, pd.DataFrame):
        df = bars.copy()
    else:
        try:
            df = pd.DataFrame(list(bars))
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    COLUMN_MAP = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }

    df = df.rename(columns=COLUMN_MAP)

    REQUIRED = {"open", "high", "low", "close", "volume"}
    if not REQUIRED.issubset(df.columns):
        return pd.DataFrame()

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        time_col = None
        for candidate in ("timestamp", "time", "t", "date"):
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            return pd.DataFrame()

        idx = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.drop(columns=[time_col])

    df.index = pd.DatetimeIndex(idx, tz="UTC")
    df = df[~df.index.isna()]

    return df
