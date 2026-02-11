from __future__ import annotations

import datetime as dt
import pandas as pd
from typing import Iterable, Mapping, Union
from tenacity import retry, wait_exponential, stop_after_attempt

from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient

from live_trader.config import Config

BarLike = Union[
        pd.DataFrame,
        Iterable[Mapping[str, object]]
    ]

class MarketDataPipeline():
    """
    Market data loader and normalizer for a single symbol.

    Fetches historical OHLCV data from Alpaca and converts
    multiple bar formats into a standardized DataFrame.
    Ensures clean, UTC-indexed data for strategies and ML models.
    """
    def __init__(self, config: Config, symbol: str, position_data: Mapping[str, object] | None = None, lookback = 750):
        """
        Initialize market data pipeline.

        Args:
            config: Application configuration object.
            symbol: Ticker symbol to fetch.
            position_data: Optional runtime position metadata.
            lookback: Number of historical bars to consider.
        """
        self._config = config
        self.symbol = symbol
        self.lookback = lookback

        self._key, self._secret = config.load_keys()

        self._client = StockHistoricalDataClient(api_key = self._key, secret_key = self._secret)

        self._position_data: Mapping[str, object] = position_data or {}
        self._data: pd.DataFrame = self._create_bars()



    @property
    def position_data(self):
        return self._position_data

    @position_data.setter
    def position_data(self, new_position_data):
        self._position_data = new_position_data or {}

    @property
    def data(self):
        return self._data



    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _get_bars(self, request_params):
        """
        Retrieve historical bars from Alpaca with retry logic.

        Automatically retries using exponential backoff
        to handle transient API failures.

        Args:
            request_params: Configured StockBarsRequest object.

        Returns:
            Raw Alpaca bar response.
        """
        return self._client.get_stock_bars(request_params)
    def _fetch_data(
            self,
            start_date: tuple[int, int, int] = (2020, 1, 1),
            end_date: tuple[int, int, int] = (2026, 1, 1)
    ) -> pd.DataFrame:
        """
        Fetch daily historical OHLCV data from Alpaca.

        Validates date boundaries, enforces UTC index,
        flattens multi-index responses, and filters
        to relevant market columns.

        Args:
            start_date: (year, month, day) start tuple.
            end_date: (year, month, day) end tuple.

        Returns:
            pd.DataFrame: Cleaned historical bar dataset.
        """

        if self._client is None:
            self._config.log_error("Alpaca client not initialized")
            return pd.DataFrame()

        y, m, d = start_date
        start = dt.date(y, m, max(1, d))

        y, m, d = end_date
        end = dt.date(y, m, max(1, d))


        today = dt.date.today()
        if end > today:
            end = today
        if start > today:
            self._config.log_error("start_date cannot be in the future.")
            return pd.DataFrame()

        request_params = StockBarsRequest(
            symbol_or_symbols = self.symbol,
            timeframe = TimeFrame.Day,
            start = start,
            end = end
        )

        bars = self._get_bars(request_params)

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
    
    def _normalize_bars(self, bars: BarLike) -> pd.DataFrame:
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

        return df.sort_index()
    
    def _create_bars(self) -> pd.DataFrame:
        """
        Build final bar dataset for the symbol.

        Prefers runtime history from position_data when
        available; otherwise fetches historical data
        from the market data provider.

        Returns:
            pd.DataFrame: Normalized OHLCV dataset.
        """
        bars = self._normalize_bars(self._position_data.get("history"))

        if bars.empty:
            bars = self._normalize_bars(self._fetch_data())

        return bars
