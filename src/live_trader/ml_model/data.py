from alpaca.data.requests import StockLatestTradeRequest
import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, List

from live_trader.strategies.data import MarketDataPipeline
from live_trader.config import Config

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.preprocessing import StandardScaler


class MLDataPipeline(MarketDataPipeline):
    def __init__(
                self,
                config: Config,
                symbol: str,
                position_data: dict,

                feature_columns: List[str] = [
                    "open", "high", "low", "close_z", "volume_z", "trade_count",
                    "vwap", "SMA5", "SMA20", "SMA50",
                    "price_change", "RSI", "MACD", "MACD_Signal"],
                ):
        
        super().__init__(config, symbol, position_data)

        self._feature_columns = feature_columns

        self._sma_windows = config.load_sma_windows()
        self._rsi_window = config.load_ml_var("rsi_window")

        self._macd_fast = config.load_ml_var("macd_fast")
        self._macd_slow = config.load_ml_var("macd_slow")
        self._macd_signal = config.load_ml_var("macd_signal")
        self._macd_stabilization = config._macd_stabilization

        self._zscore_window = config.load_ml_var("zscore_window")

        self._time_steps = config.load_ml_var("time_steps")

        self._safety_margin = max(10, self._time_steps // 2)

        self._min_lookback = config.load_min_lookback()

        self._pred_history = self._time_steps + self._min_lookback + self._safety_margin
        self._ml_training_lookback = config.load_ml_var("ml_training_lookback")
        self._is_backtest = self._position_data.get("backtest", False)
        self._data = self._create_df()
        self._pred_df = self._build_prediction_dataframe()

    @property
    def pred_history(self):
        return self._pred_history
    
    @property
    def time_steps(self):
        return self._time_steps

    @property
    def is_backtest(self):
        return self._is_backtest
    
    @property
    def data(self):
        return self._data
    
    @property
    def pred_df(self):
        return self._pred_df


    def load_slice(self, end_idx: int):
        super().load_slice(end_idx)
        self._pred_df = self._build_prediction_dataframe()



    def _get_one_realtime_bar(self, last_close: float) -> pd.DataFrame:
        """
        Retrieves the most recent real-time trade for a given stock symbol using
        Alpaca's REST API and constructs a synthetic OHLCV bar that preserves
        price continuity with historical data.

        This function replaces the slower WebSocket streaming approach by using
        the 'LatestTrade' REST endpoint, enabling near-instant response times
        (~30 - 120 ms). Because real-time data is based on a single trade snapshot,
        the bar is constructed using the previous historical close to avoid
        zero-range candles and feature distribution shifts.

        The resulting synthetic bar follows these rules:

            - open:  previous historical close
            - high:  max(previous close, latest trade price)
            - low:   min(previous close, latest trade price)
            - close: latest trade price
            - volume: reported trade size (or fallback if unavailable)
            - trade_count: 1
            - vwap: equal to the latest trade price

        This structure ensures consistency with historical OHLCV bars and keeps
        technical indicators (RSI, MACD, SMA) well-behaved at inference time.

        Args:
            last_close (float):
                The most recent historical closing price for the symbol.
                This value is used to construct a realistic OHLC range for
                the synthetic realtime bar.

        Returns:
            pd.DataFrame:
                A single-row DataFrame containing the following columns:

                    ['timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'trade_count', 'vwap']

                If no realtime trade is available, a safe fallback bar is returned
                using the provided `last_close` and zero volume.
        """

        trade_req = StockLatestTradeRequest(symbol_or_symbols=self._symbol)

        try:
            latest_trade = self._client.get_latest_trade(trade_req)
        except Exception:
            return pd.DataFrame([{
                "timestamp": pd.Timestamp.utcnow(),
                "open": last_close,
                "high": last_close,
                "low": last_close,
                "close": last_close,
                "volume": 0,
                "trade_count": 0,
                "vwap": last_close,
            }])

        price = float(latest_trade.price)
        size = float(latest_trade.size or 1)

        bar = {
            "timestamp": latest_trade.timestamp,
            "open": last_close,
            "high": max(last_close, price),
            "low": min(last_close, price),
            "close": price,
            "volume": size,
            "trade_count": 1,
            "vwap": price,
        }

        return pd.DataFrame([bar])



    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators consistently for both
        training and inference.

        Args:
            df (pd.DataFrame): OHLCV dataframe indexed by timestamp.

        Returns:
            pd.DataFrame: Feature-enriched dataframe.
        """
        df = df.copy()

        # Moving averages
        for window in self._sma_windows:
            df[f"SMA{window}"] = df["close"].rolling(window).mean()

        # Price change
        df["price_change"] = df["close"].diff()

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(self._rsi_window).mean()
        avg_loss = loss.rolling(self._rsi_window).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df["close"].ewm(span=self._macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self._macd_slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=self._macd_signal, adjust=False).mean()

        # Z-score normalization (past-only)
        df["close_z"] = self._rolling_zscore(df["close"], window = self._zscore_window)
        df["volume_z"] = self._rolling_zscore(df["volume"], window = self._zscore_window)

        return df
    
    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """
        Compute a rolling z-score using only past information.

        Args:
            series (pd.Series): Input time series.
            window (int): Rolling window length.

        Returns:
            pd.Series: Z-scored series with no lookahead bias.
        """
        mean = series.rolling(window).mean().shift(1)
        std = series.rolling(window).std().shift(1)
        return (series - mean) / std



    @staticmethod
    def _create_target(df: pd.DataFrame) -> pd.Series:
        """
        Create a binary classification target with no leakage.

        Target = 1 if next close is higher than current close.

        Args:
            df (pd.DataFrame): Feature dataframe.

        Returns:
            pd.Series: Binary target aligned with features.
        """
        return (df["close"].shift(-1) > df["close"]).astype(int)


    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create rolling time-window sequences.

        For N = time_steps:
        - Training: multiple overlapping sequences
        - Inference: exactly ONE valid sequence

        Args:
            X: (N, F) feature matrix
            y: (N,) targets (dummy allowed for inference)
            time_steps: sequence length

        Returns:
            X_seq: (N - T + 1, T, F)
            y_seq: (N - T + 1,)
        """
        Xs, ys = [], []

        for i in range(len(X) - time_steps + 1):
            Xs.append(X[i : i + time_steps])
            ys.append(y[i + time_steps - 1])

        return np.asarray(Xs), np.asarray(ys)

    def sequence_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_steps: int = 50,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert features into time sequences and split chronologically.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            time_steps (int): Sequence length.
            train_ratio (float): Fraction used for training.
            val_ratio (float): Fraction used for validation.

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """

        X_seq, y_seq = self.create_sequences(X, y, time_steps)

        n = len(X_seq)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]

        X_val = X_seq[train_end:val_end]
        y_val = y_seq[train_end:val_end]

        X_test = X_seq[val_end:]
        y_test = y_seq[val_end:]

        return X_train, X_val, X_test, y_train, y_val, y_test
    


    @staticmethod
    def _ensure_clean_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            def unwrap(x):
                if isinstance(x, tuple):
                    return x[-1]
                return x

            df["timestamp"] = df["timestamp"].apply(unwrap)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.set_index("timestamp")

        elif isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(-1)

        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[~df.index.isna()]

        if df.empty:
            raise RuntimeError("No valid timestamps after normalization")

        return df
    
    @staticmethod
    def _sanitize_time_index(df: pd.DataFrame, context: str) -> pd.DataFrame:
        """
        Validates and normalizes a DataFrame's DatetimeIndex.

        This function ensures that the DataFrame index is a valid
        'pandas.DatetimeIndex', localizes naive timestamps to UTC,
        removes rows with invalid (NaT) timestamps, and sorts the
        DataFrame by time.

        If the index is not a DatetimeIndex or if the DataFrame becomes
        empty after cleanup, a RuntimeError is raised. Any dropped
        timestamps are logged with contextual information to aid debugging.

        Args:
            df (pd.DataFrame):
                The input DataFrame whose index is expected to represent time.
            context (str):
                A descriptive label used in log messages and exception text
                to identify the caller or data source.

        Returns:
            pd.DataFrame:
                A cleaned DataFrame with a timezone-aware UTC DatetimeIndex
                and sorted in ascending time order.
        """

        if not isinstance(df.index, pd.DatetimeIndex):
            raise RuntimeError(f"{context}: index is not DatetimeIndex")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        bad = df.index.isna()
        if bad.any():
            df = df.loc[~bad]

        if df.empty:
            raise RuntimeError(f"{context}: dataframe empty after timestamp cleanup")

        return df.sort_index()
    

    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Prepare training features, targets, and fitted scaler.

        Args:
            df (pd.DataFrame): Raw OHLCV dataframe.

        Returns:
            Tuple[np.ndarray, np.ndarray, StandardScaler]:
                X: Scaled feature matrix
                y: Target vector
                scaler: Fitted StandardScaler
        """
        df = self._compute_features(df)
        df["target"] = self._create_target(df)
        df = df.dropna()

        X_raw = df[self._feature_columns]
        y = df["target"]

        scaler = StandardScaler()
        scaler.set_output(transform = "pandas")
        X = scaler.fit_transform(X_raw)

        scaler.feature_columns = self._feature_columns

        return X, y, scaler



    def prepare_prediction_data(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler,
    ) -> np.ndarray:
        """
        Prepare features for inference using a pre-fitted scaler.

        Args:
            df (pd.DataFrame): Raw OHLCV dataframe (historical + realtime).
            scaler (StandardScaler): Fitted scaler from training.

        Returns:
            np.ndarray: Scaled feature matrix.
        """
        df = self._compute_features(df)
        df = df.dropna()

        X_raw = df[scaler.feature_columns]
        X = scaler.transform(X_raw.values)

        return np.asarray(X)

    

    def _build_prediction_dataframe(self) -> None:
        """
        Fetches and prepares historical and realtime market data for prediction.

        Combines historical bars from `pred_start_date` to now with the latest
        realtime bar, normalizes timestamps, and ensures required raw columns exist.
        """

        if self._is_backtest:
            hist = pd.DataFrame(self._position_data["history"])
            hist["timestamp"] = pd.to_datetime(hist["t"], utc=True)
            hist = hist.rename(columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }).set_index("timestamp")

            pred_df = hist.tail(self._pred_history).copy()

            for col in ["vwap", "trade_count"]:
                if col not in pred_df.columns:
                    pred_df[col] = 0.0

            self._pred_df = pred_df
            return pred_df


        hist_df = self._sanitize_time_index(self._data, "PREDICTION DATA")
        hist_df = hist_df.reset_index()     # ensure no double index

        if hist_df.empty:
            raise RuntimeError("Prediction history empty after timestamp normalization")

        last_close = hist_df.iloc[-1]["close"]
        realtime_bar = self._get_one_realtime_bar(
            last_close=last_close,
        )

        # Drop fully-NA rows/columns
        realtime_bar = realtime_bar.dropna(how="all")
        realtime_bar = realtime_bar.dropna(axis=1, how="all")

        pred_df = pd.concat(
            [hist_df.tail(self._pred_history + 1), realtime_bar],
            ignore_index=True,
        ).copy()

        pred_df = self._ensure_clean_timestamp(pred_df)

        for col in ["vwap", "trade_count"]:
            if col not in pred_df.columns:
                pred_df.loc[:, col] = 0.0

        self._pred_df = pred_df
        return pred_df
    

    def _create_df(self):
        if self.is_backtest:
            # last available bar timestamp
            current_ts = pd.to_datetime(self._position_data["history"][-1]["t"], utc=True)
        else:
            current_ts = dt.datetime.now(dt.UTC)

        # pred_start_date must be more than 50 days in the past
        pred_start_date = current_ts - dt.timedelta(days = self._pred_history)

        start_dt = pred_start_date - dt.timedelta(days = self._ml_training_lookback)
        start_date = (start_dt.year, start_dt.month, start_dt.day)

        end_dt = pred_start_date - dt.timedelta(days = 1)
        end_date = (end_dt.year, end_dt.month, end_dt.day)

        # fetching data from as early as possible till pred_start_date (pred_start_date till today will be used for prediction)
        df = self._fetch_data(
                        start_date = start_date, 
                        end_date = end_date
                        )
        df = self._ensure_clean_timestamp(df)
        
        self._data = self._sanitize_time_index(df, "TRAINING DATA")

        return self._data