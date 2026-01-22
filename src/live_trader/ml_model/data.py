from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
import pandas as pd
import numpy as np
from typing import Tuple, List

from live_trader.config import load_api_keys, make_logger

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.preprocessing import StandardScaler

# Settings
logger = make_logger()
KEY, SECRET = load_api_keys()

BASE_URL = "https://paper-api.alpaca.markets"

FEATURE_COLUMNS: List[str] = [
    "open", "high", "low", "close_z", "volume_z", "trade_count",
    "vwap", "SMA5", "SMA20", "SMA50",
    "price_change", "RSI", "MACD", "MACD_Signal",
]

SMA_WINDOWS = [5, 20, 50]
RSI_WINDOW = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_STABILIZATION = MACD_SLOW * 3

ZSCORE_WINDOW = 100

TIME_STEPS = 50

SAFETY_MARGIN = max(10, TIME_STEPS // 2)

MIN_LOOKBACK = max(
    max(SMA_WINDOWS),
    RSI_WINDOW,
    MACD_STABILIZATION,
    ZSCORE_WINDOW,
)


async def get_one_realtime_bar(symbol: str, last_close: float) -> pd.DataFrame:
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
        symbol (str):
            Ticker symbol to query (e.g., "AAPL").

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

    client = StockHistoricalDataClient(KEY, SECRET)

    trade_req = StockLatestTradeRequest(symbol_or_symbols=symbol)

    try:
        latest_trade = client.get_latest_trade(trade_req)
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



def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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
    for window in SMA_WINDOWS:
        df[f"SMA{window}"] = df["close"].rolling(window).mean()

    # Price change
    df["price_change"] = df["close"].diff()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    # Z-score normalization (past-only)
    df["close_z"] = rolling_zscore(df["close"], window = ZSCORE_WINDOW)
    df["volume_z"] = rolling_zscore(df["volume"], window = ZSCORE_WINDOW)

    return df



def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Create a binary classification target with no leakage.

    Target = 1 if next close is higher than current close.

    Args:
        df (pd.DataFrame): Feature dataframe.

    Returns:
        pd.Series: Binary target aligned with features.
    """
    return (df["close"].shift(-1) > df["close"]).astype(int)



def prepare_training_data(
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
    df = compute_features(df)
    df["target"] = create_target(df)
    df = df.dropna()

    X_raw = df[FEATURE_COLUMNS]
    y = df["target"]

    scaler = StandardScaler()
    scaler.set_output(transform = "pandas")
    X = scaler.fit_transform(X_raw)

    scaler.feature_columns = FEATURE_COLUMNS

    return X, y, scaler


def prepare_prediction_data(
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
    df = compute_features(df)
    df = df.dropna()

    X_raw = df[scaler.feature_columns]
    X = scaler.transform(X_raw.values)

    return np.asarray(X)



def create_sequences(
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



def compute_trade_qty(position_data: dict, prob: float) -> int:
    """
    Calculates an intelligent stock quantity to trade based on model confidence and risk management.

    The function automatically retrieves your Alpaca account equity, then uses a hybrid 
    risk model that combines confidence scaling and fixed risk-per-trade rules. This ensures 
    trades are dynamically sized while respecting account-level risk limits.

    Args:
        position_data (dict): Alpaca position data containing 'symbol' and price info.
        prob (float): Model probability (0.0 - 1.0) that the trade prediction is correct.

    Returns:
        int: Recommended quantity of shares to buy or sell.
    """

    try:
        client = TradingClient(KEY, SECRET, paper=True)
        account = client.get_account()
        equity = float(account.equity)
    except Exception as e:
        logger.info(f"Failed to fetch account equity: {e}")
        equity = 10000.0  # fallback default for safety

    # Risk parameters
    confidence_threshold = 0.55
    max_position_frac = 0.10        # Max 10% of total equity
    risk_per_trade = 0.01           # Risk 1% of equity per trade
    stop_pct = 0.02                 # 2% stop loss assumption

    try:
        price = float(position_data.get("avg_entry_price") or position_data.get("market_price"))
    except Exception:
        logger.info("Invalid price data in position_data.")
        return 0

    # Confidence-based scaling
    confidence_scale = max(0.0, (prob - confidence_threshold) / (1 - confidence_threshold))

    # Hybrid risk model
    max_position_value = equity * max_position_frac
    risk_dollars = equity * risk_per_trade

    hybrid_qty = (max_position_value * confidence_scale) / price
    risk_qty = risk_dollars / (price * stop_pct)

    qty = int(min(hybrid_qty, risk_qty))

    # If model wants to SELL (negative qty), cap by position size
    try:
        position_qty = int(float(position_data.get("qty", 0)))

        if qty < 0:
            qty = max(qty, -position_qty)
    except Exception:
        logger.info("Invalid position quantity data.\n")
        return 0

    return qty


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
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


def ensure_clean_timestamp(df: pd.DataFrame) -> pd.DataFrame:
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
