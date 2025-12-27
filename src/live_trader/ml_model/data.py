from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockLatestQuoteRequest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import load_api_keys, make_logger


# Settings
logger = make_logger()
KEY, SECRET = load_api_keys()

client = StockHistoricalDataClient(api_key = KEY, secret_key = SECRET)

BASE_URL = "https://paper-api.alpaca.markets"



async def get_one_realtime_bar(symbol: str) -> pd.DataFrame:
    """
    Retrieves the most recent real-time trade (and optional quote) for a given stock
    symbol using Alpaca's REST API and constructs a synthetic OHLCV bar.

    This function replaces the slower WebSocket streaming approach by using the
    'LatestTrade' and 'LatestQuote' REST endpoints, enabling near-instant response times
    (~30 - 120 ms). Because real-time data is based on a single trade snapshot, the bar
    represents a minimal OHLCV structure with:

        - open:  latest trade price
        - high:  latest trade price
        - low:   latest trade price
        - close: latest trade price
        - volume: reported trade size (or fallback if unavailable)
        - trade_count: 1
        - vwap: equal to the trade price

    The resulting DataFrame is compatible with your existing feature-engineering and
    prediction pipelines, which expect historical-style rows with OHLCV fields.

    Args:
        symbol (str):
            Ticker symbol to query (e.g., "AAPL").

    Returns:
        pd.DataFrame:
            A single-row DataFrame containing:
                ['timestamp', 'open', 'high', 'low', 'close',
                 'volume', 'trade_count', 'vwap']

            If no trade is available, returns a row of safe fallback values.
    """

    client = StockHistoricalDataClient(KEY, SECRET)

    # Fetch latest trade
    trade_req = StockLatestTradeRequest(symbol_or_symbols = symbol)
    quote_req = StockLatestQuoteRequest(symbol_or_symbols = symbol)

    try:
        latest_trade = client.get_latest_trade(trade_req)

    except Exception:
        # Fallback empty bar
        now = pd.Timestamp.utcnow()
        return pd.DataFrame([{
            "timestamp": now,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": 0,
            "trade_count": 0,
            "vwap": None
        }])

    # Fetch latest quote for context (more stable than trades alone)
    try:
        latest_quote = client.get_latest_quote(quote_req)
        bid = latest_quote.bid_price or None
        ask = latest_quote.ask_price or None
    except:
        bid = None
        ask = None

    price = float(latest_trade.price)
    size = float(latest_trade.size or 1)

    # Build synthetic OHLCV
    bar = {
        "timestamp": latest_trade.timestamp,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": size,
        "trade_count": 1,
        "vwap": price
    }

    # Ensure consistency — replace None with NaN
    df = df.replace({None: np.nan})

    # If the bar is all NaN (API hiccup), create a safe fallback
    if df[["open", "high", "low", "close"]].isna().all(axis=None):
        logger.warning(f"Realtime bar empty for {symbol}, applying fallback bar.")
        df["open"] = df["open"].fillna(0)
        df["high"] = df["high"].fillna(0)
        df["low"] = df["low"].fillna(0)
        df["close"] = df["close"].fillna(0)
        df["vwap"] = df["close"]
        df["volume"] = df["volume"].fillna(0)
        df["trade_count"] = df["trade_count"].fillna(0)

    return pd.DataFrame([bar])



def stock_data_feature_engineering(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """
    Perform feature engineering and preprocessing for stock price prediction.

    This function prepares a stock market DataFrame for LSTM or other machine learning models
    by adding technical indicators, computing price changes, creating a target variable, 
    and scaling the features.

    Steps:
    1. Reset index and ensure 'timestamp' is a datetime index.
    2. Add technical indicators:
        - SMA5, SMA20, SMA50: Simple Moving Averages
        - Price change: difference of closing prices
        - RSI: Relative Strength Index (14-day window)
        - MACD: Moving Average Convergence Divergence
        - MACD_Signal: 9-day EMA of MACD
    3. Drop rows with NaN values (resulting from rolling/EMA calculations)
    4. Create binary target: 1 if next day's close is higher than current day, else 0
    5. Scale all features using StandardScaler (mean=0, std=1)
    
    Args:
        df : pd.DataFrame
            DataFrame containing stock data with at least the following columns:
            ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']

    Returns:
        X_scaled : np.ndarray
            Standardized feature matrix including technical indicators.
        y : pd.Series
            Binary target variable indicating whether the next day's close is higher (1) or not (0).

    Notes:
        - Ensure 'timestamp' is present in the DataFrame.
        - The function drops initial rows affected by rolling calculations.
        - Features are scaled across all numeric columns, including the target column.
        Make sure to handle the target appropriately if needed (e.g., exclude from scaling if required).
    """

    df.reset_index(inplace = True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace = True)

    # Adding more feautures, SMA
    df['SMA5']  = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    # Adding price change
    df['price_change'] = df['close'].diff()

    # Calculating and adding RSI
    window = 14
    delta = df['close'].diff()

    gain = delta.clip(lower = 0)
    loss = -delta.clip(upper = 0)

    avg_gain = gain.rolling(window = window).mean()
    avg_loss = loss.rolling(window = window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Short-term EMA (12), long-term EMA (26)
    ema12 = df['close'].ewm(span = 12, adjust = False).mean()
    ema26 = df['close'].ewm(span = 26, adjust = False).mean()

    # MACD line
    df['MACD'] = ema12 - ema26

    # Signal line (9-day EMA of MACD)
    df['MACD_Signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()

    df = df.dropna()

    # Addind a target
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df[:-1]  # Remove last row with NaN target

    # Splitting data
    features = ['open', 'high', 'low', 'close', 'volume', 'trade_count',
                'vwap', 'SMA5', 'SMA20', 'SMA50', 'price_change', 'RSI', 'MACD', 'MACD_Signal']

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df['target'].loc[X.index]  # align target

    return X_scaled, y, scaler


def stock_data_prediction_pipeline(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Prepare new stock data for prediction by applying the same feature engineering 
    as during training, without fitting the scaler or creating targets.

    Args:
        df : pd.DataFrame
            New stock data with required columns.
        scaler : StandardScaler
            Pre-fitted scaler from the training phase.

    Returns:
        X_scaled : np.ndarray
            Scaled feature matrix ready for model.predict().
    """

    df = df.copy()

    df = df.reset_index()

    if 'timestamp' not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': 'timestamp'}, inplace=True)
        else:
            raise KeyError("No 'timestamp' column found in prediction DataFrame.")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # SMA
    df['SMA5']  = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    # Price change
    df['price_change'] = df['close'].diff()

    # RSI (14-day)
    window = 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df = df.dropna()

    features = [
        'open', 'high', 'low', 'close', 'volume', 'trade_count',
        'vwap', 'SMA5', 'SMA20', 'SMA50', 'price_change', 'RSI', 'MACD', 'MACD_Signal'
    ]
    X = df[features]

    X_scaled = scaler.transform(X)

    return X_scaled
