from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import asyncio

from config import load_api_keys, make_logger

logger = make_logger()
KEY, SECRET = load_api_keys()

client = StockHistoricalDataClient(api_key = KEY, secret_key = SECRET)

BASE_URL = "https://paper-api.alpaca.markets"

async def get_one_realtime_bar(symbol: str, num_trades: int = 20) -> pd.DataFrame:
    """
    Streams real-time trades for a stock symbol and computes a proper OHLC bar
    after receiving a fixed number of trades.

    Args:
        symbol (str): Stock ticker symbol (e.g., "MSFT").
        num_trades (int): Number of trades to collect before computing OHLC.

    Returns:
        pd.DataFrame: Single-row DataFrame with columns ['open','high','low','close'].
    """

    df_bar = pd.DataFrame(columns = ['open', 'high', 'low', 'close'])
    live_bar = {"open": None, "high": None, "low": None, "close": None}
    trade_count = 0
    stop_event = asyncio.Event()

    async def trade_callback(data):
        nonlocal trade_count
        price = data.price

        # Update OHLC
        live_bar["open"] = live_bar["open"] or price
        live_bar["high"] = max(live_bar["high"] or price, price)
        live_bar["low"] = min(live_bar["low"] or price, price)
        live_bar["close"] = price

        trade_count += 1
        if trade_count >= num_trades:
            stop_event.set()  # stop after enough trades


    stream = StockDataStream(api_key = KEY, secret_key = SECRET)
    stream.subscribe_trades(trade_callback, symbol)

    task = asyncio.create_task(stream._run_forever())
    await stop_event.wait()  # wait until desired number of trades
    task.cancel()
    await stream.close()

    df_bar = pd.DataFrame([live_bar])
    return df_bar



def fetch_data(symbol: str, 
               start_date: tuple[int, int, int] = (2020, 1, 1), 
               end_date: tuple[int, int, int] = (2024, 12, 31)):
    """
    Fetching historical stock data for a given symbol using Alpaca's API.

    Args:
        symbol (str): The ticker symbol of the stock to fetch (e.g., 'AAPL').
        start_date (tuple[int, int, int]), optional: The start date as a tuple (year, month, day). Defaults to (2020, 1, 1).
        end_date (tuple[int, int, int]), optional: The end date as a tuple (year, month, day). Defaults to (2024, 12, 31).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical daily stock bars, 
        including open, high, low, close, volume, and timestamp indexed by date.
    """

    if client is None:
        logger.log("Alpaca client must be provided")

    # Convert tuples to datetime.date
    start = dt.date(*start_date)
    end = dt.date(*end_date)


    # Make sure dates are not in the future
    today = dt.date.today()
    if end > today:
        logger.log(f"Warning: end_date {end} is in the future. Setting it to today.")
        end = today
    if start > today:
        print(f"Error: start_date {start} is in the future. Cannot fetch data.")
        return pd.DataFrame() 


    request_params = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe = TimeFrame.Day,
        start = start,
        end = end
    )

    bars = client.get_stock_bars(request_params)

    return bars.df



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
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Splitting data
    features = ['open', 'high', 'low', 'close', 'volume', 'trade_count',
                'vwap', 'SMA5', 'SMA20', 'SMA50', 'price_change', 'RSI', 'MACD', 'MACD_Signal']

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df['target'].loc[X.index]  # align target

    return X_scaled, y