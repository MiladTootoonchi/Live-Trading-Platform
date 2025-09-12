from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime as dt

from config import load_api_keys

key, secret = load_api_keys()

client = StockHistoricalDataClient(key, secret)

def fetch_data(symbol: str, 
             start_date: tuple[int, int, int] = (2023, 1, 1), 
             end_date: tuple[int, int, int] = (2023, 6, 1)):
    """
    Fetching historical stock data for a given symbol using Alpaca's API.

    Args:
        symbol (str): The ticker symbol of the stock to fetch (e.g., 'AAPL').
        start_date (tuple[int, int, int]), optional: The start date as a tuple (year, month, day). Defaults to (2023, 1, 1).
        end_date (tuple[int, int, int]), optional: The end date as a tuple (year, month, day). Defaults to (2023, 6, 1).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical daily stock bars, 
        including open, high, low, close, volume, and timestamp indexed by date.
    """

    # Convert tuples to datetime.date
    start = dt.date(*start_date)
    end = dt.date(*end_date)

    request_params = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe = TimeFrame.Day,
        start = start,
        end = end
    )

    bars = client.get_stock_bars(request_params)

    return bars.df