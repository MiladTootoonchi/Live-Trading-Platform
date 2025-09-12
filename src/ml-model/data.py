from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime as dt

from config import load_api_keys, make_logger

key, secret = load_api_keys()

client = StockHistoricalDataClient(key, secret)

request_params = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Day,
    start=dt.date(2023, 1, 1),
    end=dt.date(2023, 6, 1)
)

bars = client.get_stock_bars(request_params)

print(bars.df)