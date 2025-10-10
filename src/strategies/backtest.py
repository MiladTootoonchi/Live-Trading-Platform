import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Dict
from config import load_api_keys, make_logger
from ..alpaca_trader.order import SideSignal

logger = make_logger()


def fetch_price_data(symbol: str, timeframe: str = "1Min", days: int = 5) -> List[dict]:
    """
    Fetch historical price data from Alpaca API with pagination support.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        timeframe (str): Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
        days (int): Number of days of data to fetch

    Returns:
        List[dict]: List of bar data
    """
    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing Alpaca API keys.")
        return []

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    base_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    url = (
        f"{base_url}?start={start_time.isoformat()}Z"
        f"&end={end_time.isoformat()}Z"
        f"&timeframe={timeframe}"
        f"&limit=10000"  # max limit per request
    )

    all_bars = []
    next_page_token = None

    logger.info(f"Fetching {timeframe} data for {symbol} from Alpaca")

    while True:
        full_url = url + (f"&page_token={next_page_token}" if next_page_token else "")
        try:
            response = requests.get(full_url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            bars = data.get("bars", [])
            if not bars:
                logger.warning(f"No bars returned for {symbol} on this page.")
                break

            all_bars.extend(bars)
            logger.info(f"Fetched {len(bars)} bars (total so far: {len(all_bars)})")

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        except requests.RequestException as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            break

    logger.info(f"Fetched total of {len(all_bars)} bars for {symbol} ({timeframe})")
    return all_bars

class Backtester:
    def __init__(self, symbol: str, days: int = 250, initial_cash: float = 10000):
        self.symbol = symbol
        self.days = days
        self.initial_cash = initial_cash
        self.bars = fetch_price_data(symbol, days=days)

        if not self.bars:
            raise ValueError(f"No data fetched for {symbol}")
        else: 
            logger.info(f"Fetched {len(self.bars)} bars for {symbol}")

    def run_strategy(self, strategy_func: Callable) -> pd.DataFrame:
        """
        Run a strategy and returns the portofolio developemnt 

        """
        cash = self.initial_cash
        position_qty = 0
        position_avg_price = 0.0


        portofolio_values = []
        dates = []

        for i in range(20, len(self.bars)):
            bar = self.bars[i]
            date = bar["t"][:10]
            current_price = bar["c"]

            position_data = {
                "symbol": self.symbol,
                "qty": position_qty,
                "history": self.bars[:i+1]
            }

            try:
                signal, qty = strategy_func(position_data)
            except Exception as e:
                logger.error(f"Strategy {strategy_func.__name__} failed: {e}")
                signal, qty = SideSignal.HOLD, 0
                