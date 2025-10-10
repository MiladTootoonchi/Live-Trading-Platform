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
    def __init__(self, symbol: str, days: int = 250, initial_cash: float = 10000, timeframe: str = "1Min"):
        """
        Backtesting engine for a single symbol.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            days (int): Number of historical days to fetch
            initial_cash (float): Starting portfolio value
            timeframe (str): Bar timeframe ('1Min', '5Min', '15Min', '1Day', etc.)

        """
        self.symbol = symbol
        self.days = days
        self.initial_cash = initial_cash
        self.timeframe = timeframe

        #fetch data
        self.bars  = fetch_price_data(symbol, timeframe=timeframe, days=days)

        if not self.bars:
            raise ValueError(f"No data fetched for {symbol} ({timeframe})")
        else: 
            logger.info(f"Fetched {len(self.bars)} bars for {symbol} ({timeframe})")

    def run_strategy(self, strategy_func: Callable) -> pd.DataFrame:
        """ Run a strategy and returns the portofolio developemnt """

        cash = self.initial_cash
        position_qty = 0
        position_avg_price = 0.0
        portofolio_values = []
        dates = []
        trades = []

        for i in range(20, len(self.bars)):
            bar = self.bars[i]
            date = bar["t"][:19]
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

            # Execute trades
            if signal == SideSignal.BUY and qty > 0 and cash >= current_price * qty:
                cost = current_price * qty
                if position_qty > 0:
                    position_avg_price = ((position_avg_price * position_qty) + cost) / (position_qty + qty)
                else:
                    position_avg_price = current_price
                position_qty += qty
                cash -= cost
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'qty': qty,
                    'price': current_price,
                    'cost': cost

                })
                logger.info(f"{date}: BUY {qty} @ ${current_price:.2f}")

            elif signal == SideSignal.SELL and qty > 0 and position_qty >= qty:
                proceeds = current_price * qty
                position_qty -= qty
                cash += proceeds

                if position_qty == 0:
                    position_avg_price = 0.0
                trades.append({
                'date': date,
                'action': 'SELL',
                'qty': qty,
                'price': current_price,
                'proceeds': proceeds
                })
                logger.info(f"{date}: SELL {qty} @ ${current_price:.2f}")

            # Calculate portofoli value
            position_value = position_qty * current_price
            total_value = cash + position_value
            portofolio_values.append(total_value)
            dates.append(date)
        
        results = pd.DataFrame({
            'date': dates,
            'portfolio_value': portofolio_values
        })
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        return results
    
                