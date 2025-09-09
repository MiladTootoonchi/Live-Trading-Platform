import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from config import load_api_keys, make_logger

logger = make_logger()

def fetch_price_data(symbol: str, days: int = 250) -> List[Dict]:
    alpaca_key, alpaca_secret = load_api_keys()
    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.isoformat()}Z"
        f"&end={end_date.isoformat()}Z"
        f"&timeframe=1Day&limit={days}"
    )
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        bars = response.json().get("bars", [])
        return bars
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}\n")
        return []

def backtest_strategy(strategy_func, symbol: str, initial_cash: float = 10000):
    bars = fetch_price_data(symbol)
    if not bars:
        print("No data fetched for backtest")
        return
    
    cash = initial_cash
    position_qty = 0
    position_avg_price = 0.0

    portfolio_values = []
    dates = []

    for i in range(len(bars)):
        bar = bars[i]
        date = bar['t'][:10]
        current_price = bar['c']

        position_data = {
            "symbol": symbol,
            "qty": position_qty
        }
