import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple
from .strategy_basics import SideSignal
from config import load_api_keys

def moving_average_strategy(position: dict) -> Tuple[SideSignal, int]:
    """
    Moving Average Crossover Strategy using Alpaca Historical data.
    Logic:
        - Buy if current price > MA20 > MA50 > MA200
        - Sell if current price < MA20 < MA50 < MA200 and we have qty
        - Hold otherwise
    Args:
        position (dict): Position info, expects keys "symbol" and "qty".
    Returns:
        Tuple[SideSignal, int]: Signal and quantity to trade.
    """
    symbol = position.get("symbol")
    if not symbol:
        print("No symbol provided in position.")
        return SideSignal.HOLD, 0

    alpaca_key, alpaca_secret = load_api_keys()

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=250)

    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.isoformat()}Z"
        f"&end={end_date.isoformat()}Z"
        f"&timeframe=1Day&limit=250"
    )

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return SideSignal.HOLD, 0

    bars = response.json().get("bars", [])
    if len(bars) < 200:
        print(f"Not enough data for {symbol}. Requires at least 200 days of data.")
        return SideSignal.HOLD, 0

    closes = [bar["c"] for bar in bars]
    current_price = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200

    print(f"[{symbol}] Price: {current_price:.2f}, MA20: {ma20:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")

    qty = int(float(position.get("qty", 0)))

    if current_price > ma20 > ma50 > ma200:
        print(f"Signal: BUY 1 {symbol}")
        return SideSignal.BUY, 1
    elif current_price < ma20 < ma50 < ma200 and qty > 0:
        print(f"Signal: SELL all {qty} {symbol}")
        return SideSignal.SELL, qty
    else:
        print(f"Signal: HOLD {symbol}")
        return SideSignal.HOLD, 0
