from ..alpaca_trader.order import SideSignal

import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple
from config import load_api_keys, make_logger

logger = make_logger()

def bollinger_bands_strategy(position: dict) -> Tuple[SideSignal, int]:
    symbol = position.get("symbol")
    if not symbol:
        logger.error("No symbol provided in position\n")
        return SideSignal.HOLD, 0

    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys\n")
        return SideSignal.HOLD, 0

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=60)  # Request 60 days for safety buffer

    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.isoformat()}Z"
        f"&end={end_date.isoformat()}Z"
        f"&timeframe=1Day&limit=60"
    )

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}\n")
        return SideSignal.HOLD, 0

    bars = response.json().get("bars", [])
    if len(bars) < 20:
        logger.info(f"Not enough data for {symbol}\n")
        return SideSignal.HOLD, 0

    closes = [bar["c"] for bar in bars]

    sma20 = sum(closes[-20:]) / 20

    # Calculate standard deviation of the last 20 closes
    mean = sma20
    variance = sum((price - mean) ** 2 for price in closes[-20:]) / 20
    stddev = variance ** 0.5

    upper_band = sma20 + (2 * stddev)
    lower_band = sma20 - (2 * stddev)
    current_price = closes[-1]

    logger.info(f"[{symbol}] Price: {current_price:.2f}, SMA20: {sma20:.2f}, Upper: {upper_band:.2f}, Lower: {lower_band:.2f}")

    qty = int(float(position.get("qty", 0)))

    # Strategy logic:
    # Buy when price crosses below the lower Bollinger Band (oversold signal)
    # Sell when price crosses above the upper Bollinger Band (overbought signal)
    if current_price < lower_band:
        return SideSignal.BUY, 1
    elif current_price > upper_band and qty > 0:
        return SideSignal.SELL, qty
    else:
        return SideSignal.HOLD, 0

