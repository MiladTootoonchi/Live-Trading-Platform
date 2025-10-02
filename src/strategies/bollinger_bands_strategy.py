from ..alpaca_trader.order import SideSignal
from typing import Tuple
from config import load_api_keys, make_logger
import requests

logger = make_logger()

def bollinger_bands_strategy(position: dict) -> Tuple[SideSignal, int]:
    symbol = position.get("symbol")
    if not symbol:
        logger.error("No symbol provided in position")
        return SideSignal.HOLD, 0

    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys")
        return SideSignal.HOLD, 0

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?timeframe=1Min&limit=100"
    )

    logger.info(f"Fetching data from: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return SideSignal.HOLD, 0

    data = response.json()
    bars = data.get("bars", [])

    #Bollinger stratgey requires at least 20 bars to generate valid signals
    if len(bars) < 20:
        logger.warning(f"Not enough data for {symbol} - only {len(bars)} bars")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    sma20 = sum(closes[-20:]) / 20
    variance = sum((p - sma20) ** 2 for p in closes[-20:]) / 20
    stddev = variance ** 0.5
    upper_band = sma20 + 2 * stddev
    lower_band = sma20 - 2 * stddev
    current_price = closes[-1]

    logger.info(
        f"[{symbol}] Price: {current_price:.2f}, SMA20: {sma20:.2f}, "
        f"Upper: {upper_band:.2f}, Lower: {lower_band:.2f}"
    )

    qty = int(float(position.get("qty", 0)))

    if current_price < lower_band:
        logger.info(f"[{symbol}] BUY signal - price below lower band")
        return SideSignal.BUY, 0
    elif current_price > upper_band and qty > 0:
        logger.info(f"[{symbol}] SELL signal - price above upper band")
        return SideSignal.SELL, 0
    else:
        logger.info(f"[{symbol}] HOLD - price within bands")
        return SideSignal.HOLD, 0
