from ..alpaca_trader.order import SideSignal
from config import load_api_keys, make_logger
import requests

logger = make_logger()

def fetch_price_data(symbol: str, limit: int = 200):
    """Fetch latest 1-minute bars from Alpaca."""
    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys")
        return []

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Min&limit={limit}"
    logger.info(f"Fetching 1-minute bars from: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        bars = data.get("bars") or []
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        return bars
    except requests.RequestException as e:
        logger.error(f"Error fetching 1-minute bars for {symbol}: {e}")
        return []

def moving_average_strategy(position: dict) -> tuple[SideSignal, int]:
    """
    Moving Average Crossover Strategy.
    Logic:
    - Buy if current price > MA20 > MA50 > MA200
    - Sell if current price < MA20 < MA50 < MA200 and we have qty
    - Hold otherwise
    """
    symbol = position.get("symbol")
    if not symbol:
        logger.error("No symbol provided in position")
        return SideSignal.HOLD, 0

    bars = fetch_price_data(symbol)
    if len(bars) < 200:
        logger.info(f"Not enough bars for {symbol}. Need at least 200 minute bars")
        return SideSignal.HOLD, 0


    closes = [float(bar["c"]) for bar in bars]
    current_price = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200

    logger.info(f"[{symbol}] Price: {current_price:.2f}, MA20: {ma20:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")

    # Strategy logic:
    # Return BUY if price is above MA20, MA50, and MA200 (bullish alignment)
    # Return SELL if price is below MA20, MA50, and MA200 and we hold a position (bearish alignment)
    # Otherwise, return HOLD (i.e., price is within the moving averages and no clear trend)
    if current_price > ma20 > ma50 > ma200:
        logger.info(f"[{symbol}] BUY signal - bullish MA alignment")
        return SideSignal.BUY, 0
    elif current_price < ma20 < ma50 < ma200:
        logger.info(f"[{symbol}] SELL signal - bearish MA alignment")
        return SideSignal.SELL, 0
    else:
        logger.info(f"[{symbol}] HOLD - no clear MA trend")
        return SideSignal.HOLD, 0 
