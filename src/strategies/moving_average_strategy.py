from ..alpaca_trader.order import SideSignal
from config import make_logger
from .fetch_price_data import fetch_data 

logger = make_logger()


def moving_average_strategy(position: dict) -> tuple[SideSignal, int]:
    """
    Moving Average Crossover Strategy.
    Logic:
    - Buy if current price > MA20 > MA50 > MA200
    - Sell if current price < MA20 < MA50 < MA200
    - Hold otherwise
    """
    symbol = position.get("symbol")
    if not symbol:
        logger.error("No symbol provided in position")
        return SideSignal.HOLD, 0
    
    bars = position.get("history", [])

    if not bars:
        logger.warning(f"[MA Strategy] No history in position data for {symbol}, fetching from API")
        bars = fetch_data(symbol, limit=200)

    if not bars or len(bars) < 200:
        logger.info(f"Not enough bars for {symbol}. Need at least 200 minute bars")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    current_price = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200

    logger.info(f"[{symbol}] Price: {current_price:.2f}, MA20: {ma20:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")

    # Buy signal - bullish MA alignment
    if current_price > ma20 > ma50 > ma200:
        logger.info(f"[{symbol}] BUY signal - bullish MA alignment")
        return SideSignal.BUY, 0

    # Sell signal - bearish MA alignment
    elif current_price < ma20 < ma50 < ma200:
        logger.info(f"[{symbol}] SELL signal - bearish MA alignment")
        return SideSignal.SELL, 0

    # Otherwise hold
    else:
        logger.info(f"[{symbol}] HOLD - no clear MA trend")
        return SideSignal.HOLD, 0
