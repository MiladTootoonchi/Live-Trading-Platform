from ..alpaca_trader.order import SideSignal
from typing import Tuple
from config import make_logger
from .fetch_price_data import fetch_price_data 

logger = make_logger()


def bollinger_bands_strategy(position: dict) -> Tuple[SideSignal, int]:
    """Generates a buy/sell/hold signal based on Bollinger Bands."""
    symbol = position.get("symbol")
    if not symbol:
        logger.error("No symbol provided in position")
        return SideSignal.HOLD, 0

    bars = position.get("history, []")

    if not bars:
        from .fetch_price_data import fetch_price_data
        bars = fetch_price_data
        
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

    if current_price < lower_band:
        logger.info(f"[{symbol}] BUY signal - price below lower band")
        return SideSignal.BUY, 0
    elif current_price > upper_band:
        logger.info(f"[{symbol}] SELL signal - price above upper band")
        return SideSignal.SELL, 0
    else:
        logger.info(f"[{symbol}] HOLD - price within bands")
        return SideSignal.HOLD, 0


