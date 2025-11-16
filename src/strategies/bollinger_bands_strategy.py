from ..alpaca_trader.order import SideSignal
from typing import Tuple, Dict, Any, List, Union
from config import make_logger
from .utils import fetch_data, normalize_bars

logger = make_logger()


def bollinger_bands_strategy(position: Dict[str, Any]) -> Tuple[SideSignal, int]:
    """
    Bollinger Bands mean reversion strategy.
    Generates a signal when price moves above or below
    the standard deviation bands around a moving average.

    Args:
        position (dict):
            Contains:
                symbol (str): the ticker symbol
                history (list or DataFrame): optional price history
                current_price (float): fallback price

    Returns:
        tuple(SideSignal, int):
            The trade signal and quantity (always zero for safety).
    """
    symbol = position.get("symbol")
    if not symbol:
        logger.error("Bollinger strategy missing symbol.")
        return SideSignal.HOLD, 0

    bars = position.get("history")

    # Fetch remote data if nothing was provided
    if bars is None or (isinstance(bars, list) and len(bars) == 0):
        logger.warning(f"No history for {symbol}. Fetching price data.")
        bars = fetch_data(symbol)

    # Format cleanup so nothing crashes
    bars = normalize_bars(bars)

    if len(bars) < 20:
        logger.warning(f"Not enough data for {symbol}. Need twenty bars minimum.")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]

    # Compute simple moving average
    sma20 = sum(closes[-20:]) / 20

    # Compute standard deviation
    variance = sum((p - sma20) ** 2 for p in closes[-20:]) / 20
    stddev = variance ** 0.5

    # Compute upper and lower bands
    upper_band = sma20 + (2 * stddev)
    lower_band = sma20 - (2 * stddev)

    current_price = closes[-1]

    logger.info(
        f"[{symbol}] Price {current_price:.2f}, SMA20 {sma20:.2f}, "
        f"Upper {upper_band:.2f}, Lower {lower_band:.2f}"
    )

    # Price below the lower band
    if current_price < lower_band:
        logger.info(f"[{symbol}] BUY signal triggered. Price below lower band.")
        return SideSignal.BUY, 1

    # Price above the upper band
    if current_price > upper_band:
        logger.info(f"[{symbol}] SELL signal triggered. Price above upper band.")
        return SideSignal.SELL, 1

    # Price in the middle band range
    logger.info(f"[{symbol}] HOLD. Price within the Bollinger range.")
    return SideSignal.HOLD, 0
