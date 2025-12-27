from live_trader.alpaca_trader.order import SideSignal
from typing import Dict, Any, Tuple
from config import make_logger
from .utils import fetch_data, normalize_bars
import pandas as pd

logger = make_logger()


def moving_average_strategy(position: Dict[str, Any]) -> Tuple[SideSignal, int]:
    """
    Moving Average Crossover Strategy.

    Logic:
        - Buy when price > MA20 > MA50 > MA200
        - Sell when price < MA20 < MA50 < MA200
        - Hold when neither condition is met

    Args:
        position (dict):
            Contains at minimum:
            {
                "symbol": "AAPL",
                "history": DataFrame | List | None
            }

    Returns:
        Tuple[SideSignal, int]:
            - A SideSignal (BUY, SELL, HOLD)
            - Quantity (always 0 here)
    """

    symbol = position.get("symbol")
    if not symbol:
        logger.error("Strategy called without a symbol in position data.")
        return SideSignal.HOLD, 0

    bars = position.get("history")

    # If no local history, fetch from API
    if bars is None or (isinstance(bars, pd.DataFrame) and bars.empty) or (isinstance(bars, list) and len(bars) == 0):
        logger.warning(f"[MA Strategy] No history for {symbol}. Fetching from API.")
        bars = fetch_data(symbol)  # FIXED: removed unsupported "limit" argument

    if bars is None:
        logger.error(f"[MA Strategy] API returned no data for {symbol}.")
        return SideSignal.HOLD, 0

    # Normalize formats (dicts with {"c": close})
    bars = normalize_bars(bars)

    if len(bars) < 200:
        logger.info(f"[MA Strategy] Not enough data for {symbol}. Need 200 bars.")
        return SideSignal.HOLD, 0

    # Extract closing prices
    closes = [float(bar["c"]) for bar in bars]

    current = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200

    logger.info(
        f"[{symbol}] Price: {current:.2f}, MA20: {ma20:.2f}, "
        f"MA50: {ma50:.2f}, MA200: {ma200:.2f}"
    )

    # Bullish alignment: BUY
    if current > ma20 > ma50 > ma200:
        logger.info(f"[{symbol}] BUY signal - bullish MA alignment.")
        return SideSignal.BUY, 1

    # Bearish alignment: SELL
    if current < ma20 < ma50 < ma200:
        logger.info(f"[{symbol}] SELL signal - bearish MA alignment.")
        return SideSignal.SELL, 1

    # Default: HOLD
    logger.info(f"[{symbol}] HOLD - MAs are neutral.")
    return SideSignal.HOLD, 0
