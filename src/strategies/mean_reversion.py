from ..alpaca_trader.order import SideSignal
from typing import Dict, Any, Tuple
from config import make_logger
from .utils import fetch_data, normalize_bars
import pandas as pd

logger = make_logger()


def mean_reversion_strategy(
    position_data: Dict[str, Any],
    window: int = 20
) -> Tuple[SideSignal, int]:
    """
    Mean reversion strategy that compares the current price to a moving average
    to determine if the price has deviated significantly.

    Args:
        position_data (dict):
            Contains:
                symbol: ticker of the asset
                history: optional list or DataFrame of price bars
                current_price: fallback current price
        window (int):
            Number of bars used to calculate the moving average.

    Returns:
        tuple:
            A SideSignal indicating buy, sell, or hold,
            and a recommended quantity of one for buy or sell signals.
    """
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("Mean reversion strategy invoked without a symbol.")
        return SideSignal.HOLD, 0

    bars = position_data.get("history")

    if bars is None or (isinstance(bars, pd.DataFrame) and bars.empty) or (isinstance(bars, list) and len(bars) == 0):
        logger.warning(f"Mean reversion strategy found no history. Fetching fresh data for {symbol}.")
        bars = fetch_data(symbol)

    bars = normalize_bars(bars)

    if len(bars) == 0:
        logger.warning("Mean reversion strategy received no usable price history.")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]

    current_price = closes[-1]

    if len(closes) < window:
        logger.info("Not enough bar data to compute moving average.")
        return SideSignal.HOLD, 0

    moving_avg_price = sum(closes[-window:]) / window

    deviation = (current_price - moving_avg_price) / moving_avg_price * 100

    logger.info(
        f"Mean reversion evaluated {symbol}. Current price is {current_price:.2f}, "
        f"moving average is {moving_avg_price:.2f}, deviation is {deviation:.2f} percent."
    )

    if deviation < -0.5:
        logger.info("Mean reversion generated a buy signal because price is significantly below average.")
        return SideSignal.BUY, 1

    if deviation > 0.5:
        logger.info("Mean reversion generated a sell signal because price is significantly above average.")
        return SideSignal.SELL, 1

    logger.info("Mean reversion strategy recommends holding.")
    return SideSignal.HOLD, 0
