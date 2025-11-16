from ..alpaca_trader.order import SideSignal
from typing import Dict, Any, Tuple
from config import make_logger
from .utils import fetch_data, normalize_bars
import pandas as pd

logger = make_logger()


def momentum_strategy(position_data: Dict[str, Any]) -> Tuple[SideSignal, int]:
    """
    Momentum trading strategy based on intraday movement and current return.

    The strategy observes how much the price has moved from the open and
    compares current price performance to the position entry price. It issues
    buy or sell signals based on strong positive or negative movement.

    Args:
        position_data (dict):
            Should contain:
            symbol: the ticker to evaluate
            history: optional price history from the position or API
            avg_entry_price: the price paid for the current position

    Returns:
        tuple:
            A pair consisting of a SideSignal value and a quantity. The
            quantity returned here is set to one for valid buy or sell signals.
    """
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("Momentum strategy received data without a symbol.")
        return SideSignal.HOLD, 0

    bars = position_data.get("history")

    if bars is None or (isinstance(bars, pd.DataFrame) and bars.empty) or (isinstance(bars, list) and len(bars) == 0):
        logger.warning(f"Momentum strategy found no history. Fetching fresh data for {symbol}.")
        bars = fetch_data(symbol)

    if bars is None:
        logger.warning(f"Momentum strategy received no data after fetching for {symbol}.")
        return SideSignal.HOLD, 0

    bars = normalize_bars(bars)

    if len(bars) == 0:
        logger.warning(f"Momentum strategy could not normalize data for {symbol}.")
        return SideSignal.HOLD, 0

    current_price = float(bars[-1]["c"])

    if "o" in bars[-1]:
        open_price_today = float(bars[-1]["o"])
    else:
        logger.warning(f"Momentum strategy missing open price. Using close price as fallback for {symbol}.")
        open_price_today = current_price

    if open_price_today == 0:
        logger.error(f"Momentum strategy detected invalid open price for {symbol}.")
        return SideSignal.HOLD, 0

    change_today = (current_price - open_price_today) / open_price_today * 100

    avg_entry_price = float(position_data.get("avg_entry_price") or current_price)
    unrealized_return_pct = (current_price - avg_entry_price) / avg_entry_price * 100

    logger.info(
        f"Momentum strategy evaluated {symbol}. Current price is {current_price:.2f} "
        f"and intraday change is {change_today:.2f} percent."
    )

    if change_today > 0.5:
        logger.info("Momentum strategy generated a buy signal.")
        return SideSignal.BUY, 1

    if unrealized_return_pct > 2:
        logger.info("Momentum strategy recommends selling due to healthy positive return.")
        return SideSignal.SELL, 1

    if change_today < -0.5 and unrealized_return_pct < 2:
        logger.info("Momentum strategy recommends selling due to negative movement.")
        return SideSignal.SELL, 1

    if unrealized_return_pct < -1.5:
        logger.info("Momentum strategy recommends selling due to loss exceeding threshold.")
        return SideSignal.SELL, 1

    logger.info("Momentum strategy holds position. No significant price movement.")
    return SideSignal.HOLD, 0
