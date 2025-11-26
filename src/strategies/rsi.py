from ..alpaca_trader.order import SideSignal
from typing import Tuple, List, Union, Dict, Any
from config import make_logger
from .utils import fetch_data, normalize_bars
import pandas as pd

logger = make_logger()


def calculate_rsi(closes: List[float], period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        closes (List[float]):
            A list of close prices ordered from oldest to newest.
        period (int):
            The RSI lookback period. Default is 14.

    Returns:
        float:
            The computed RSI value (0-100).
    """
    gains, losses = [], []

    for i in range(1, period + 1):
        change = closes[-i] - closes[-(i + 1)]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))

    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 1e-10
    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def rsi_strategy(position_data: Dict[str, Any]) -> Tuple[SideSignal, int]:
    """
    A simple RSI-based trading strategy.

    Generates signals based on RSI thresholds:
    - Buy when RSI < 30
    - Sell when RSI > 70
    - Hold otherwise

    Args:
        position_data (Dict[str, Any]):
            A dictionary containing:
            - "symbol": The ticker symbol.
            - "history": Historical bar data (various formats supported).

    Returns:
        Tuple[SideSignal, int]:
            A tuple where:
            - The first value is a SideSignal (BUY, SELL, HOLD)
            - The second value is the quantity (always 0 here)
    """

    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("Position data missing 'symbol'.")
        return SideSignal.HOLD, 0

    bars = position_data.get("history")

    history_missing = (
        bars is None or
        (isinstance(bars, pd.DataFrame) and bars.empty) or
        (isinstance(bars, list) and len(bars) == 0)
    )

    if history_missing:
        logger.warning(f"[RSI] No history found for {symbol}. Fetching from API.")
        bars = fetch_data(symbol)

    if bars is None:
        logger.error(f"[RSI] Unable to fetch data for {symbol}.")
        return SideSignal.HOLD, 0

    bars = normalize_bars(bars)

    if len(bars) < 15:
        logger.info(f"[RSI] Not enough historical bars to compute RSI for {symbol}.")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    rsi = calculate_rsi(closes[-15:])

    logger.info(f"[{symbol}] RSI: {rsi:.2f}")

    if rsi < 30:
        logger.info(f"[{symbol}] Buy signal triggered.")
        return SideSignal.BUY, 1

    if rsi > 70:
        logger.info(f"[{symbol}] Sell signal triggered.")
        return SideSignal.SELL, 1

    logger.info(f"[{symbol}] Hold signal (RSI neutral).")
    return SideSignal.HOLD, 0
