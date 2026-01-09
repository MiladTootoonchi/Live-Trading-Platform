from live_trader.alpaca_trader.order import SideSignal
from typing import Tuple, Dict, Any, List
from config import make_logger
from .utils import fetch_data, normalize_bars

logger = make_logger()


def exponential_moving_average(data: List[float], period: int) -> List[float]:
    """
    Computes an exponential moving average.

    Args:
        data (list of float):
            Closing prices used for EMA.
        period (int):
            EMA period length.

    Returns:
        list of float:
            EMA values.
    """
    if not data or period <= 0:
        return []

    ema = []
    k = 2 / (period + 1)

    for i, price in enumerate(data):
        if i == 0:
            ema.append(price)
        else:
            ema.append(price * k + ema[i - 1] * (1 - k))

    return ema


def calculate_macd(closes: List[float]) -> Tuple[List[float], List[float]]:
    """
    Calculates MACD and signal lines.

    Args:
        closes (list of float):
            Close prices in chronological order.

    Returns:
        tuple(list, list):
            MACD line and signal line.
    """
    ema12 = exponential_moving_average(closes, 12)
    ema26 = exponential_moving_average(closes, 26)

    min_len = min(len(ema12), len(ema26))
    ema12, ema26 = ema12[-min_len:], ema26[-min_len:]

    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = exponential_moving_average(macd_line, 9)

    return macd_line, signal_line


def macd_strategy(symbol: str, position_data: Dict[str, Any]) -> Tuple[SideSignal, int]:
    """
    MACD crossover strategy.
    Generates buy or sell signals when MACD crosses the signal line.

    Args:
        symbol (str): The symbol of the stock we want to calculate for.
        position_data (dict):
            Contains:
                symbol (str): ticker symbol
                history (list/DF): optional bar data
                current_price (float): fallback price

    Returns:
        tuple(SideSignal, int):
            Signal and quantity (always 0 for risk management).
    """
    
    bars = position_data.get("history")

    # Fetch if no history provided
    if bars is None or (isinstance(bars, list) and len(bars) == 0):
        logger.warning(f"No price history found for {symbol}. Fetching from API...")
        bars = fetch_data(symbol)

    # Fix: Normalize everything to the same clean format
    bars = normalize_bars(bars)

    if len(bars) < 35:
        logger.info(f"Not enough data to compute MACD for {symbol}. Need 35 bars.")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]

    macd_line, signal_line = calculate_macd(closes)

    if len(macd_line) < 2 or len(signal_line) < 2:
        logger.info(f"Insufficient MACD data for {symbol}.")
        return SideSignal.HOLD, 0

    prev_macd, curr_macd = macd_line[-2], macd_line[-1]
    prev_signal, curr_signal = signal_line[-2], signal_line[-1]

    # Bullish crossover
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        logger.info(f"[{symbol}] BUY signal — MACD bullish crossover.")
        return SideSignal.BUY, 1

    # Bearish crossover
    if prev_macd >= prev_signal and curr_macd < curr_signal:
        logger.info(f"[{symbol}] SELL signal — MACD bearish crossover.")
        return SideSignal.SELL, 1

    logger.info(f"[{symbol}] HOLD — no MACD crossover.")
    return SideSignal.HOLD, 0
