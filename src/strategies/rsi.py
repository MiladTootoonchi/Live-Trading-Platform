from ..alpaca_trader.order import SideSignal
from typing import Tuple
from config import make_logger
from src.strategies.fetch_price_data import fetch_price_data 

logger = make_logger()

def calculate_rsi(closes, period: int = 14) -> float:
    gains, losses = [], []
    for i in range(1, period + 1):
        change = closes[-i] - closes[-(i + 1)]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))

    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 1e-10  # Avoid division by zero
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def rsi_strategy(position_data: dict) -> Tuple[SideSignal, int]:
    """RSI Strategy: Buy if RSI < 30, Sell if RSI > 70, else Hold."""
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("Missing 'symbol' in position_data")
        return SideSignal.HOLD, 0

    bars = fetch_price_data(symbol)
    if len(bars) < 15:
        logger.info(f"Not enough bars to calculate RSI for {symbol}")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    rsi = calculate_rsi(closes[-15:])
    logger.info(f"[{symbol}] RSI: {rsi:.2f}")

    if rsi < 30:
        logger.info(f"[{symbol}] BUY signal - RSI below 30")
        return SideSignal.BUY, 0
    elif rsi > 70:
        logger.info(f"[{symbol}] SELL signal - RSI above 70")
        return SideSignal.SELL, 0
    else:
        logger.info(f"[{symbol}] HOLD - RSI neutral")
        return SideSignal.HOLD, 0


