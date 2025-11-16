from ..alpaca_trader.order import SideSignal
from config import make_logger
from .fetch_price_data import fetch_data 

logger = make_logger()


def momentum_strategy(position_data: dict) -> tuple[SideSignal, int]:
    """Generates a momentum-based BUY/SELL/HOLD signal."""
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("[Momentum] No symbol provided")
        return SideSignal.HOLD, 0
    
    bars = position_data.get("history", [])

    if not bars:
        logger.warning(f"[Momentum] No history in position_data, fetching from API for {symbol}")
        bars = fetch_data(symbol)

    if not bars:
        logger.warning(f"[Momentum] No data available for {symbol}")
        return SideSignal.HOLD, 0

    current_price = float(bars[-1]["c"])
    open_price_today = float(bars[-1]["o"])
    change_today = (current_price - open_price_today) / open_price_today * 100
    avg_entry_price = float(position_data.get("avg_entry_price") or current_price)
    unrealized_return_pct = (current_price - avg_entry_price) / avg_entry_price * 100

    logger.info(f"[Momentum] {symbol} Price: {current_price:.2f}, Change today: {change_today:.2f}%")

    # Buy: strong positive momentum
    if change_today > 0.5:
        logger.info("[Momentum] BUY signal triggered")
        return SideSignal.BUY, 0

    # Sell: take profit, negative momentum, or stop loss
    elif unrealized_return_pct > 2:
        logger.info("[Momentum] SELL signal - take profit")
        return SideSignal.SELL, 0

    elif change_today < -0.5 and unrealized_return_pct < 2:
        logger.info("[Momentum] SELL signal - negative momentum with small gain")
        return SideSignal.SELL, 0

    elif unrealized_return_pct < -1.5:
        logger.info("[Momentum] SELL signal - stop loss")
        return SideSignal.SELL, 0

    else:
        logger.info("[Momentum] HOLD - no significant movement")
        return SideSignal.HOLD, 0

