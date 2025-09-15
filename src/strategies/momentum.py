from ..alpaca_trader.order import SideSignal
from config import make_logger

logger = make_logger()

def momentum_strategy(position_data: dict) -> tuple[SideSignal, int]:
    """
    A simple momentum-based trading strategy.
    Robust against missing or incomplete data.
    """
    try:
        qty = int(float(position_data.get("qty", 0)))
        avg_entry_price = float(position_data.get("avg_entry_price", 0))
        current_price = float(position_data.get("current_price", 0))
        change_today = float(position_data.get("change_today", 0))
        
        # Check for insufficient data
        if current_price == 0 or avg_entry_price == 0:
            logger.info("[Momentum Strategy] Not enough data to evaluate position")
            return SideSignal.HOLD, 0
        
        unrealized_return_pct = (
            (current_price - avg_entry_price) / avg_entry_price * 100
        )
        
        logger.info(f"[Momentum] Price: {current_price:.2f}, Entry: {avg_entry_price:.2f}, "
                   f"Return: {unrealized_return_pct:.2f}%, Change today: {change_today:.2f}%")
        
        # Buy signal: strong positive momentum today and no position
        if qty == 0 and change_today > 2:
            logger.info(f"[Momentum] BUY signal - strong positive momentum: {change_today:.2f}%")
            return SideSignal.BUY, 10
        
        # Take profit: 5% gain
        if unrealized_return_pct > 5:
            logger.info(f"[Momentum] SELL signal - take profit at {unrealized_return_pct:.2f}%")
            return SideSignal.SELL, qty
        
        # Sell on negative momentum with small gain
        if change_today < -1 and unrealized_return_pct < 2:
            logger.info(f"[Momentum] SELL signal - negative momentum with small gain")
            return SideSignal.SELL, qty
        
        # Stop loss: 3% loss
        if unrealized_return_pct < -3:
            logger.info(f"[Momentum] SELL signal - stop loss at {unrealized_return_pct:.2f}%")
            return SideSignal.SELL, qty
        
        logger.info("[Momentum] HOLD - no momentum signals triggered")
        return SideSignal.HOLD, 0
        
    except KeyError as e:
        logger.error(f"[Momentum Strategy] Missing key: {e}")
        return SideSignal.HOLD, 0
    except Exception as e:
        logger.error(f"[Momentum Strategy] Error: {e}")
        return SideSignal.HOLD, 0