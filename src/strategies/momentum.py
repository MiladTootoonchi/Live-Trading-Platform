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

        if qty == 0 and change_today > 2:
            return SideSignal.BUY, 10

        if unrealized_return_pct > 5:
            return SideSignal.SELL, qty

        if change_today < -1 and unrealized_return_pct < 2:
            return SideSignal.SELL, qty

        if unrealized_return_pct < -3:
            return SideSignal.SELL, qty

        return SideSignal.HOLD, 0

    except KeyError as e:
        logger.error(f"[Momentum Strategy] Missing key: {e}\n")
        return SideSignal.HOLD, 0

    except Exception as e:
        logger.error(f"[Momentum Strategy] Error: {e}\n")
        return SideSignal.HOLD, 0
