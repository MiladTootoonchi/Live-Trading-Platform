from ..alpaca_trader.order import SideSignal
from config import make_logger

logger = make_logger()

def mean_reversion_strategy(position_data: dict, moving_avg_price: float, min_data_points: int = 1) -> tuple[SideSignal, int]:
    """
    Mean reversion strategy based on market moving average (SMA).
    Robust against missing data.
    """
    try: 
        qty = int(float(position_data.get("qty", 0)))
        current_price = float(position_data.get("current_price", 0))

        # Check if moving average is valid
        if moving_avg_price == 0:
            logger.info("[Mean Reversion] Not enough data to calculate moving average")
            return SideSignal.HOLD, 0

        deviation = (current_price - moving_avg_price) / moving_avg_price * 100

        # Buy if price is more than 3% below SMA
        if qty == 0 and deviation < -3:
            return SideSignal.BUY, 10      

        # Sell if price is more than 4% above SMA
        if qty > 0 and deviation > 4:
            return SideSignal.SELL, qty

        # Stop-loss if price drops too far
        if qty > 0 and deviation < -6:
            return SideSignal.SELL, qty

        return SideSignal.HOLD, 0

    except (KeyError, ValueError) as e:
        logger.error(f"[Mean Reversion Strategy] Error: {e}\n")
        return SideSignal.HOLD, 0
