from ..alpaca_trader.order import SideSignal
from config import make_logger
from .fetch_price_data import fetch_price_data 

logger = make_logger()


def mean_reversion_strategy(position_data: dict, moving_avg_price: float = 0, min_data_points: int = 1) -> tuple[SideSignal, int]:
    """Generates mean reversion BUY/SELL/HOLD signals."""
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("[Mean Reversion] No symbol provided")
        return SideSignal.HOLD, 0
    
    bars = position_data.get("history", [])

    if not bars:
        logger.warning(f"[Mean Reversion] No history in position_data, fetching from API for {symbol}")
        bars = fetch_price_data(symbol)


    if bars and len(bars) > 0:
        current_price = float(bars[-1]["c"])
    else:
        current_price = float(position_data.get("current_price", 0))

    if moving_avg_price == 0:
        logger.info("[Mean Reversion] Not enough data to calculate moving average")
        return SideSignal.HOLD, 0

    deviation = (current_price - moving_avg_price) / moving_avg_price * 100

    logger.info(f"[Mean Reversion] Current: {current_price:.2f}, SMA: {moving_avg_price:.2f}, Deviation: {deviation:.2f}%")

    if deviation < -2:
        logger.info(f"[Mean Reversion] SELL signal - stop-loss triggered at {deviation:.2f}%")
        return SideSignal.SELL, 0
    elif deviation < -1:
        logger.info(f"[Mean Reversion] BUY signal - price {deviation:.2f}% below SMA")
        return SideSignal.BUY, 0
    elif deviation > 1:
        logger.info(f"[Mean Reversion] SELL signal - price {deviation:.2f}% above SMA")
        return SideSignal.SELL, 0
    else:
        logger.info("[Mean Reversion] HOLD - price within acceptable range")
        return SideSignal.HOLD, 0



      

   

