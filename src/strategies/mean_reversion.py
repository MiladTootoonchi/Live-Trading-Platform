from ..alpaca_trader.order import SideSignal

def mean_reversion_strategy(position_data: dict, moving_avg_price: float) -> tuple[SideSignal, int]:
    """
   Mean reversion strategy based on market moving average (SMA).

    Logic:
    - Buy when the price is significantly below the SMA (assumed undervalued).
    - Sell when the price is significantly above the SMA (assumed overvalued).
    - Stop-loss if the price falls too far below the SMA after buying.
    - Otherwise, hold the position.

    Args:
        position_data (dict): Position details from trading API, expects keys: "qty", "current_price".
        moving_avg_price (float): Market moving average price (e.g., 20-period SMA).

    Returns:
        tuple: (SideSignal, quantity)

    """
    try: 
        qty = int(float(position_data.get("qty", 0)))
        current_price = float(position_data["current_price"])

        if moving_avg_price == 0:
            return SideSignal.HOLD, 0 # Avoid division by zero
        
        deviation = (current_price - moving_avg_price) / moving_avg_price * 100

        # buy if price is more than 3% below SMA
        if qty == 0 and deviation < -3:
            return SideSignal.BUY, 10      

        # sell if price is more than 4% above SMA
        if qty > 0 and deviation > 4:
            return SideSignal.SELL, qty

        # Stop-loss if price drops too far
        if qty > 0 and deviation < -6:
            return SideSignal.SELL, qty

        return SideSignal.HOLD, 0

    except (KeyError, ValueError) as e:
        print(f"[Mean Reversion Strategy] Error: {e}")
        return SideSignal.HOLD, 0 