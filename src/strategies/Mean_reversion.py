
from strategy import SideSignal

def mean_reversion_strategy(position_data: dict) -> tuple[SideSignal, int]:
    """
    A simple mean reversion trading strategy.

    Logic:
    - If the currrent price is significally below average entry price, 
    we assume undervaluation and buy.
    - If the price is significally above average, we assume overevaluation and sell.
    - Also cut losses if price drops too far.
    - Otherwise, hold the position

    Args: 
        position_data (dict): Position details from the trading API.

    Returns:
        tuple: (SideSignal.BUY or SideSignal.SELL, quantity), or (SideSignal.HOLD, 0).

    """
    try: 
        qty = int(float(position_data["qty"]))
        avg_entry_price = float(position_data["avg_entry_price"])
        current_price = float(position_data["current_price"])

        if avg_entry_price == 0:
            return SideSignal.HOLD, 0 # Avoid divison by zero
        
        deviation = (current_price - avg_entry_price) / avg_entry_price * 100

        # Buy if current price is much lower than historical (entry) price
        if qty == 0 and deviation < -3:
            return SideSignal.BUY, 10      

        # Sell if price has gone much lower than historical (entry) price
        if deviation > 4:
            return SideSignal.SELL, qty

        # Stop-loss if price drops too far
        if deviation < -6:
            return SideSignal.SELL, qty

        return SideSignal.HOLD, 0
    except KeyError as e:
        print(f"[Mean Reversion Strategy] Error: {e}")
        return SideSignal.HOLD, 0  