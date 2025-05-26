"""
Different functions with strategies that will generate a signal for buying, selling or holding
given position information. Then it will return a signal with quantity of order.
"""
from typing import Optional

def rule_based_strategy(position_data: dict) -> Optional[tuple]:
    """
    Evaluates a trading position from an Alpaca JSON response and recommends an action.

    Args:
        position_data (dict): JSON object from Alpaca API containing position details.

    Returns:
        tuple or None:
            ('buy' or 'sell', qty: int) if action is needed,
            None if holding the position.
    """
    try:
        qty = int(float(position_data["qty"]))
        avg_entry_price = float(position_data["avg_entry_price"])
        current_price = float(position_data["current_price"])
        change_today = float(position_data["change_today"])

        if qty == 0:
            return None  # Nothing to do

        unrealized_return_pct = (current_price - avg_entry_price) / avg_entry_price * 100

        # Decision rules
        if unrealized_return_pct > 20:
            return ("sell", qty)
        if unrealized_return_pct < -15:
            return ("sell", qty)
        if change_today < -3 and unrealized_return_pct < 0:
            return ("buy", qty)

        return None  # Hold

    except KeyError:
        print("Missing key in position data")
        return None
    
    except Exception:
        print("Error evaluating position")
        return None
