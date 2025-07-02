
from strategy import SideSignal

def momentum_strategy(position_data: dict) -> tuple[SideSignal, int]:
    """
    A simple momentum-based trading strategy.

    Logic:
    - Buy when there is strong positive price momentum (>2% intraday gain).
    - Sell when the position has gained >5%.
    - Sell if today's change is negative and gains are weak (<2%).
    - Sell to cut losses if unrealized return drops below -3%.
    - Hold otherwise.

    Args:
        position_data (dict): Position details, typically from a trading API.

    Returns:
        tuple: (SideSignal.BUY or SideSignal.SELL, quantity), or (SideSignal.HOLD, 0) to hold.
    """
    try:
        qty = int(float(position_data["qty"]))
        avg_entry_price = float(position_data["avg_entry_price"])
        current_price = float(position_data["current_price"])
        change_today = float(position_data["change_today"])

        unrealized_return_pct = (
            (current_price - avg_entry_price) / avg_entry_price * 100
            if avg_entry_price > 0 else 0
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
        print(f"[Momentum Strategy] Missing key: {e}")
        return SideSignal.HOLD, 0

    except Exception as e:
        print(f"[Momentum Strategy] Error: {e}")
        return SideSignal.HOLD, 0
