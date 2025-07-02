from typing import Callable, Dict, Any

from ..api_getter.order import SideSignal

from .strategy_basics import rule_based_strategy
from .bollinger_bands_strategy import bollinger_bands_strategy
from .macd import macd_strategy
from .mean_reversion import mean_reversion_strategy
from .momentum import momentum_strategy
from .moving_average_strategy import moving_average_strategy
from .rsi import rsi_strategy



def rule_based_strategy(position_data: dict) -> tuple[SideSignal, int]:
    """
    Evaluates a trading position from an Alpaca JSON response and recommends an action.

    Args:
        position_data (dict): JSON object from Alpaca API containing position details.

    Returns:
        tuple:
            (SideSignal.BUY or SideSignal.SELL, qty: int) if action is needed,
            (SideSignal.HOLD, 0) if holding the position.
    """

    try:
        qty = int(float(position_data["qty"]))
        avg_entry_price = float(position_data["avg_entry_price"])
        current_price = float(position_data["current_price"])
        change_today = float(position_data["change_today"])

        if qty == 0:
            return None, 0  # Nothing to do

        unrealized_return_pct = (current_price - avg_entry_price) / avg_entry_price * 100

        # Decision rules
        if unrealized_return_pct > 2:
            return (SideSignal.SELL, qty)
        if unrealized_return_pct < -1.5:
            return (SideSignal.SELL, qty)
        if change_today < -3 and unrealized_return_pct < 0:
            return (SideSignal.BUY, qty)

        return SideSignal.HOLD, 0  # Hold

    except KeyError:
        print("Missing key in position data")
        return SideSignal.HOLD, 0
    
    except Exception:
        print("Error evaluating position")
        return SideSignal.HOLD, 0



strategies = {
    "rule_based_strategy": rule_based_strategy,
    "bollinger_bands_strategy": bollinger_bands_strategy,
    "macd_strategy": macd_strategy,
    "mean_reversion_strategy": mean_reversion_strategy,
    "momentum_strategy": momentum_strategy,
    "moving_average_strategy": moving_average_strategy,
    "rsi_strategy": rsi_strategy,
}


def find_strategy() -> Callable[[Dict[str, Any]], tuple[SideSignal, int]]:
    """
    Goes through the strategies dictionary to call on the strategy function 
    that matches the promt the user inputs.

    Returns:
        Callable: the strategy function asked for.

    Raises:
        KeyError: If the strategy name is not found.
    """
    name = input("Which strategy do you want to use? ")

    try:
        return strategies[name]
    
    except KeyError:
        print(f"Strategy {name!r} was not found in the strategies dictionary.")