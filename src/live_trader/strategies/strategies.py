from typing import Callable, Dict, Any

from live_trader.alpaca_trader.order import SideSignal
from config import make_logger, load_strategy_name

from .bollinger_bands_strategy import bollinger_bands_strategy
from .macd import macd_strategy
from .mean_reversion import mean_reversion_strategy
from .momentum import momentum_strategy
from .moving_average_strategy import moving_average_strategy
from .rsi import rsi_strategy
from live_trader.ml_model.ml_strategies import (basic_lstm, attention_bilstm, 
                                                tcn_lite, patchtst_lite, gnn_lite,
                                                nad_lite, cnn_gru_lite)

logger = make_logger()

def rule_based_strategy(symbol: str, position_data: dict) -> tuple[SideSignal, int]:
    """
    Evaluates a trading position from an Alpaca JSON response and recommends an action.

    Args:
        symbol (str): The symbol of the stock we want to calculate for.
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
        logger.error("Missing key in position data\n")
        return SideSignal.HOLD, 0
    
    except Exception:
        logger.error("Error evaluating position\n")
        return SideSignal.HOLD, 0



strategies = {
    "rule_based_strategy": rule_based_strategy,
    "bollinger_bands_strategy": bollinger_bands_strategy,
    "macd_strategy": macd_strategy,
    "mean_reversion_strategy": mean_reversion_strategy,
    "momentum_strategy": momentum_strategy,
    "moving_average_strategy": moving_average_strategy,
    "rsi_strategy": rsi_strategy,
    "lstm": basic_lstm,
    "bilstm": attention_bilstm,
    "tcn": tcn_lite,
    "patchtst": patchtst_lite,
    "gnn": gnn_lite,
    "nad": nad_lite,
    "cnn_gru": cnn_gru_lite,
}


def find_strategy(name: str | None = None) -> Callable[[Dict[str, Any]], tuple[SideSignal, int]]:
    """
    Resolve and return a trading strategy function by name.

    If no strategy name is provided, the user is prompted to select one.
    The function repeatedly asks for input until a valid strategy name
    matching a key in the internal ``strategies`` dictionary is supplied.

    Args:
        name (str | None):
            The name of the strategy to use. If ``None``, the strategy name
            is loaded from configuration or requested interactively from
            the user.

    Returns:
        Callable[[Dict[str, Any]], tuple[SideSignal, int]]:
            The strategy function associated with the chosen name.

    Raises:
        KeyboardInterrupt:
            If the user aborts the selection process.
    """
    
    while True:
        if name == None:
            name = load_strategy_name()

        if name not in strategies:
            name = input("Which strategy do you want to use? ")

        try:
            return strategies[name]
        
        except KeyError:
            print(f"\nStrategy {name!r} was not found in the strategies dictionary. Try again...")