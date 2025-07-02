from typing import Callable, Dict, Any

from .strategy_basics import rule_based_strategy, SideSignal
from .bollinger_bands_strategy import bollinger_bands_strategy
from .macd import macd_strategy
from .mean_reversion import mean_reversion_strategy
from .momentum import momentum_strategy
from .moving_average_strategy import moving_average_strategy
from .rsi import rsi_strategy

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