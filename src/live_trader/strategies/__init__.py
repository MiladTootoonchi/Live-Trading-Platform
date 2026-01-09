from .run_backtest import main
from .bollinger_bands_strategy import bollinger_bands_strategy
from .macd import macd_strategy
from .mean_reversion import mean_reversion_strategy
from .momentum import momentum_strategy
from .moving_average_strategy import moving_average_strategy
from .rsi import rsi_strategy
from .strategies import rule_based_strategy, find_strategy

__all__ = ["main", "bollinger_bands_strategy", "macd_strategy", "mean_reversion_strategy", 
           "momentum_strategy", "moving_average_strategy", "rsi_strategy", "rule_based_strategy", "find_strategy"]
__author__ = [
    "Makke Dulgaeva",
    "Milad Tootoonchi"
]