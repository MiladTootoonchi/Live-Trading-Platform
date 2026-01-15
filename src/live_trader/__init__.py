from .alpaca_trader.live_trading import AlpacaTrader
from .alpaca_trader.order import SideSignal
from .strategies.strategies import find_strategy
from .config import load_api_keys

__version__ = "0.3.1"
__author__ = [
    "Milad Tootoonchi",
    "Makka Dulgaeva"
]
__description__ = "A small package for analyzing and updating posistions in Alpaca"
__all__ = ["AlpacaTrader", "find_strategy", "SideSignal", "load_api_keys"]