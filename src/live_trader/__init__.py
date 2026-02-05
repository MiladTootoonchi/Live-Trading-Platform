from .alpaca_trader.live_trading import AlpacaTrader
from .alpaca_trader.order import SideSignal
from .config import Config

__version__ = "2.0.0"
__author__ = [
    "Milad Tootoonchi",
    "Makka Dulgaeva"
]
__description__ = "A small package for analyzing and updating posistions in Alpaca"
__all__ = ["AlpacaTrader", "SideSignal", "Config"]