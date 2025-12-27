from .alpaca_trader.live_trading import AlpacaTrader
from .strategies.strategies import find_strategy

__version__ = "0.1.0"
__author__ = "Milad Tootoonchi"
__email__ = "milad.tootoonchi04@gmail.com"
__description__ = "A small package for updating posistions in Alpaca"
__all__ = ["AlpacaTrader", "find_strategy"]