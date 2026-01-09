import asyncio

from ..alpaca_trader import SideSignal
from .training import ML_Pipeline
from .modelling import build_lstm, build_attention_bilstm



async def AI_strategy(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_lstm model from modelling.

    Args:
        symbol (str):           a string consisting of the symbol og the stock.
        position_data (dict):   JSON object from Alpaca API containing position details.
                                This is only used as a parameter if you have a posistion in that stock.

    Returns:
        tuple:
            (SideSignal.BUY or SideSignal.SELL, qty: int)
    """

    if position_data is None:
        position_data = {}

    side, qty = await ML_Pipeline(build_lstm, symbol, position_data)
    return side, qty


async def attention_bilstm_strategy(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_attention_bilstm model from modelling.

    Args:
        symbol (str):           a string consisting of the symbol og the stock.
        position_data (dict):   JSON object from Alpaca API containing position details.
                                This is only used as a parameter if you have a posistion in that stock.

    Returns:
        tuple:
            (SideSignal.BUY or SideSignal.SELL, qty: int)
    """

    if position_data is None:
        position_data = {}

    side, qty = await ML_Pipeline(build_attention_bilstm, symbol, position_data)
    return side, qty