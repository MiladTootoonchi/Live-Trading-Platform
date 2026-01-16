import asyncio

from live_trader.config import make_logger
from live_trader.alpaca_trader.order import SideSignal
from live_trader.ml_model.training import ML_Pipeline
from live_trader.ml_model.modelling import (build_lstm, build_attention_bilstm, 
                                            build_tcn_lite, build_patchtst_lite, build_gnn_lite, 
                                            build_autoencoder_classifier_lite, build_cnn_gru_lite)

logger = make_logger()

ML_SEMAPHORE = asyncio.Semaphore(1)


async def basic_lstm(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
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

    try:
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_lstm, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0
    
    return side, qty


async def attention_bilstm(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
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

    try:
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_attention_bilstm, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0

    return side, qty


async def tcn_lite(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_tcn_lite model from modelling.

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

    try:    
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_tcn_lite, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0

    return side, qty


async def patchtst_lite(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_patchtst_lite model from modelling.

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

    try:
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_patchtst_lite, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0

    return side, qty


async def gnn_lite(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_gnn_lite model from modelling.

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

    try:
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_gnn_lite, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0

    return side, qty


async def nad_lite(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_autoencoder_classifier_lite model from modelling.

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

    try:
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_autoencoder_classifier_lite, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0

    return side, qty


async def cnn_gru_lite(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with build_cnn_gru_lite model from modelling.

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

    try:
        async with ML_SEMAPHORE:
            side, qty = await ML_Pipeline(build_cnn_gru_lite, symbol, position_data)
    except Exception as e:
        logger.error(f"ML strategy failed: {e}")
        return SideSignal.HOLD, 0

    return side, qty