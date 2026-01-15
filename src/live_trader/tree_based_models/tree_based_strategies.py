from live_trader.ml_model.training import ML_Pipeline
from live_trader.alpaca_trader import SideSignal

from live_trader.tree_based_models.tree_modelling import build_rf, buildXGB, buildLGBM, build_catboost

async def random_forest(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with a classical ML model from tree_modelling.

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

    side, qty = await ML_Pipeline(build_rf, symbol, position_data)
    return side, qty



async def xgboost(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with a classical ML model from tree_modelling.

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

    side, qty = await ML_Pipeline(buildXGB, symbol, position_data)
    return side, qty



async def lightgbm(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with a classical ML model from tree_modelling.

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

    side, qty = await ML_Pipeline(buildLGBM, symbol, position_data)
    return side, qty



async def catboost(symbol: str, position_data: dict = None) -> tuple[SideSignal, int]:
    """
    A strategy using the ML pipeline with a classical ML model from tree_modelling.

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

    side, qty = await ML_Pipeline(build_catboost, symbol, position_data)
    return side, qty
