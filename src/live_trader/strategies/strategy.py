from live_trader.alpaca_trader.order import SideSignal
from live_trader.config import Config
from live_trader.strategies.data import MarketDataPipeline
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, config: Config):
        self._config = config
        self._datapipeline = None

    def prepare_data(self, symbol: str, position_data: dict):
        self._datapipeline = MarketDataPipeline(self._config, symbol, position_data)

    def run(self) -> tuple[SideSignal, int]:
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

        if not self._datapipeline:
            self._config.log_error(f"You must prepare data before running. ")
            return (SideSignal.HOLD, 0)
        
        return self._run()
    
    @abstractmethod
    def _run(self) -> tuple[SideSignal, int]:
        pass



class RuleBasedStrategy(BaseStrategy):
    def _run(self) -> tuple[SideSignal, int]:
        try:
            qty = int(float(self._datapipeline.position_data["qty"]))
            avg_entry_price = float(self._datapipeline.position_data["avg_entry_price"])
            current_price = float(self._datapipeline.position_data["current_price"])
            change_today = float(self._datapipeline.position_data["change_today"])

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
            self._config.log_error("Missing key in position data\n")
            return SideSignal.HOLD, 0
        
        except Exception:
            self._config.log_error("Error evaluating position\n")
            return SideSignal.HOLD, 0