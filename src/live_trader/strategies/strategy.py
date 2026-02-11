from live_trader.alpaca_trader.order import SideSignal
from live_trader.config import Config
from live_trader.strategies.data import MarketDataPipeline
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Defines the lifecycle for preparing market data
    and generating trading signals. Subclasses must
    implement the '_run' method with decision logic.
    """
    def __init__(self, config: Config):
        """
        Initialize strategy with configuration.

        Args:
            config: Application configuration object.
        """
        self._config = config
        self._datapipeline = None

    @property
    def data(self) -> MarketDataPipeline | None:
        return self._datapipeline

    def prepare_data(self, symbol: str, position_data: dict | None = None) -> None:
        """
        Initialize market data pipeline for a symbol.

        Creates a MarketDataPipeline instance used
        during strategy evaluation.

        Args:
            symbol: Ticker symbol.
            position_data: Optional position metadata.
        """
        self._datapipeline = MarketDataPipeline(self._config, symbol, position_data or {})

    @abstractmethod
    def _run(self) -> tuple[SideSignal, int]:
        """
        Execute strategy decision logic.

        Must return a trading signal and quantity
        based on prepared market data.

        Returns:
            tuple[SideSignal, int]: Signal and quantity.
        """
        pass

    def run(self) -> tuple[SideSignal, int]:
        """
        Execute strategy evaluation.

        Validates that market data has been prepared
        before delegating to the subclass `_run`
        implementation.

        Returns:
            tuple[SideSignal, int]: Trading signal and quantity.
        """

        if not self._datapipeline: 
            self._config.log_error(f"You must prepare data before running. ") 
            return (SideSignal.HOLD, 0)
        
        return self._run()



class RuleBasedStrategy(BaseStrategy):
    def _run(self) -> tuple[SideSignal, int]:
        """
        Evaluate position using predefined risk rules.

        Uses unrealized return percentage and daily
        price movement thresholds to trigger buy
        or sell decisions.

        Returns:
            tuple[SideSignal, int]: Signal and quantity.
        """
        try:
            pd = self._datapipeline.position_data

            qty = int(pd.get("qty", 0))
            avg_entry_price = float(pd.get("avg_entry_price", 0.0))
            current_price = float(pd.get("market_price", 0.0))
            change_today = float(pd.get("change_today", 0.0))
            
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