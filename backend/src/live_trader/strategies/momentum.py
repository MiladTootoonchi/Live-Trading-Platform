from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy
from typing import Tuple

class MomentumStrategy(BaseStrategy):
    def _run(self) -> Tuple[SideSignal, int]:
        """
        Momentum trading strategy based on intraday movement and current return.

        The strategy observes how much the price has moved from the open and
        compares current price performance to the position entry price. It issues
        buy or sell signals based on strong positive or negative movement.

        Returns:
            tuple:
                A pair consisting of a SideSignal value and a quantity. The
                quantity returned here is set to one for valid buy or sell signals.
        """
        if self._datapipeline.data.empty:
            return SideSignal.HOLD, 0


        current_price = float(self._datapipeline.data["close"].iloc[-1])

        open_price_today = float(self._datapipeline.data["open"].iloc[-1])

        if open_price_today == 0:
            self._config.log_error(f"Momentum strategy detected invalid open price for {self._datapipeline.symbol}.")
            return SideSignal.HOLD, 0

        change_today = (current_price - open_price_today) / open_price_today * 100

        avg_entry_price = float(self._datapipeline.position_data.get("avg_entry_price") or current_price)
        unrealized_return_pct = (current_price - avg_entry_price) / avg_entry_price * 100

        self._config.log_info(
            f"Momentum strategy evaluated {self._datapipeline.symbol}. Current price is {current_price:.2f} "
            f"and intraday change is {change_today:.2f} percent."
        )

        if change_today > 0.5:
            self._config.log_info("Momentum strategy generated a buy signal.")
            return SideSignal.BUY, 1

        if unrealized_return_pct > 2:
            self._config.log_info("Momentum strategy recommends selling due to healthy positive return.")
            return SideSignal.SELL, 1

        if change_today < -0.5 and unrealized_return_pct < 2:
            self._config.log_info("Momentum strategy recommends selling due to negative movement.")
            return SideSignal.SELL, 1

        if unrealized_return_pct < -1.5:
            self._config.log_info("Momentum strategy recommends selling due to loss exceeding threshold.")
            return SideSignal.SELL, 1

        self._config.log_info("Momentum strategy holds position. No significant price movement.")
        return SideSignal.HOLD, 0
