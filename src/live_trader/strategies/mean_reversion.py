from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy
from typing import Tuple

class MeanReversionStrategy(BaseStrategy):
    def _run(self, window: int = 20) -> Tuple[SideSignal, int]:
        """
        Mean reversion strategy that compares the current price to a moving average
        to determine if the price has deviated significantly.

        Args:
            window (int):
                Number of bars used to calculate the moving average.

        Returns:
            tuple:
                A SideSignal indicating buy, sell, or hold,
                and a recommended quantity of one for buy or sell signals.
        """
        bars = self._datapipeline.data
        symbol = self._datapipeline.symbol

        if bars.empty:
            return SideSignal.HOLD, 0


        closes = bars["close"].astype(float).tolist()

        current_price = closes[-1]

        if len(closes) < window:
            self._config.log_info("Not enough bar data to compute moving average.")
            return SideSignal.HOLD, 0

        moving_avg_price = sum(closes[-window:]) / window

        deviation = (current_price - moving_avg_price) / moving_avg_price * 100

        self._config.log_info(
            f"Mean reversion evaluated {symbol}. Current price is {current_price:.2f}, "
            f"moving average is {moving_avg_price:.2f}, deviation is {deviation:.2f} percent."
        )

        if deviation < -0.5:
            self._config.log_info("Mean reversion generated a buy signal because price is significantly below average.")
            return SideSignal.BUY, 1

        if deviation > 0.5:
            self._config.log_info("Mean reversion generated a sell signal because price is significantly above average.")
            return SideSignal.SELL, 1

        self._config.log_info("Mean reversion strategy recommends holding.")
        return SideSignal.HOLD, 0
