from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy
from typing import Tuple

class MovingAverageStrategy(BaseStrategy):
    def _run(self) -> Tuple[SideSignal, int]:
        """
        Moving Average Crossover Strategy.

        Logic:
            - Buy when price > MA20 > MA50 > MA200
            - Sell when price < MA20 < MA50 < MA200
            - Hold when neither condition is met

        Args:
            symbol (str): The symbol of the stock we want to calculate for.
            position (dict):
                Contains at minimum:
                {
                    "symbol": "AAPL",
                    "history": DataFrame | List | None
                }

        Returns:
            Tuple[SideSignal, int]:
                - A SideSignal (BUY, SELL, HOLD)
                - Quantity (always 0 here)
        """
        if self._datapipeline.data.empty:
            return SideSignal.HOLD, 0


        if len(self._datapipeline.data) < 200:
            self._config.log_info(f"[MA Strategy] Not enough data for {self._datapipeline.symbol}. Need 200 bars")
            return SideSignal.HOLD, 0

        # Extract closing prices
        closes = self._datapipeline.data["close"].astype(float).tolist()

        current = closes[-1]
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        ma200 = sum(closes[-200:]) / 200

        self._config.log_info(
            f"[{self._datapipeline.symbol}] Price: {current:.2f}, MA20: {ma20:.2f}, "
            f"MA50: {ma50:.2f}, MA200: {ma200:.2f}"
        )

        # Bullish alignment: BUY
        if current > ma20 > ma50 > ma200:
            self._config.log_info(f"[{self._datapipeline.symbol}] BUY signal - bullish MA alignment.")
            return SideSignal.BUY, 1

        # Bearish alignment: SELL
        if current < ma20 < ma50 < ma200:
            self._config.log_info(f"[{self._datapipeline.symbol}] SELL signal - bearish MA alignment.")
            return SideSignal.SELL, 1

        # Default: HOLD
        self._config.log_info(f"[{self._datapipeline.symbol}] HOLD - MAs are neutral.")
        return SideSignal.HOLD, 0
