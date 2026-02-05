from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy
from typing import Tuple, List

class RSIStrategy(BaseStrategy):
    @staticmethod
    def _calculate_rsi(closes: List[float], period: int = 14) -> float:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            closes (List[float]):
                A list of close prices ordered from oldest to newest.
            period (int):
                The RSI lookback period. Default is 14.

        Returns:
            float:
                The computed RSI value (0-100).
        """
        gains, losses = [], []

        for i in range(1, period + 1):
            change = closes[-i] - closes[-(i + 1)]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 1e-10
        rs = avg_gain / avg_loss

        return 100 - (100 / (1 + rs))


    def _run(self) -> Tuple[SideSignal, int]:
        """
        A simple RSI-based trading strategy.

        Generates signals based on RSI thresholds:
        - Buy when RSI < 30
        - Sell when RSI > 70
        - Hold otherwise

        Args:
            symbol (str): The symbol of the stock we want to calculate for.
            position_data (Dict[str, Any]):
                A dictionary containing:
                - "symbol": The ticker symbol.
                - "history": Historical bar data (various formats supported).

        Returns:
            Tuple[SideSignal, int]:
                A tuple where:
                - The first value is a SideSignal (BUY, SELL, HOLD)
                - The second value is the quantity (always 0 here)
        """

        if self._datapipeline.data is None:
            self._config.log_error(f"[RSI] Unable to fetch data for {self._datapipeline.symbol}.")
            return SideSignal.HOLD, 0

        if len(self._datapipeline.data) < 15:
            self._config.log_info(f"[RSI] Not enough historical bars to compute RSI for {self._datapipeline.symbol}.")
            return SideSignal.HOLD, 0

        closes = self._datapipeline.data["close"].astype(float).tolist()
        rsi = self._calculate_rsi(closes[-15:])

        self._config.log_info(f"[{self._datapipeline.symbol}] RSI: {rsi:.2f}")

        if rsi < 30:
            self._config.log_info(f"[{self._datapipeline.symbol}] Buy signal triggered.")
            return SideSignal.BUY, 1

        if rsi > 70:
            self._config.log_info(f"[{self._datapipeline.symbol}] Sell signal triggered.")
            return SideSignal.SELL, 1

        self._config.log_info(f"[{self._datapipeline.symbol}] Hold signal (RSI neutral).")
        return SideSignal.HOLD, 0
