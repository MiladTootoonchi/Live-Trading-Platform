from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy
from typing import Tuple, List

class MACDStrategy(BaseStrategy):
    @staticmethod
    def _exponential_moving_average(data: List[float], period: int) -> List[float]:
        """
        Computes an exponential moving average.

        Args:
            data (list of float):
                Closing prices used for EMA.
            period (int):
                EMA period length.

        Returns:
            list of float:
                EMA values.
        """
        if not data or period <= 0:
            return []

        ema = []
        k = 2 / (period + 1)

        for i, price in enumerate(data):
            if i == 0:
                ema.append(price)
            else:
                ema.append(price * k + ema[i - 1] * (1 - k))

        return ema

    def _calculate_macd(self, closes: List[float]) -> Tuple[List[float], List[float]]:
        """
        Calculates MACD and signal lines.

        Args:
            closes (list of float):
                Close prices in chronological order.

        Returns:
            tuple(list, list):
                MACD line and signal line.
        """
        ema12 = self._exponential_moving_average(closes, 12)
        ema26 = self._exponential_moving_average(closes, 26)

        min_len = min(len(ema12), len(ema26))
        ema12, ema26 = ema12[-min_len:], ema26[-min_len:]

        macd_line = [a - b for a, b in zip(ema12, ema26)]
        signal_line = self._exponential_moving_average(macd_line, 9)

        return macd_line, signal_line

    def _run(self) -> Tuple[SideSignal, int]:
        """
        MACD crossover strategy.
        Generates buy or sell signals when MACD crosses the signal line.

        Returns:
            tuple(SideSignal, int):
                Signal and quantity (always 0 for risk management).
        """
        bars = self._datapipeline.data
        symbol = self._datapipeline.symbol

        if bars.empty:
            return SideSignal.HOLD, 0


        if len(bars) < 35:
            self._config.log_info(f"Not enough data to compute MACD for {symbol}. Need 35 bars.")
            return SideSignal.HOLD, 0

        closes = bars["close"].astype(float).tolist() # this can go inside normalize_bars

        macd_line, signal_line = self._calculate_macd(closes)

        if len(macd_line) < 2 or len(signal_line) < 2:
            self._config.log_info(f"Insufficient MACD data for {symbol}.")
            return SideSignal.HOLD, 0

        prev_macd, curr_macd = macd_line[-2], macd_line[-1]
        prev_signal, curr_signal = signal_line[-2], signal_line[-1]

        # Bullish crossover
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            self._config.log_info(f"[{symbol}] BUY signal — MACD bullish crossover.")
            return SideSignal.BUY, 1

        # Bearish crossover
        if prev_macd >= prev_signal and curr_macd < curr_signal:
            self._config.log_info(f"[{symbol}] SELL signal — MACD bearish crossover.")
            return SideSignal.SELL, 1

        self._config.log_info(f"[{symbol}] HOLD — no MACD crossover.")
        return SideSignal.HOLD, 0
