from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy
from typing import Tuple

class BollingerBandsStrategy(BaseStrategy):
    def bollinger_bands_strategy(self) -> Tuple[SideSignal, int]:
        """
        Bollinger Bands mean reversion strategy.
        Generates a signal when price moves above or below
        the standard deviation bands around a moving average.

        Args:
            symbol (str): The symbol of the stock we want to calculate for.
            position (dict):
                Contains:
                    symbol (str): the ticker symbol
                    history (list or DataFrame): optional price history
                    current_price (float): fallback price

        Returns:
            tuple(SideSignal, int):
                The trade signal and quantity (always zero for safety).
        """
        bars = self._datapipeline.data
        symbol = self._datapipeline.symbol

        if bars.empty:
            return SideSignal.HOLD, 0


        if len(bars) < 20:
            self._config.log_warning(f"Not enough data for {symbol}. Need twenty bars minimum.")
            return SideSignal.HOLD, 0

        closes = bars["close"].astype(float).tolist()

        # Compute simple moving average
        sma20 = sum(closes[-20:]) / 20

        # Compute standard deviation
        variance = sum((p - sma20) ** 2 for p in closes[-20:]) / 20
        stddev = variance ** 0.5

        # Compute upper and lower bands
        upper_band = sma20 + (2 * stddev)
        lower_band = sma20 - (2 * stddev)

        current_price = closes[-1]

        self._config.log_info(
            f"[{symbol}] Price {current_price:.2f}, SMA20 {sma20:.2f}, "
            f"Upper {upper_band:.2f}, Lower {lower_band:.2f}"
        )

        # Price below the lower band
        if current_price < lower_band:
            self._config.log_info(f"[{symbol}] BUY signal triggered. Price below lower band.")
            return SideSignal.BUY, 1

        # Price above the upper band
        if current_price > upper_band:
            self._config.log_info(f"[{symbol}] SELL signal triggered. Price above upper band.")
            return SideSignal.SELL, 1

        # Price in the middle band range
        self._config.log_info(f"[{symbol}] HOLD. Price within the Bollinger range.")
        return SideSignal.HOLD, 0
