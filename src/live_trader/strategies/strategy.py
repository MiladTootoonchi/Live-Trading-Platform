from live_trader.alpaca_trader.order import SideSignal
from live_trader.config import Config
from live_trader.strategies.data import MarketDataPipeline
from abc import ABC, abstractmethod
import pandas as pd
import inspect


class BaseStrategy(ABC):
    def __init__(self, config: Config):
        self._config = config
        self._datapipeline = None

    def prepare_data(self, symbol: str, position_data: dict | None = None):
        self._datapipeline = MarketDataPipeline(self._config, symbol, position_data or {})

    async def run_backtest(
        self,
        symbol: str,
        initial_cash: float = 10_000,
        position_size_pct: float = 0.95,
    ) -> pd.DataFrame:

        # --- initialize pipeline ---
        self.prepare_data(symbol, position_data={})
        pipeline = self._datapipeline
        df = pipeline.data

        lookback = pipeline.lookback
        if len(df) <= lookback:
            raise RuntimeError("Not enough data for backtest")

        cash = initial_cash
        position_qty = 0
        avg_entry_price = 0.0

        equity = []
        dates = []
        trades = []

        for i in range(lookback, len(df)):
            # --- advance time ---
            pipeline.load_slice(i)

            current_price = float(pipeline.df.iloc[-1]["close"])
            date = str(pipeline.df.index[-1])


            position_data = {
                "qty": position_qty,
                "avg_entry_price": avg_entry_price,
                "market_price": current_price,
                "backtest": True,
            }

            pipeline.update_position_data(position_data)


            result = self.run()
            if inspect.isawaitable(result):
                result = await result

            signal, qty = result

            # ---- default sizing (non-ML strategies) ----
            if qty == 0 and signal == SideSignal.BUY:
                qty = int((cash * position_size_pct) / current_price)

            # ---- execute trades ----
            if signal == SideSignal.BUY and qty > 0 and cash >= qty * current_price:
                cost = qty * current_price
                avg_entry_price = (
                    (avg_entry_price * position_qty + cost) / (position_qty + qty)
                    if position_qty > 0 else current_price
                )
                position_qty += qty
                cash -= cost
                trades.append((date, "BUY", qty, current_price))

            elif signal == SideSignal.SELL and qty > 0 and position_qty > 0:
                qty = min(qty, position_qty)
                cash += qty * current_price
                position_qty -= qty
                if position_qty == 0:
                    avg_entry_price = 0.0
                trades.append((date, "SELL", qty, current_price))

            total_value = cash + position_qty * current_price
            equity.append(total_value)
            dates.append(date)

        self.trades = pd.DataFrame(
            trades, columns=["date", "action", "qty", "price"]
        )

        return pd.DataFrame(
            {"date": dates, "portfolio_value": equity}
        )


class RuleBasedStrategy(BaseStrategy):
    def _run(self) -> tuple[SideSignal, int]:
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