from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import inspect

from live_trader.config import Config
from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.strategy import BaseStrategy


class Backtester:
    """
    Single-symbol backtester using full-bar format.

    Args:
        symbol: ticker to backtest
        initial_cash: starting cash
        position_size_pct: fraction of available cash to use when buying
    """

    def __init__(
        self,
        config: Config,
        strategy: BaseStrategy,
        initial_cash: float = 10000,
        position_size_pct: float = 0.95,
    ):
        self._config = config

        self.strategy = strategy

        self.initial_cash = initial_cash
        self.position_size_pct = position_size_pct
        self._strategy_state = {}

        self._min_lookback = config.load_min_lookback()

        self._key, self._secret = config.load_keys()

        self.bars = self._df_to_full_bars(self.strategy.data.data)
        self.symbol = strategy.data.symbol


    def calculate_quantity(self, signal: SideSignal, cash: float, position_qty: int, current_price: float) -> int:
        """
        Calculate trade quantity.

        For buy, use position_size_pct of cash; for sell, return current holding qty.
        """
        if signal == SideSignal.BUY:
            max_value = cash * self.position_size_pct
            try:
                qty = int(max_value / current_price)
            except Exception:
                qty = 0
            return max(qty, 0)
        if signal == SideSignal.SELL:
            return position_qty
        return 0

    async def run_strategy(self, lookback: int) -> pd.DataFrame:
        total_days = len(self.bars)
        warmup_days = lookback
        test_days = total_days - lookback

        start_date = pd.to_datetime(self.bars[lookback]["t"]).date()
        end_date = pd.to_datetime(self.bars[-1]["t"]).date()

        self._config.log_info(
            f"""
            Backtest period for {self.symbol}
            --------------------------------
            Total trading days : {total_days}
            Warmup days        : {warmup_days}
            Test days          : {test_days}
            Test start date    : {start_date}
            Test end date      : {end_date}
            """
        )

        if lookback < self._min_lookback:
            raise ValueError(
                f"[Backtester] lookback={lookback} too small for ML strategies "
                f"(min {self._min_lookback})"
            )

        if len(self.bars) <= lookback:
            raise ValueError(
                f"[Backtester] Not enough data: self.bars={len(self.bars)}, lookback={lookback}"
            )

        cash = self.initial_cash
        position_qty = 0
        position_avg_price = 0.0
        portfolio_values: List[float] = []
        dates: List[str] = []
        trades: List[Dict[str, Any]] = []

        # iterate through time steps
        for i in range(lookback, len(self.bars)):
            bar = self.bars[i]
            date = bar.get("t", "")[:19]
            current_price = float(bar.get("c", 0.0))

            # history available to strategy: all self.bars up to and including this index
            self._current_bars = self.bars[: i + 1]

            position_data: Dict[str, Any] = {
                "symbol": self.symbol,
                "qty": position_qty,
                "history": list(self._current_bars),
                "avg_entry_price": position_avg_price,
                "current_price": current_price,
                "backtest": True
            }


            state = self._strategy_state.setdefault(self.strategy, {})
            position_data["state"] = state

            try:
                result = self.strategy.run()
                
                if inspect.iscoroutine(result):
                    result = await result

                if (not isinstance(result, tuple) or len(result) != 2):
                    raise ValueError(
                        f"{self.strategy.__class__.__name__} must return (signal, qty), got {result}"
                    )

                signal, qty = result

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._config.log_error(
                    f"Strategy {self.strategy.__class__.__name__} failed at {date}: {e}"
                )
                signal, qty = SideSignal.HOLD, 0


            # If strategy returned no quantity, calculate a sensible one
            if qty == 0 and signal != SideSignal.HOLD:
                qty = self.calculate_quantity(signal, cash, position_qty, current_price)

            # Execute buy
            if signal == SideSignal.BUY and qty > 0 and cash >= current_price * qty:
                cost = current_price * qty
                position_avg_price = ((position_avg_price * position_qty) + cost) / (position_qty + qty) if position_qty > 0 else current_price
                position_qty += qty
                cash -= cost
                trades.append({'date': date, 'action': 'BUY', 'qty': qty, 'price': current_price, 'cost': cost})
                self._config.log_info(f"{date}: BUY {qty} @ ${current_price:.2f}")

            # Execute sell
            elif signal == SideSignal.SELL and qty > 0 and position_qty > 0:
                qty = min(qty, position_qty)
                proceeds = current_price * qty
                position_qty -= qty
                cash += proceeds
                if position_qty == 0:
                    position_avg_price = 0.0
                trades.append({'date': date, 'action': 'SELL', 'qty': qty, 'price': current_price, 'proceeds': proceeds})
                self._config.log_info(f"{date}: SELL {qty} @ ${current_price:.2f}")

            total_value = cash + position_qty * current_price
            portfolio_values.append(total_value)
            dates.append(date)

        # store trades as DataFrame for metric calculations
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()

        return pd.DataFrame({'date': dates, 'portfolio_value': portfolio_values})


    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from the portfolio time series.

        Returns a dictionary with:
            total_return_pct, final_value, sharpe_ratio, max_drawdown_pct, num_trades, win_rate_pct
        """
        if results.empty or len(results) < 2:
            return {}

        final_value = results['portfolio_value'].iloc[-1]
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100

        # daily returns based on the value series
        results = results.copy()
        results['daily_return'] = results['portfolio_value'].pct_change().fillna(0.0)

        mean = results['daily_return'].mean()
        std = results['daily_return'].std()
        sharpe_ratio = (mean / std * (252 ** 0.5)) if std > 0 else 0.0

        cumulative_max = results['portfolio_value'].expanding().max()
        max_drawdown = ((results['portfolio_value'] - cumulative_max) / cumulative_max).min() * 100

        win_rate = 0.0
        if not self.trades.empty and 'SELL' in self.trades['action'].values:
            buy_trades = self.trades[self.trades['action'] == 'BUY']
            sell_trades = self.trades[self.trades['action'] == 'SELL']
            winning_trades = 0
            total_closed_trades = 0

            # pair sells with latest prior buy and judge P&L
            for _, sell in sell_trades.iterrows():
                prior_buys = buy_trades[buy_trades['date'] < sell['date']]
                if not prior_buys.empty:
                    last_buy_price = prior_buys.iloc[-1]['price']
                    if sell['price'] > last_buy_price:
                        winning_trades += 1
                    total_closed_trades += 1

            if total_closed_trades > 0:
                win_rate = (winning_trades / total_closed_trades) * 100

        return {
            'total_return_pct': round(total_return, 2),
            'final_value': round(final_value, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'num_trades': int(len(self.trades)),
            'win_rate_pct': round(win_rate, 2)
        }

    @staticmethod    
    def _df_to_full_bars(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert a DataFrame returned by your fetch_data into the full-bar list.

        The function accepts a variety of column names and normalizes them into:
        t (ISO timestamp string), o, h, l, c, v

        Args:
            df: pandas.DataFrame with OHLCV data and a DatetimeIndex or timestamp column.

        Returns:
            List of dicts with full bar fields.
        """
        bars = []

        for idx, row in df.iterrows():
            bar = {
                "t": str(idx),
                "o": row.get("open"),
                "h": row.get("high"),
                "l": row.get("low"),
                "c": row.get("close"),
                "v": row.get("volume"),
            }

            # PASS THROUGH — do not alter
            if "trade_count" in row:
                bar["trade_count"] = row["trade_count"]

            if "vwap" in row:
                bar["vwap"] = row["vwap"]

            bars.append(bar)

        return bars
