"""
Backtesting engine (single-symbol).

This backtester fetches historical bars via utils.fetch_data, converts them to a
consistent full-bar format, and then runs each strategy step-by-step while
monkey-patching fetch_data so strategies see only the history up to the current
time step.

Bar format used across the engine and injected into strategies:
{
    "t": "<ISO timestamp string>",
    "o": float,
    "h": float,
    "l": float,
    "c": float,
    "v": int
}

Args / Returns: see individual functions and public compare_strategies.
"""
from __future__ import annotations

from typing import Callable, List, Dict, Any
import pandas as pd
import inspect

from live_trader.config import make_logger
from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.utils import fetch_data

logger = make_logger()



class Backtester:
    """
    Single-symbol backtester using full-bar format.

    Args:
        symbol: ticker to backtest
        days: approximate number of days to fetch (best-effort)
        initial_cash: starting cash
        position_size_pct: fraction of available cash to use when buying
        test_mode: currently unused, included for compatibility

    Usage:
        backtester = Backtester("TSLA", days=30)
        history = backtester.run_strategy(my_strategy)
        metrics = backtester.calculate_metrics(history)
    """

    def __init__(
        self,
        symbol: str,
        days: int = 30,
        initial_cash: float = 10000,
        position_size_pct: float = 0.95,
        test_mode: bool = False,
    ):
        self.symbol = symbol
        self.days = days
        self.initial_cash = initial_cash
        self.position_size_pct = position_size_pct
        self.test_mode = test_mode
        self._strategy_state = {}

        self.bars: List[Dict[str, Any]] = []

        df = fetch_data(symbol)

        if isinstance(df, pd.DataFrame):
            self.bars = self._df_to_full_bars(df)
        else:
            self.bars = df

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

    async def run_strategy(self, strategy_func: Callable, lookback: int = 50) -> pd.DataFrame:
        """
        Run the strategy step-by-step and produce a time series of portfolio value.

        Args:
            strategy_func: function(position_data) -> (SideSignal, int)
            lookback: initial warm-up bars to skip so indicators have enough history

        Returns:
            pandas.DataFrame with columns ['date', 'portfolio_value']
        """
        cash = self.initial_cash
        position_qty = 0
        position_avg_price = 0.0
        portfolio_values: List[float] = []
        dates: List[str] = []
        trades: List[Dict[str, Any]] = []

        # Monkey patch utils.fetch_data so strategies that call it get the current slice
        import live_trader.strategies.utils as strat_utils
        import live_trader.ml_model.training as ml_training

        original_fetch_strat = strat_utils.fetch_data
        original_fetch_ml = ml_training.fetch_data

        def _mock_fetch(symbol: str, *args, **kwargs):
            df = pd.DataFrame(self._current_bars)

            for col in ["trade_count", "vwap"]:
                if col not in df.columns:
                    df[col] = pd.NA

            if "t" in df.columns:
                # t is always (symbol, pandas.Timestamp)
                def extract_ts(t):
                    if isinstance(t, tuple):
                        return t[-1]
                    return t

                df["timestamp"] = df["t"].apply(extract_ts)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"])

                # enforce clean tz-aware DatetimeIndex
                df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")

                df = df.rename(columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                })

                df = df.drop(columns=["t"])

            else:
                raise RuntimeError("_mock_fetch: missing timestamp column")


            # Set clean DatetimeIndex
            df = df.set_index("timestamp")
            df.index = pd.DatetimeIndex(df.index)

            # Final safety check
            if not isinstance(df.index, pd.DatetimeIndex):
                raise RuntimeError(
                    f"_mock_fetch produced invalid index: {type(df.index)}"
                )

            return df

        strat_utils.fetch_data = _mock_fetch
        ml_training.fetch_data = _mock_fetch


        try:
            # iterate through time steps
            for i in range(lookback, len(self.bars)):
                bar = self.bars[i]
                date = bar.get("t", "")[:19]
                current_price = float(bar.get("c", 0.0))

                # history available to strategy: all bars up to and including this index
                self._current_bars = self.bars[: i + 1]

                position_data: Dict[str, Any] = {
                    "symbol": self.symbol,
                    "qty": position_qty,
                    "history": list(self._current_bars),
                    "avg_entry_price": position_avg_price,
                    "current_price": current_price,
                    "backtest": True
                }


                state = self._strategy_state.setdefault(strategy_func, {})
                position_data["state"] = state

                try:
                    result = strategy_func(self.symbol, position_data)
                    
                    if inspect.iscoroutine(result):
                        result = await result

                    if (not isinstance(result, tuple) or len(result) != 2):
                        raise ValueError(
                            f"{strategy_func.__name__} must return (signal, qty), got {result}"
                        )

                    signal, qty = result

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(
                        f"Strategy {getattr(strategy_func, '__name__', '<unknown>')} failed at {date}: {e}"
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
                    logger.info(f"{date}: BUY {qty} @ ${current_price:.2f}")

                # Execute sell
                elif signal == SideSignal.SELL and qty > 0 and position_qty > 0:
                    qty = min(qty, position_qty)
                    proceeds = current_price * qty
                    position_qty -= qty
                    cash += proceeds
                    if position_qty == 0:
                        position_avg_price = 0.0
                    trades.append({'date': date, 'action': 'SELL', 'qty': qty, 'price': current_price, 'proceeds': proceeds})
                    logger.info(f"{date}: SELL {qty} @ ${current_price:.2f}")

                total_value = cash + position_qty * current_price
                portfolio_values.append(total_value)
                dates.append(date)

        finally:
            strat_utils.fetch_data = original_fetch_strat
            ml_training.fetch_data = original_fetch_ml

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

    def _df_to_full_bars(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
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

async def _compare_strategies(
    symbol: str,
    strategies: Dict[str, Callable],
    days: int = 80,
    initial_cash: float = 10000,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Run multiple strategies against the same symbol and return a comparison DataFrame.

    Args:
        symbol: ticker symbol to evaluate
        strategies: mapping of strategy name -> callable function
        days: approximate days to fetch (unused with current fetch_data signature)
        initial_cash: starting cash for each run
        test_mode: placeholder for future options

    Returns:
        pandas.DataFrame with columns including strategy, symbol and performance metrics
    """
    results = []

    logger.info("\n" + "=" * 60)
    logger.info(f"Starting backtest for {symbol} with {len(strategies)} strategies")
    logger.info(f"Days: {days}, Initial Cash: ${initial_cash}")
    logger.info("=" * 60 + "\n")

    for name, func in strategies.items():
        logger.info(f"Testing strategy: {name}")
        try:
            backtester = Backtester(
                symbol,
                days=days,
                initial_cash=initial_cash,
                test_mode=test_mode,
            )

            history = await backtester.run_strategy(func, lookback = 50)
            metrics = backtester.calculate_metrics(history)

            # attach metadata
            if metrics:
                metrics['strategy'] = name
                metrics['symbol'] = symbol
            else:
                metrics = {'strategy': name, 'symbol': symbol, 'error': 'no results or insufficient data'}

            results.append(metrics)

            if 'total_return_pct' in metrics:
                logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
                logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
            else:
                logger.info(f"  Strategy {name} produced no metrics.\n")

        except Exception as e:
            logger.exception(f"Failed to run strategy {name}: {e}")
            results.append({'strategy': name, 'symbol': symbol, 'error': str(e)})

    df = pd.DataFrame(results)
    if 'total_return_pct' in df.columns:
        df = df.sort_values('total_return_pct', ascending=False)

    logger.info("=" * 60)
    logger.info("Backtest complete!")
    logger.info("=" * 60 + "\n")

    return df


async def run_multi_symbol_backtest(
    symbols: List[str],
    strategies: Dict[str, Callable],
    days: int = 30,
    initial_cash: float = 10000,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Convenience wrapper to run compare_strategies across multiple symbols.
    """
    all_results = []
    for s in symbols:
        try:
            res = await _compare_strategies(s, strategies, days=days, initial_cash=initial_cash, test_mode=test_mode)
            all_results.append(res)
        except Exception as e:
            logger.exception(f"Failed backtest for {s}: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
