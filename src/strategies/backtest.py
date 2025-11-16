import pandas as pd
from typing import Callable, List, Dict
from config import make_logger
from ..alpaca_trader.order import SideSignal
from .fetch_price_data import fetch_data

logger = make_logger()


class Backtester:
    def __init__(
        self,
        symbol: str,
        days: int = 30,  
        initial_cash: float = 10000,
        position_size_pct: float = 0.95,
        test_mode: bool = False
    ):
        """
        Backtesting engine for a single symbol with batch fetching for long 1-min histories.
        """
        self.symbol = symbol
        self.days = days
        self.initial_cash = initial_cash
        self.position_size_pct = position_size_pct
        self.test_mode = test_mode

        self.bars = []
        remaining_days = self.days
        batch_size = 5  

        while remaining_days > 0:
            fetch_days = min(batch_size, remaining_days)
            logger.info(f"Fetching {fetch_days} days for {symbol}")
            batch_bars = fetch_data(symbol, limit=fetch_days * 390)  
            if not batch_bars:
                logger.warning(f"No data fetched for {symbol} in last batch")
                break
            self.bars.extend(batch_bars)
            remaining_days -= fetch_days

        if not self.bars:
            raise ValueError(f"No data fetched for {symbol}")

        logger.info(f"Total bars fetched for {symbol}: {len(self.bars)}")

    def calculate_quantity(self, signal: SideSignal, cash: float, position_qty: int, current_price: float) -> int:
        """Calculate quantity to trade based on signal and available capital."""
        if signal == SideSignal.BUY:
            max_value = cash * self.position_size_pct
            qty = int(max_value / current_price)
            return max(qty, 0)
        elif signal == SideSignal.SELL:
            return position_qty
        return 0

    def run_strategy(self, strategy_func: Callable, lookback: int = 20) -> pd.DataFrame:
        """Run a strategy and return portfolio development."""
        cash = self.initial_cash
        position_qty = 0
        position_avg_price = 0.0
        portfolio_values = []
        dates = []
        trades = []

        from . import fetch_price_data as fpd_module
        original_fetch = fpd_module.fetch_data

        def mock_fetch_price_data(symbol, limit=None):
            """Return cached bars up to current iteration."""
            return self._current_bars

        fpd_module.fetch_data = mock_fetch_price_data

        try:
            for i in range(lookback, len(self.bars)):
                bar = self.bars[i]
                date = bar["t"][:19]
                current_price = bar["c"]

                self._current_bars = self.bars[:i+1]

                position_data = {
                    "symbol": self.symbol,
                    "qty": position_qty,
                    "history": self.bars[:i+1]
                }

                try:
                    signal, qty = strategy_func(position_data)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Strategy {strategy_func.__name__} failed at {date}: {e}")
                    signal, qty = SideSignal.HOLD, 0

                if qty == 0 and signal != SideSignal.HOLD:
                    qty = self.calculate_quantity(signal, cash, position_qty, current_price)

                # BUY
                if signal == SideSignal.BUY and qty > 0 and cash >= current_price * qty:
                    cost = current_price * qty
                    position_avg_price = ((position_avg_price * position_qty) + cost) / (position_qty + qty) if position_qty > 0 else current_price
                    position_qty += qty
                    cash -= cost
                    trades.append({'date': date, 'action': 'BUY', 'qty': qty, 'price': current_price, 'cost': cost})
                    logger.info(f"{date}: BUY {qty} @ ${current_price:.2f}")

                # SELL
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
            fpd_module.fetch_data = original_fetch

        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        return pd.DataFrame({'date': dates, 'portfolio_value': portfolio_values})

    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        if results.empty or len(results) < 2:
            return {}

        final_value = results['portfolio_value'].iloc[-1]
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100

        results['daily_return'] = results['portfolio_value'].pct_change()
        sharpe_ratio = (results['daily_return'].mean() / results['daily_return'].std() * (252 ** 0.5)) if results['daily_return'].std() > 0 else 0.0

        cumulative_max = results['portfolio_value'].expanding().max()
        max_drawdown = ((results['portfolio_value'] - cumulative_max) / cumulative_max).min() * 100

        win_rate = 0.0
        if not self.trades.empty and 'SELL' in self.trades['action'].values:
            buy_trades = self.trades[self.trades['action'] == 'BUY']
            sell_trades = self.trades[self.trades['action'] == 'SELL']
            winning_trades = 0
            total_closed_trades = 0

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
            'num_trades': len(self.trades),
            'win_rate_pct': round(win_rate, 2)
        }


def compare_strategies(
    symbol: str,
    strategies: Dict[str, Callable],
    days: int = 30,
    initial_cash: float = 10000,
    test_mode: bool = False
) -> pd.DataFrame:
    results = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting backtest for {symbol} with {len(strategies)} strategies")
    logger.info(f"Days: {days}, Initial Cash: ${initial_cash}")
    logger.info(f"{'='*60}\n")

    for strategy_name, strategy_func in strategies.items():
        logger.info(f"Testing strategy: {strategy_name}")

        try:
            backtester = Backtester(symbol, days=days, initial_cash=initial_cash, test_mode=test_mode)
            portfolio_history = backtester.run_strategy(strategy_func)
            metrics = backtester.calculate_metrics(portfolio_history)

            metrics['strategy'] = strategy_name
            metrics['symbol'] = symbol
            results.append(metrics)

            logger.info(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%\n")

        except Exception as e:
            logger.error(f"Failed to run strategy {strategy_name}: {e}")
            results.append({'strategy': strategy_name, 'symbol': symbol, 'error': str(e)})

    comparison_df = pd.DataFrame(results)
    if 'total_return_pct' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('total_return_pct', ascending=False)

    logger.info(f"{'='*60}")
    logger.info("Backtest complete!")
    logger.info(f"{'='*60}\n")

    return comparison_df


def run_multi_symbol_backtest(
    symbols: List[str],
    strategies: Dict[str, Callable],
    days: int = 30,
    initial_cash: float = 10000,
    test_mode: bool = False
) -> pd.DataFrame:
    all_results = []

    logger.info(f"\n{'#'*60}")
    logger.info(f"Multi-Symbol Backtest: {len(symbols)} symbols, {len(strategies)} strategies")
    logger.info(f"{'#'*60}\n")

    for symbol in symbols:
        logger.info(f"Processing: {symbol}")
        try:
            symbol_results = compare_strategies(symbol, strategies, days, initial_cash, test_mode=test_mode)
            all_results.append(symbol_results)
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"\n{'#'*60}")
        logger.info(f"Multi-Symbol Backtest Complete!")
        logger.info(f"Total results: {len(combined)} strategy-symbol combinations")
        logger.info(f"{'#'*60}\n")
        return combined

    return pd.DataFrame()
