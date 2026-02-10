import requests
import asyncio
from typing import Callable, List, Dict, Any
import inspect
import pandas as pd

from live_trader.alpaca_trader.order import OrderData, SideSignal
from live_trader.config import Config

""" Strategies """
from live_trader.strategies.strategy import RuleBasedStrategy, BaseStrategy
from live_trader.strategies.bollinger_bands_strategy import BollingerBandsStrategy
from live_trader.strategies.macd import MACDStrategy
from live_trader.strategies.mean_reversion import MeanReversionStrategy
from live_trader.strategies.momentum import MomentumStrategy
from live_trader.strategies.moving_average_strategy import MovingAverageStrategy
from live_trader.strategies.rsi import RSIStrategy
from live_trader.ml_model.ml_strategies import (LSTMModel, BiLSTM, TCN, PatchTST, GNN, NAD, CNNGRU)
from live_trader.tree_based_models.tree_strategies import (XGB, CatBoost, RandomForest, LGBM)

from live_trader.strategies.backtest import Backtester


STRATEGIES = {
        "rule_based_strategy": RuleBasedStrategy,
        "bollinger_bands_strategy": BollingerBandsStrategy,
        "macd_strategy": MACDStrategy,
        "mean_reversion_strategy": MeanReversionStrategy,
        "momentum_strategy": MomentumStrategy,
        "moving_average_strategy": MovingAverageStrategy,
        "rsi_strategy": RSIStrategy,
        "lstm": LSTMModel,
        "bilstm": BiLSTM,
        "tcn": TCN,
        "patchtst": PatchTST,
        "gnn": GNN,
        "nad": NAD,
        "cnn_gru": CNNGRU,
        "random_forest": RandomForest,
        "lightgbm": LGBM,
        "xgboost": XGB,
        "catboost": CatBoost,
    }

class AlpacaTrader:
    """
    A class to handle the Alpaca trading given API keys and the order data.

    This class will create market orders that will either buy or sell positions,
    Get the order object from Alpaca and
    Get all the posistions.
    Get orders from a strategy and send it.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the trader.
        
        Args:
            key: The key given by Alpaca
            secret_key: The secret key alpaca key
        """

        self._config = config

        key, secret = config.load_keys()

        self._HEADERS = {
            "APCA-API-KEY-ID" : key,
            "APCA-API-SECRET-KEY": secret,
        }

        self._APCA_API_BASE_URL =  config.apca_url

        self._strategy = self._find_strategy()



    async def get_orders(self) -> List[Dict]:
        """
        Retrieves and returns a list of recent order objects from the trading account.

        This method sends an asynchronous request to the Alpaca API to fetch order data.
        Each order dictionary in the returned list contains details such as the order ID,
        stock symbol, quantity, side (buy/sell), current status, and the timestamp of order creation.

        Returns:
            list[dict]: A list of dictionaries, each representing an individual order.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.get, url, headers = self._HEADERS)
        return response.json()

    async def get_positions(self) -> List[Dict]:
        """
        Retrieves and returns all current account positions.

        This method sends an asynchronous request to the Alpaca API to fetch the user's
        current stock positions. Each position in the returned list includes details
        such as the stock symbol, quantity held, current market value, and unrealized 
        profit or loss.

        Returns:
            list[dict]: A list of dictionaries representing each position.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/positions"
        response = await asyncio.to_thread(requests.get, url, headers = self._HEADERS)
        return response.json()
    
    
    async def is_market_open(self) -> bool:
        """
        Checks whether the US equity market is currently open according to Alpaca.

        This method queries Alpaca's `/v2/clock` endpoint, which provides the
        current market timestamp and open/close status. The request is executed
        asynchronously using a background thread to avoid blocking the event loop.

        If the API request fails or returns an unexpected response, the method
        logs the error and returns False by default to ensure trading safety.

        Returns:
            bool:
                True if the market is currently open, False otherwise.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/clock"

        try:
            response = await asyncio.to_thread(
                requests.get, url, headers=self._HEADERS, timeout=10
            )

            if response.status_code != 200:
                self._config.log_error(
                    f"Failed to fetch market clock: {response.status_code} - {response.text}"
                )
                return False

            data: Dict[str, bool] = response.json()
            is_open: bool = bool(data.get("is_open", False))

            self._config.log_debug(f"Market open status: {is_open}")
            return is_open

        except Exception:
            self._config.log_expectation("Error while checking market open status")
            return False



    async def place_order(self, order_data: OrderData) -> None:
        """
        Submits an order to buy or sell assets based on the provided order data.

        Sends a POST request to the Alpaca API with the order details encapsulated
        in the 'OrderData' object. The order can represent either a buy or sell
        action, including specifics such as symbol, quantity, and order type.

        Args:
            order_data (OrderData): An object containing all necessary fields to
                                    execute a trade order (e.g., symbol, quantity,
                                    side, type).
        """
        data = order_data.get_dict()

        ORDERS_URL = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.post, ORDERS_URL, json = data, headers = self._HEADERS)

        self._config.log_info(f"Sending Order Data:\n {data} \n Headers: {self._HEADERS}\n")

        response_json = response.json()

        if "id" not in response_json:
            self._config.log_error(f"Order rejected: {response_json} \n")
            return

        order_id = response_json["id"]
        await self._wait_until_order_filled(order_id)

    async def close_position(self, symbol: str) -> None:
        """
        Closes (liquidates) the entire open position for a given stock symbol.

        This method sends a request to Alpaca's position liquidation endpoint to
        close all shares held for the specified symbol at market price. The
        operation is executed asynchronously using a background thread to avoid
        blocking the event loop.

        If the position is successfully closed, an informational log message is
        recorded. If the request fails (e.g., the position does not exist or the
        API returns an error), an error message containing the response details
        is logged.

        Args:
            symbol (str): The stock symbol whose entire position should be closed.
        """
        
        url = f"{self._APCA_API_BASE_URL}/v2/positions/{symbol}"
        response = await asyncio.to_thread(
            requests.delete, url, headers=self._HEADERS
        )

        if response.status_code == 200:
            self._config.log_info(f"Successfully closed entire position for {symbol}.")
        else:
            self._config.log_error(f"Failed to close position for {symbol}: {response.text}")
    
    async def cancel_all_orders(self) -> None:
        """
        Cancels all open orders for the account.

        Sends a request to the Alpaca API to delete all active orders.
        Upon completion, prints the response content for confirmation or debugging purposes.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.delete, url, headers = self._HEADERS)
        self._config.log_info(f"Cancel response: {response.content}\n")

    async def cancel_last_order(self) -> None:
        """
        Cancels the most recently submitted open order.

        This method retrieves recent orders from the Alpaca API, identifies the
        most recent order that is still cancelable (e.g., 'new', 'accepted',
        or 'partially_filled'), and submits a cancellation request for that order.
        If no such order exists, the method logs an informational message and exits.
        """

        # Fetch recent orders (most recent first)
        url = f"{self._APCA_API_BASE_URL}/v2/orders?limit=10&direction=desc"
        response = await asyncio.to_thread(requests.get, url, headers=self._HEADERS)

        if response.status_code != 200:
            self._config.log_error(f"Failed to fetch orders: {response.text}\n")
            return

        orders = response.json()

        cancelable_statuses = {"new", "accepted", "partially_filled"}

        for order in orders:
            if order.get("status") in cancelable_statuses:
                order_id = order["id"]
                symbol = order.get("symbol")

                cancel_url = f"{self._APCA_API_BASE_URL}/v2/orders/{order_id}"
                cancel_response = await asyncio.to_thread(
                    requests.delete, cancel_url, headers=self._HEADERS
                )

                if cancel_response.status_code in (200, 204):
                    self._config.log_info(
                        f"Canceled last open order {order_id} for {symbol}.\n"
                    )
                else:
                    self._config.log_error(
                        f"Failed to cancel order {order_id}: {cancel_response.text}\n"
                    )
                return

        self._config.log_info("No open orders found to cancel.\n")
    


    async def create_buy_order(self) -> None:
        """
        Prompts the user for input to create and submit a buy order.

        This method interactively gathers required details from the user,
        including the stock symbol, quantity, and order type. It validates
        the stock symbol via an external API to ensure the asset is tradable,
        checks that the quantity is a positive integer, and confirms the order
        type is one of the supported options. Once all inputs are validated,
        it constructs an OrderData object and submits the order using place_order().
        """

        # Asking for stock
        while True:
            symbol = input("Which stock do you want to buy (symbol)? ").strip().upper()

            url = f"{self._APCA_API_BASE_URL}/v2/assets/{symbol}"
            response = requests.get(url, headers = self._HEADERS)

            if response.status_code == 200:
                data = response.json()
                if data.get("tradable", False):
                    break
                else:
                    print(f"The symbol '{symbol}' is not valid. Please try another.")
            else:
                print(f"The symbol '{symbol}' is not valid or not found. Please try again.")

        # Asking for quantity
        while True:
            try:
                qty = int(input("How much do you want to buy (quantity)? "))
                if qty > 0:
                    break
                else:
                    print("Quantity must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")


        # Asking for order type
        valid_order_types = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop']

        while True:
            order_type = input("What type of order do you want (e.g. market, limit, stop, stop_limit, trailing_stop)? ").lower()
            if order_type in valid_order_types:
                break

            print("Invalid order type. Please enter one of the following:", ", ".join(valid_order_types))


        order = OrderData(symbol = symbol, quantity = qty, side = "buy", type = order_type)
        await self.place_order(order)

    async def create_sell_order(self) -> None:
        """
        Interactively creates and submits a sell order or closes an entire position.

        This method prompts the user to specify a stock symbol to sell and validates
        that the asset exists and is tradable via the Alpaca API. The user is then
        asked to provide either a positive integer quantity to sell or the keyword
        'all' to liquidate the entire position for the selected symbol.

        If 'all' is provided, the method closes the full position using Alpaca's
        position liquidation endpoint, bypassing order creation and submitting
        a market close for all shares held. If a quantity is provided, the method
        continues by prompting for a valid order type, constructs an OrderData
        object, and submits the order using the place_order() method.

        The method handles input validation, ensures asynchronous API safety, and
        exits early when a full position close is requested.
        """

        # Asking for stock
        while True:
            symbol = input("Which stock do you want to sell (symbol)? ").strip().upper()

            url = f"{self._APCA_API_BASE_URL}/v2/assets/{symbol}"
            response = requests.get(url, headers = self._HEADERS)

            if response.status_code == 200:
                data = response.json()
                if data.get("tradable", False):
                    break
                else:
                    print(f"The symbol '{symbol}' is not valid. Please try another.")
            else:
                print(f"The symbol '{symbol}' is not valid or not found. Please try again.")

        # Asking for quantity
        while True:
            qty_input = input(
                "How much do you want to sell (quantity or 'all' to sell entire position)? "
            ).strip().lower()

            if qty_input == "all":
                await self.close_position(symbol)
                return

            try:
                qty = int(qty_input)
                if qty > 0:
                    break
                else:
                    print("Quantity must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer or 'all'.")


        # Asking for order type
        valid_order_types = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop']

        while True:
            order_type = input("What type of order do you want (e.g. market, limit, stop, stop_limit, trailing_stop)? ").lower()
            if order_type in valid_order_types:
                break

            print("Invalid order type. Please enter one of the following:", ", ".join(valid_order_types))


        order = OrderData(symbol = symbol, quantity = qty, side = "sell", type = order_type)
        await self.place_order(order)



    async def _wait_until_order_filled(self, order_id: str) -> None:
        """
        Asynchronously waits until a specific order is marked as 'filled'.

        This method gets the order status from the Alpaca API 
        for the given 'order_id'. It checks the status every minute by 
        making a GET request to the order endpoint. Once the order is detected 
        as 'filled'.

        Args:
            order_id (str): The ID of the order to monitor.
            seconds (int): How many 
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders/{order_id}"
        while True:
            response = await asyncio.to_thread(requests.get, url, headers=self._HEADERS)
            order = response.json()

            if order.get("status") == "filled":
                self._config.log_info(f"Order {order_id} filled for {order['symbol']}. \n")
                return
            print(f"Waiting for order {order_id} to fill...")   # this is just temporary info, does not need to log this
            await asyncio.sleep(60) # Sleep for a minute



    async def update(self, symbol: str) -> None:
        """
        Updates one or all positions based on the provided trading strategy.

        The strategy function should accept (symbol, position) and
        return (signal, quantity). The function may be sync or async.

        Args:
            symbol (str): 
                The stock symbol to update. Use "ALL" to update all positions.

        Raises:
            Exception: Catches and logs any exceptions that occur during update.
        """

        if not await self.is_market_open():
            self._config.log_info("Market is closed — skipping trading cycle")
            return

        try:
            positions = await self.get_positions()
            positions_to_update = (
                positions if symbol == "ALL" else [p for p in positions if p.get("symbol") == symbol]
            )

            if len(positions_to_update) == 0:
                print("Did not find any positions, try --order or -o to place an order... ")

            if symbol == "ALL": self._config.log_info("Updating all positions")
            else: self._config.log_info(f"Updating: {symbol}")

            tasks = []
            for position in positions_to_update:
                symbol_i = position.get("symbol")

                self._strategy.prepare_data(symbol_i, position)
                result = self._strategy.run()
                if inspect.isawaitable(result):
                    signal, qty = await result
                else:
                    signal, qty = result

                self._config.log_info(f"{symbol_i}: {signal.value}")

                if signal != SideSignal.HOLD:
                    order = OrderData(
                        symbol = symbol_i,
                        quantity = qty,
                        side = signal.value,
                        type = "market"
                    )
                    tasks.append(self.place_order(order))

            watchlist_tasks = await self._analyze_watchlist()
            tasks.extend(watchlist_tasks)
            await asyncio.gather(*tasks)

        except Exception:
            self._config.log_expectation(f"Failed to update position(s) for {symbol}")

    async def _analyze_watchlist(self) -> list:
        """
        Analyzes symbols in the watchlist and prepares order placement tasks.

        For each symbol in the watchlist, this method runs the AI-based trading
        strategy to determine whether a trade should be placed. If the strategy
        returns a signal other than HOLD, a market order is created and the
        corresponding order placement coroutine is added to the returned list.

        Any exceptions raised while analyzing individual symbols are caught and
        logged, allowing analysis of remaining symbols to continue.

        Returns:
            list:
                A list of order placement coroutines for symbols that generated
                actionable trading signals. The returned coroutines are intended
                to be awaited (e.g. via asyncio.gather) by the caller.
        """


        watchlist = self._config.watchlist

        if isinstance(watchlist, str):
            raise ValueError(
                "Watchlist must be a list of symbols, not a string. "
                'Example: ["AAPL", "GOOG", "SPY"]'
            )

        self._config.log_info(f"Checking watchlist: {watchlist}")

        tasks = []
        for symbol in watchlist:
            symbol = symbol.strip().upper()
            if not symbol or len(symbol) < 1:
                self._config.log_warning("Skipping invalid symbol in watchlist")
                continue

            try:
                self._strategy.prepare_data(symbol, {})
                signal, qty = await self._strategy.run()    # The ML strategy need awaiting

                self._config.log_info(f"{symbol}: {signal.value}")

                if signal != SideSignal.HOLD:
                    order = OrderData(
                        symbol = symbol,
                        quantity = qty,
                        side = signal.value,
                        type = "market"
                    )
                    tasks.append(self.place_order(order))
                    
            except Exception:
                self._config.log_expectation(f"Failed to analyze {symbol} from watchlist")
        
        return tasks



    async def live(self) -> None:
        """
        Runs the trader in continuous live-trading mode using a given strategy.

        This method enters an infinite asynchronous loop that periodically
        updates all current positions and analyzes the watchlist by invoking
        the provided strategy. On each iteration, it calls `update()` with
        the strategy applied to all positions ("ALL") and then sleeps for
        a fixed interval (60 seconds) before repeating.

        The loop continues indefinitely until interrupted by the user
        (KeyboardInterrupt) or cancelled by the event loop
        (asyncio.CancelledError), at which point the trader shuts down
        gracefully.

        Raises:
            KeyboardInterrupt:
                Raised when the user manually stops the live trading loop.
            asyncio.CancelledError:
                Raised when the task running this method is cancelled.
        """
        try:
            while True:
                await self.update("ALL")
                await asyncio.sleep(60) # sleep for a minute

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutting down... ")

        finally:
            pass


    def _find_strategy(self) -> Callable[[Dict[str, Any]], tuple[SideSignal, int]]:
        """
        Resolve and return a trading strategy function by name.

        If no strategy name is provided, the user is prompted to select one.
        The function repeatedly asks for input until a valid strategy name
        matching a key in the internal ``strategies`` dictionary is supplied.

        Returns:
            Callable[[Dict[str, Any]], tuple[SideSignal, int]]:
                The strategy function associated with the chosen name.

        Raises:
            KeyboardInterrupt:
                If the user aborts the selection process.
        """
        name = None

        while True:
            if name == None:
                name = self._config.strategy_name

            if name not in STRATEGIES:
                name = input("Which strategy do you want to use? ")

            try:
                return STRATEGIES[name]
            
            except KeyError:
                print(f"\nStrategy {name!r} was not found in the strategies dictionary. Try again...")

# --------- backtesting ----------------

    async def _compare_strategies(
        self,
        symbol: str,
        strategies: Dict[str, Callable],
        days: int = 80,
        initial_cash: float = 10000,
    ) -> pd.DataFrame:
        """
        Run multiple strategies against the same symbol and return a comparison DataFrame.

        Args:
            symbol: ticker symbol to evaluate
            strategies: mapping of strategy name -> callable function
            days: approximate days to fetch (unused with current fetch_data signature)
            initial_cash: starting cash for each run

        Returns:
            pandas.DataFrame with columns including strategy, symbol and performance metrics
        """
        results = []

        self._config.log_info("\n" + "=" * 60)
        self._config.log_info(f"Starting backtest for {symbol} with {len(strategies)} strategies")
        self._config.log_info(f"Initial Cash: ${initial_cash}")
        self._config.log_info("=" * 60 + "\n")

        for name, strat in strategies.items():
            self._config.log_info(f"Testing strategy: {name}")
            strat = strat(self._config)
            strat.prepare_data(symbol, {})
            try:
                backtester = Backtester(
                    self._config,
                    strat,
                    initial_cash=initial_cash,
                )

                total_bars = len(backtester.strategy.data.data)
                lookback = max(
                    total_bars - days,
                    self._config.load_min_lookback()
                )

                history = await backtester.run_strategy(lookback = lookback)
                metrics = backtester.calculate_metrics(history)

                # attach metadata
                if metrics:
                    metrics['strategy'] = name
                    metrics['symbol'] = symbol
                else:
                    metrics = {'strategy': name, 'symbol': symbol, 'error': 'no results or insufficient data'}

                results.append(metrics)

                if 'total_return_pct' in metrics:
                    self._config.log_info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
                    self._config.log_info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    self._config.log_info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
                else:
                    self._config.log_info(f"  Strategy {name} produced no metrics.\n")

            except Exception as e:
                self._config.log_expectation(f"Failed to run strategy {name}: {e}")
                results.append({'strategy': name, 'symbol': symbol, 'error': str(e)})

        df = pd.DataFrame(results)
        if 'total_return_pct' in df.columns:
            df = df.sort_values('total_return_pct', ascending=False)

        self._config.log_info("=" * 60)
        self._config.log_info("Backtest complete!")
        self._config.log_info("=" * 60 + "\n")

        return df


    async def _run_multi_symbol_backtest(
        self,
        symbols: List[str],
        strategies: Dict[str, Callable],
        days: int = 30,
        initial_cash: float = 10000,
    ) -> pd.DataFrame:
        """
        Convenience wrapper to run compare_strategies across multiple symbols.
        """
        all_results = []
        for s in symbols:
            try:
                res = await self._compare_strategies(s, strategies, initial_cash=initial_cash, days = days)
                all_results.append(res)
            except Exception as e:
                self._config.log_expectation(f"Failed backtest for {s}: {e}")

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    async def run_backtest(self):

        symbols = self._config.watchlist
        days, initial_cash, strategies_list = self._config.load_backtesting_variables()

        test_strategies = dict(STRATEGIES)
        test_strategies.pop("rule_based_strategy", None)
        test_strategies = {
            name.lower(): STRATEGIES[name.lower()]
            for name in strategies_list
            if name.lower() in STRATEGIES
        }

        print("Starting Backtest")
        print("───────────────────────────────────────────────")
        print(f"Symbol: {symbols}")
        print(f"Strategies: {len(test_strategies)}")

        try:
            results = await self._run_multi_symbol_backtest(
            symbols = symbols,
            strategies = test_strategies,
            initial_cash = initial_cash,
            days = days
        )
        except Exception as e:
            print(f"Error during backtest: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\n Results type: {type(results)}")
        print(f"Results columns: {results.columns.tolist()}")
        print(f"Results shape: {results.shape}")
        
        if results is not None and not results.empty:
            print("\n Backtest Results:")
            print(results.to_string(index=False))
            
            if 'total_return_pct' in results.columns:
                results = results.sort_values("total_return_pct", ascending=False)
                results.to_csv("backtest_results.csv", index=False)
                
                best = results.iloc[0]
                print("\n Best strategy:")
                print(f"   → {best['strategy']} with {best['total_return_pct']:.2f}% return")
                print("\n Results saved to backtest_results.csv")
            else:
                print("\n No 'total_return_pct' column found!")
                print(f"Available columns: {results.columns.tolist()}")
        else:
            print("\n No results returned from compare_strategies()")