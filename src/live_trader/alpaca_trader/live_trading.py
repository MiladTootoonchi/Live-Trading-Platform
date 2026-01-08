import requests
import asyncio
from typing import Callable, List, Dict, Any
import inspect

from .order import OrderData, SideSignal
from config import make_logger, load_watchlist


logger = make_logger()

class AlpacaTrader:
    """
    A class to handle the Alpaca trading given API keys and the order data.

    This class will create market orders that will either buy or sell positions,
    Get the order object from Alpaca and
    Get all the posistions.
    Get orders from a strategy and send it.
    """

    def __init__(self, key: str, secret_key: str) -> None:
        """
        Initialize the trader.
        
        Args:
            key: The key given by Alpaca
            secret_key: The secret key alpaca key
        """

        self._HEADERS = {
            "APCA-API-KEY-ID" : key,
            "APCA-API-SECRET-KEY": secret_key,
        }

        self._APCA_API_BASE_URL =  "https://paper-api.alpaca.markets"



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

        logger.info(f"Sending Order Data:\n {data} \n Headers: {self._HEADERS}\n")

        response_json = response.json()

        if "id" not in response_json:
            logger.error(f"Order rejected: {response_json} \n")
            return

        order_id = response_json["id"]
        await self._wait_until_order_filled(order_id)

    async def close_position(self, symbol: str) -> None:
        """
        Closes (liquidates) the entire open position for a given stock symbol.

        This method sends a request to Alpaca’s position liquidation endpoint to
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
            logger.info(f"Successfully closed entire position for {symbol}.")
        else:
            logger.error(f"Failed to close position for {symbol}: {response.text}")
    
    async def cancel_all_orders(self) -> None:
        """
        Cancels all open orders for the account.

        Sends a request to the Alpaca API to delete all active orders.
        Upon completion, prints the response content for confirmation or debugging purposes.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.delete, url, headers = self._HEADERS)
        logger.info(f"Cancel response: {response.content}\n")

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
            logger.error(f"Failed to fetch orders: {response.text}\n")
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
                    logger.info(
                        f"Canceled last open order {order_id} for {symbol}.\n"
                    )
                else:
                    logger.error(
                        f"Failed to cancel order {order_id}: {cancel_response.text}\n"
                    )
                return

        logger.info("No open orders found to cancel.\n")
    


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

        If 'all' is provided, the method closes the full position using Alpaca’s
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
                logger.info(f"Order {order_id} filled for {order['symbol']}. \n")
                return
            print(f"Waiting for order {order_id} to fill...")   # this is just temporary info, does not need to log this
            await asyncio.sleep(60) # Sleep for a minute



    async def update(self, strategy: Callable, analyzer_strategy: Callable, symbol: str) -> None:
        """
        Updates one or all positions based on the provided trading strategy.

        The strategy function should accept (symbol, position) and
        return (signal, quantity). The function may be sync or async.

        Args:
            strategy (Callable): 
                A function that takes the position dict and returns a (signal, quantity) tuple.
            analyzer_strategy(Callable):
                Same as strategy, however used for watchlist analyzer.
            symbol (str): 
                The stock symbol to update. Use "ALL" to update all positions.

        Raises:
            Exception: Catches and logs any exceptions that occur during update.
        """

        try:
            positions = await self.get_positions()
            positions_to_update = (
                positions if symbol == "ALL" else [p for p in positions if p.get("symbol") == symbol]
            )

            if len(positions_to_update) == 0:
                print("Did not find any positions, try --order or -o to place an order... ")

            if symbol == "ALL": logger.info("Updating all positions")
            else: logger.info(f"Updating: {symbol}")

            tasks = []
            for position in positions_to_update:
                symbol_i = position.get("symbol")

                result = strategy(symbol_i, position)
                if inspect.isawaitable(result):
                    signal, qty = await result
                else:
                    signal, qty = result

                logger.info(f"{symbol_i}: {signal.value}")

                if signal != SideSignal.HOLD:
                    order = OrderData(
                        symbol = symbol_i,
                        quantity = qty,
                        side = signal.value,
                        type = "market"
                    )
                    tasks.append(self.place_order(order))

            watchlist_tasks = self._analyze_watchlist(analyzer_strategy)
            tasks.extend(watchlist_tasks)
            await asyncio.gather(*tasks)

        except Exception:
            logger.exception(f"Failed to update position(s) for {symbol}")

    async def _analyze_watchlist(self, analyzer_strategy: Callable) -> list:
        """
        Analyzes symbols in the watchlist and prepares order placement tasks.

        For each symbol in the watchlist, this method runs the AI-based trading
        strategy to determine whether a trade should be placed. If the strategy
        returns a signal other than HOLD, a market order is created and the
        corresponding order placement coroutine is added to the returned list.

        Any exceptions raised while analyzing individual symbols are caught and
        logged, allowing analysis of remaining symbols to continue.

        analyzer_strategy(Callable):
                A function that takes the position dict and returns a (signal, quantity) tuple.
                however used for analyzing watchlist.

        Returns:
            list:
                A list of order placement coroutines for symbols that generated
                actionable trading signals. The returned coroutines are intended
                to be awaited (e.g. via asyncio.gather) by the caller.
        """


        watchlist = load_watchlist()
        logger.info(f"Checking watchlist: {watchlist}")

        tasks = []
        for symbol in watchlist:
            try:
                signal, qty = await analyzer_strategy(symbol)    # The ML strategy need awaiting

                logger.info(f"{symbol}: {signal.value}")

                if signal != SideSignal.HOLD:
                    order = OrderData(
                        symbol = symbol,
                        quantity = qty,
                        side = signal.value,
                        type = "market"
                    )
                    tasks.append(self.place_order(order))
                    
            except Exception:
                logger.exception(f"Failed to analyze {symbol} from watchlist")
        
        return tasks
