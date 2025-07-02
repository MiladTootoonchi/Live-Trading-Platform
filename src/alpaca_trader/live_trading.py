import requests
import asyncio
from typing import Callable, List, Dict, Any

from .order import OrderData, SideSignal

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

        print("Sending Order Data:", data)
        print("Headers:", self._HEADERS, "\n")

        response_json = response.json()
        order_id = response_json["id"]
        await self._wait_until_order_filled(order_id)
    
    async def cancel_all_orders(self) -> None:
        """
        Cancels all open orders for the account.

        Sends a request to the Alpaca API to delete all active orders.
        Upon completion, prints the response content for confirmation or debugging purposes.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.delete, url, headers = self._HEADERS)
        print("Cancel response:", response.content)

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
                print(f"Order {order_id} filled for {order['symbol']}.")
                return
            print(f"Waiting for order {order_id} to fill...")
            await asyncio.sleep(60)



    async def update(self, strategy: Callable[[Dict[str, Any]], tuple[SideSignal, int]], symbol: str) -> None:
        """
        Updates one or all positions based on the provided trading strategy.

        The 'strategy' function should accept a position dictionary and return a tuple:
        (signal, quantity), where 'signal' is one of the SideSignal enum values 
        (BUY, SELL, HOLD), and 'quantity' is the number of shares to trade.

        Args:
            strategy (Callable[[Dict[str, Any]], Tuple[SideSignal, int]]): 
                A function that takes a position dict and returns a (signal, quantity) tuple.
            symbol (str): The stock symbol to update. Use "ALL" to update all positions.

        Raises:
            Exception: Catches and logs any exceptions that occur during update.
        """

        try:
            positions = await self.get_positions()
            positions_to_update = (
                positions if symbol == "ALL" else [p for p in positions if p.get("symbol") == symbol]
            )

            if symbol == "ALL": print("Updating all positions")
            else: print(f"Updating: {symbol}")

            for position in positions_to_update:
                symbol_i = position.get("symbol")
                signal, qty = strategy(position)
                print(f"{symbol_i}: {signal.value}\n")

                if signal != SideSignal.HOLD:
                    order = OrderData(
                        symbol = symbol_i,
                        quantity = qty,
                        side = signal.value,
                        type = "market"
                    )
                    await self.place_order(order)

                else: await asyncio.sleep(60)  # sleep for a minute

        except Exception as e:
            print(f"Failed to update position(s) for {symbol}: {e}\n")