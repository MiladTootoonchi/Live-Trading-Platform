import requests
import asyncio
from typing import Callable, Dict, Any

from .order import OrderData
from ..strategies.strategy import SideSignal

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
            secret_key: The secret ket alpaca key
        """

        self._HEADERS = {
            "APCA-API-KEY-ID" : key,
            "APCA-API-SECRET-KEY": secret_key,
        }

        self._APCA_API_BASE_URL =  "https://paper-api.alpaca.markets"


    async def place_order(self, order_data: OrderData):
        """
        This method will create an order that will either buy or sell positions given
        by the order information from the order data.

        Args:
            order_data: An OrderData object with the necessary information for making a trade.
        """
        data = order_data.get_dict()

        ORDERS_URL = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.post, ORDERS_URL, json = data, headers = self._HEADERS)

        print("Sending Order Data:", data)
        print("Headers:", self._HEADERS, "\n")

        return response.content
    
    async def wait_until_orders_filled(self) -> None:
        """
        A method that will pause the run till all the order are filled.
        """

        while True:
            orders = await self.get_orders()
            if orders == []: return # Stops if there is an empty list of orders
            for order in orders:
                if order["status"] == "filled":
                    print(f"{order['symbol']} filled.")
                    return
                
            print(f"Waiting for all orders to fill...")
            await asyncio.sleep(60) # Waits 1 minute to check again

    async def cancel_all_orders(self) -> None:
        """
        A method that will cancel all orders.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.delete, url, headers = self._HEADERS)
        print("Cancel response:", response.content)



    async def get_orders(self) -> list[dict]:
        """
        This method will return a list of order objects.
        Each one of them includes id, symbol, quantity, side, status and a timestamp of order creation
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = await asyncio.to_thread(requests.get, url, headers = self._HEADERS)
        return response.json()

    async def get_positions(self) -> list[dict]:
        """
        This method shows you the symbol, quantity, market value and the pl so far
        """

        url = f"{self._APCA_API_BASE_URL}/v2/positions"
        response = await asyncio.to_thread(requests.get, url, headers = self._HEADERS)
        return response.json()
    


    async def create_buy_order(self) -> None:
        """
        A method that creates a buy orders from question inputs.
        """

        # Asking for stock
        symbol = input("Which stock do you want to buy (symbol)? ").strip().upper()


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



    async def update(self, strategy: Callable[[Dict[str, Any]], tuple[SideSignal, int]], symbol: str) -> None:
        """
        Updates one or all positions using the provided strategy function.
        The strategy should return a tuple of (signal, quantity), where signal is "BUY", "SELL" or "HOLD".
        
        Args:
            strategy: A function that takes a position and returns (signal, quantity).
            symbol: The stock symbol to update. If empty, updates all positions.
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

                if signal != SideSignal.HOLD:
                    print(signal)
                    if signal == SideSignal.BUY: signal = "buy"
                    if signal == SideSignal.SELL: signal = "sell"

                    order = OrderData(
                        symbol = symbol_i,
                        quantity = qty,
                        side = signal,
                        type = "market"
                    )
                    await self.place_order(order)

                else: print(f"Holding {symbol_i}")

        except Exception:
            print(f"Failed to update position(s) for symbol {symbol}")