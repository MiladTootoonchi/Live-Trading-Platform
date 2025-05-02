import requests
import time
from ..strategies.order import OrderData
from ..strategies.strategy import StrategyForOrder

class AlpacaTrader:
    """
    A class to handle the Alpaca trading given API keys and the order data.

    This class will create market orders that will either buy or sell positions,
    Get the order object from Alpaca and
    Get all the posistions.
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

        self._strategies = []


    def send_order_to_market(self, order_data: OrderData):
        """
        This method will create an order that will either buy or sell positions given
        by the order information from the order data.

        Args:
            order_data: An OrderData object with the necessary information for making a trade.
        """
        data = order_data.get_dict()

        ORDERS_URL = "{}/v2/orders".format(self._APCA_API_BASE_URL)
        response = requests.post(ORDERS_URL, json = data, headers = self._HEADERS)

        print("Sending Order Data:", data)
        print("Headers:", self._HEADERS, "\n")

        return response.content
    
    def wait_until_order_filled(self, order_data: OrderData) -> None:
        """
        A method that will pause the run till the order is filled.

        Args:
            order_data: An OrderData object with the necessary information for making a trade.
        """

        data = order_data.get_dict()

        while True:
            orders = self.get_orders()
            for order in orders:
                if order["symbol"] == data and order["status"] == "filled":
                    print("Buy order filled.")
                    return
                
            print("Waiting for order to fill...")
            time.sleep()

    def cancel_all_orders(self) -> None:
        """
        A method that will cancel all orders.
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = requests.delete(url, headers = self._HEADERS)
        print("Cancel response:", response.content)



    def get_orders(self) -> list[dict]:
        """
        This method will return a list of order objects.
        Each one of them includes id, symbol, quantity, side, status and a timestamp of order creation
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = requests.get(url, headers = self._HEADERS)
        return response.json()

    def get_positions(self) -> list[dict]:
        """
        This method shows you the symbol, quantity, market value and the pl so far
        """

        url = f"{self._APCA_API_BASE_URL}/v2/positions"
        response = requests.get(url, headers = self._HEADERS)
        return response.json()
    


    def order(self, strategy: StrategyForOrder) -> None:
        """
        This method will buy the order, then append it to a list of strategies with positions

        Args:
            strategy: a strategy object with an order inforamtion.
        """

        order = strategy.order
        self.send_order_to_market(order)

        self._strategies.append(strategy)

    def update(self):
        """
        A method that will get a marketorder from the strategies.
        """
        for strategy in self._strategies:
            order = strategy.generate_new_order()
            
            if order is not None:
                self.send_order_to_market(order)
