import requests
from order import OrderData
import time

# Config
from dotenv import load_dotenv
import os

load_dotenv()

ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_KEY = os.getenv("ALPACA_KEY")

# Package
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



    def createMarketOrder(self, order_data: OrderData):
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
            orders = self.getOrders()
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



    def getOrders(self):
        """
        This method will return a list of order objects.
        Each one of them includes id, symbol, quantity, side, status and a timestamp of order creation
        """

        url = f"{self._APCA_API_BASE_URL}/v2/orders"
        response = requests.get(url, headers = self._HEADERS)
        return response.json()

    def getPositions(self):
        """
        This method shows you the symbol, quantity, market value and the pl so far
        """

        url = f"{self._APCA_API_BASE_URL}/v2/positions"
        response = requests.get(url, headers = self._HEADERS)
        return response.json()
    

"""data = {
    "symbol" : "AAPL",
    "qty" : 2,
    "side" : "buy",
    "type" : "market",
    "time_in_force" : "day"
}"""

if __name__ == "__main__":
    trader = AlpacaTrader(ALPACA_KEY, ALPACA_SECRET_KEY)