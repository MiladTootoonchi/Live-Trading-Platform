import requests
from order import OrderData

# Config
ALPACA_SECRET_KEY = "jLwj0J33MMhcGVP9oFTZngMTXDbowMpLj2VN8dmL"
ALPACA_KEY = "PKIAS4EHSEJ45519IHHC"

# Package
class AlpacaTrader:
    """
    A class to handle the Alpaca trading given API keys and the order data.

    This class will create market orders that will either buy or sell positions,
    Get the order object from Alpaca and
    Get all the posistions.
    """

    def __init__(self, key: str, secret_key: str, order_data: OrderData) -> None:
        """
        Initialize the trader.
        
        Args:
            key: The key given by Alpaca
            secret_ket: The secret ket alpaca key
            order_data: An OrderData object with the necessary information
        """

        self._HEADERS = {
            "APCA-API-KEY-ID" : key,
            "APCA-API-SECRET-KEY": secret_key,
        }

        self._data = order_data.get_dict()

        self._APCA_API_BASE_URL =  "https://paper-api.alpaca.markets"



    def createMarketOrder(self):
        """
        This method will create an order that will either buy or sell positions given
        by the order information from the order data.
        """

        ORDERS_URL = "{}/v2/orders".format(self._APCA_API_BASE_URL)
        response = requests.post(ORDERS_URL, json = self._data, headers = self._HEADERS)
        return response.content

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