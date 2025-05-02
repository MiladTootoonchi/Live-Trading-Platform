from dataclasses import dataclass

@dataclass
class OrderData:
    """
    A data class representing a trading order.

    Args:
        symbol: The stock ticker symbol (e.g., "AAPL" for Apple).
        quantity: The number of shares to trade.
        side: The side of the order, either "buy" or "sell".
        type: The type of order, typically "market" or "limit".
        time_in_force: Duration the order remains active. Default is "day".
    """

    symbol: str
    quantity: int
    side: str
    type: str
    time_in_force: str = "day"

    def get_dict(self) -> dict:
        """
        Makes a market dictionary that the API can read

        Returns:
            A dictionary of the order information
        """

        data = {
            "symbol" : self.symbol,
            "qty" : self.quantity,
            "side" : self.side,
            "type" : self.type,
            "time_in_force" : self.time_in_force
        }

        return data


def createOrder() -> OrderData:
    """
    A function that creates orders from question inputs.
    """

    symbol = input("Which market do you want to buy from (symbol)? ")
    qty = input("\nHow much do you want to buy (quantity)? ")
    market_type = input("\nWhat type of order do you want (f.eks. market or limit)? ")

    return OrderData(symbol = symbol, quantity = qty, side = "buy", market_type = market_type)