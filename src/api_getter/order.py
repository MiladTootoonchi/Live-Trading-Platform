from dataclasses import dataclass

@dataclass
class OrderData:
    """Class representing a trading order."""
    symbol: str
    quantity: int
    side: str
    type: str
    time_in_force = "day"

    def get_dict(self) -> dict:
        data = {
            "symbol" : self.symbol,
            "qty" : self.quantity,
            "side" : self.side,
            "type" : self.type,
            "time_in_force" : self.time_in_force
        }

        return data
    
"""data = {
    "symbol" : "AAPL",
    "qty" : 2,
    "side" : "buy",
    "type" : "market",
    "time_in_force" : "day"
}"""