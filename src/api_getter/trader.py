from strategy import Strategy
from typing import Dict, Any


class Trader:
    """
    Simple stock trading class focused on buying and selling
    """
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the trader with starting capital.
        Args:
        initial_capital: Starting trading capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital  
        self.strategy = None
        
        # Current position tracking
        self.current_position = {
            "symbol": None,
            "quantity": 0,
            "entry_price": 0.0,
        }
        
        # Trade history
        self.trade_history = []
        
    def add_strategy(self, strategy: Strategy):
        """
        Add a trading strategy to the trader.

        Args:
        strategy (Strategy): A strategy that defines how trades should be executed.
        """
        self.strategy = strategy

    def remove_strategy(self):
        """
        Remove the current strategy from the trader.
        """
        if self.strategy is None:
            print("No strategy to remove")
        else: 
            print(f"Strategy {self.strategy} removed.")
            self.strategy = None

    def open_long(self, symbol: str, quantity: int, price: float):
        """
        Open a long position for the trader.

        Args:
        symbol (str): The symbol of the stock
        quantity (int): The number of shares to buy
        price (float): The price at which to buy the shares.
        """
        if price * quantity > self.current_capital:
            print("Insufficient funds to open long position.")
            return
        
        self.current_position = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": price,
        }

        self.current_capital -= price * quantity 
        print(f"Opened long position: {quantity} shares of {symbol} at ${price} each.")

    def update(self, data: Dict[str, Any]):
        """
        Update the trader's strategy with new market data.
        This method does not execute trades, just updates the strategy.

        Args:
        data (Dict): A dictionary containing market data, such as price.
        """    
        if not self.strategy:
            print("No strategy is set.")
            return
        
        self.strategy.update(data)
        print(f"Strategy updated with new data: {data}")

    def execute_trade(self, data: Dict[str, Any]) -> None:
        """
        Uses the strategy to analyze the market and execute a trade
        based on the analysis.

        Args:
        data (Dict): A dictionary containing market data
        """
        if self.strategy:
            order = self.strategy.execute_strategy(data)
            if order:
                if order.direction:  # If the order is a buy (long)
                    total_cost = order.quantity * order.price
                    if total_cost > self.current_capital:
                        raise ValueError("Insufficient funds for this purchase")

                    self.current_position = {
                        "symbol": data["symbol"],
                        "quantity": order.quantity,
                        "entry_price": order.price
                    }
                    self.current_capital -= total_cost

                    # Record trade
                    self.trade_history.append({
                        "action": "buy",
                        "symbol": data["symbol"],
                        "quantity": order.quantity,
                        "price": order.price
                    })
                    print(f"Bought {order.quantity} shares of {data['symbol']} at ${order.price}")
                else:  # If the order is a sell (short)
                    profit_loss = (order.price - self.current_position["entry_price"]) * order.quantity
                    
                    self.current_capital += order.quantity * order.price
                    self.trade_history.append({
                        "action": "sell",
                        "symbol": data["symbol"],
                        "quantity": order.quantity,
                        "price": order.price,
                        "profit_loss": profit_loss
                    })
                    
                    # Reset position
                    self.current_position = {
                        "symbol": None,
                        "quantity": 0,
                        "entry_price": 0.0,
                    }

                    print(f"Sold {order.quantity} shares of {data['symbol']} at ${order.price}")

    def get_portfolio_summary(self):
        """
        Return a summary of the trader's portfolio (capital and positions).
        """
        portfolio_summary = {
        "current_capital": self.current_capital,
        "current_position": self.current_position,
        "trade_history": self.trade_history,
        }
        return portfolio_summary
