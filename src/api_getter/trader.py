from strategy import Strategy
from typing import Dict, Any


class Trader:
    """
    Simple stock trading class focused on execution of trades based on strategy signals
    """
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the trader with starting capital.

        Args:
            initial_capital: Starting trading capital
        """
        
        self._initial_capital = initial_capital
        self._current_capital = initial_capital  
        self._positions = {}
        self._trade_history = []
        self._commission_rate = 0.001
        
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a trading strategy to the traders strategies.

        Args:
            strategy (Strategy): A strategy that defines how trades should be executed.
        """
        self._positions[strategy._name] = strategy
        print(f"Added strategy: {strategy._name}")

    def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove the current strategy from the trader.

        Args:
            strategy_name: Name of the unwanted strategy
        """

        strategy = self._positions[strategy_name]
        try:
            if strategy.active:
                self._close_position(strategy_name)
                
            del self._positions[strategy_name]

            print(f"Removed {strategy_name} from stategies")

        except:
            print(f"Strategy not found: {strategy_name}")

    def update(self, data: Dict[str, Any]):
        """
        Update the trader's strategy with new market data.
        This method does not execute trades, just updates the strategy.

        Args:
            data (Dict): A dictionary containing market data.
        """ 

        current_price = data.get('price', 0.0)

        for name, strategy in self._positions.items():
            strategy.update(data)
            
            is_long = strategy.is_long
            is_active = strategy.is_active
            size = strategy.position_size
            entry_price = strategy.entry_price

            # Handle position opening
            if is_long and not is_active:
                self._open_long(name, size, current_price, strategy.stop_loss, strategy.take_profit)
            
            # Handle position closing
            elif not is_long and is_active:
                self._close_position(name, current_price)

            # Update position info if still active
            if is_active and is_long:
                strategy.pnl= (current_price - entry_price) * size
            

    def _open_long(self, symbol: str, quantity: int, price: float) -> None:
        """
        Open a long position for the trader.

        Args:
            symbol (str): The symbol of the stock
            quantity (int): The number of shares to buy
            price (float): The price at which to buy the shares.
        """
        if price * quantity > self._current_capital:
            print("Insufficient funds to open long position.")
            return
        
        self._current_positions = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": price,
        }

        self._current_capital -= price * quantity

        print(f"Opened long position: {quantity} shares of {symbol} at ${price} each.")


    def execute_trade(self, data: Dict[str, Any]) -> None:
        """
        Uses the strategy to analyze the market and execute a trade
        based on the analysis.

        Args:
        data (Dict): A dictionary containing market data
        """
        if self._positions:
            order = self._positions.execute_strategy(data)
            if order:
                if order.direction:  # If the order is a buy (long)
                    total_cost = order.quantity * order.price
                    if total_cost > self._current_capital:
                        raise ValueError("Insufficient funds for this purchase")

                    self._current_positions = {
                        "symbol": data["symbol"],
                        "quantity": order.quantity,
                        "entry_price": order.price
                    }
                    self._current_capital -= total_cost

                    # Record trade
                    self._trade_history.append({
                        "action": "buy",
                        "symbol": data["symbol"],
                        "quantity": order.quantity,
                        "price": order.price
                    })
                    print(f"Bought {order.quantity} shares of {data['symbol']} at ${order.price}")
                else:  # If the order is a sell (short)
                    profit_loss = (order.price - self._current_positions["entry_price"]) * order.quantity
                    
                    self._current_capital += order.quantity * order.price
                    self._trade_history.append({
                        "action": "sell",
                        "symbol": data["symbol"],
                        "quantity": order.quantity,
                        "price": order.price,
                        "profit_loss": profit_loss
                    })
                    
                    # Reset position
                    self._current_positions = {
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
        "current_capital": self._current_capital,
        "current_position": self._current_positions,
        "trade_history": self._trade_history,
        }
        return portfolio_summary
