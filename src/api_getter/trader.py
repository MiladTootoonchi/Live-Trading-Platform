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

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the trader's strategy with new market data.
        This method does not execute trades, just updates the strategy.

        Args:
            data (Dict): A dictionary containing market data.
        """ 

        current_price = data.get('price', 0.0)

        for _, strategy in self._positions.items():
            strategy.update(data)
            
            is_long = strategy.is_long
            is_active = strategy.active
            size = strategy.position_size

            # Save position opening
            if is_long and not is_active:
                trade = {
                    'strategy': strategy.name,
                    'action': 'open_long',
                    'size': size,
                    'price': current_price,
                    'stop_loss': strategy.stop_loss,
                    'take_profit': strategy.take_profit
                }
                self._trade_history.append(trade)

                print(f"Opened long strategy, {strategy.name}, Size = {size}, Price = {current_price}")

            # Save position closing
            elif not is_long and is_active:
                self._close_position(strategy, current_price)

            # Update position info if still active
            if is_active:
                strategy.current_position_price = current_price
                if is_long:
                    strategy.pnl = (current_price - strategy.entry_price) * size

                self._check_exit_conditions(strategy, current_price)

    def _close_position(self, strategy, price = None):
        """
        Close a position for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            price: Optional closing price (uses current price if not provided)
        """
        
        if not strategy.active:
            return
            
        if price is None:
            price = strategy.current_position_price
            
        # Calculate P&L
        if strategy.is_long:
            pnl = (price - strategy.entry_price * strategy.position_size)
        else:  # short
            pnl = (strategy.entry_price - price) * strategy.position_size
            
        # Apply commission
        commission = price * strategy.position_size * self.commission_rate
        net_pnl = pnl - commission
        
        # Update capital
        self.capital += net_pnl
        
        # Record the trade
        trade = {
            'strategy': strategy.name,
            'action': 'close',
            'size': strategy.position_size,
            'price': price,
            'pnl': pnl,
            'commission': commission,
            'net_pnl': net_pnl
        }
        self._trade_history.append(trade)
        
        print(f"Closed strategy, {strategy.name}, Price = {price}, PnL = {net_pnl:.2f}")
        
        # Reset position
        strategy.active = False
        strategy.is_long = None
        strategy.position_size = 0.0
        strategy.entry_price = 0.0
        strategy.current_position_price = 0.0
        strategy.pnl = 0.0
        strategy.stop_loss = None
        strategy.take_profit = None

    def _check_exit_conditions(self, strategy, current_price):
        """
        Check if stop loss or take profit conditions are met.
        
        Args:
            strategy_name: Name of the strategy
            current_price: Current price of the market
        """
        
        if not strategy.active:
            return
            
        # Check stop loss
        if strategy.stop_loss is not None:
            if strategy.is_long and current_price <= strategy.stop_loss:
                print(f"STOP LOSS TRIGGERED: Strategy = {strategy.name}, Price={current_price}")
                self._close_position(strategy, current_price)
                return
                
        # Check take profit
        if strategy.take_profit is not None:
            if strategy.is_long and current_price >= strategy.take_profit:
                print(f"TAKE PROFIT TRIGGERED: Strategy = {strategy.name}, Price = {current_price}")
                self._close_position(strategy, current_price)
                return
    
    def get_performance_summary(self) -> dict:
        """
        Generate a performance summary for all strategies.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_trades = len([t for t in self.trade_history if t['action'] == 'close'])
        winning_trades = len([t for t in self.trade_history if t['action'] == 'close' and t.get('net_pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t['action'] == 'close' and t.get('net_pnl', 0) <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum([t.get('net_pnl', 0) for t in self.trade_history if t['action'] == 'close' and t.get('net_pnl', 0) > 0])
        total_loss = sum([t.get('net_pnl', 0) for t in self.trade_history if t['action'] == 'close' and t.get('net_pnl', 0) <= 0])
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_commission': sum([t.get('commission', 0) for t in self.trade_history if t['action'] == 'close'])
        }


    """
    def execute_trade(self, data: Dict[str, Any]) -> None:

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
    """
