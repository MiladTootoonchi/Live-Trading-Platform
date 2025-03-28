from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


@dataclass
class Order:
    """ Class defining a trade order. """
    price: float
    quantity: float
    direction: bool # (Long or Neutral)


class Strategy(ABC):
    """
    Foundation class for algorithmic trading strategies.
    This class offers a framework to build trading strategies, 
    including common methods for managing positions and generating signals.
    """

    def __init__(self, name: str):
        """
        Creating strategy.

        Args:
            name: A name that the strategy will identefy as
        """

        self._name = name
        self._position = False # (Long or not)
        self._position_size = 0.0
        self._entry_price = 0.0
        self._stop_loss = 0.0
        self._take_profit = 0.0

        # For order management
        self._order_pending = False
        self._pending_order = None


    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """
        Processing new market data and updating the strategy state.

        Args:
            data: Dictionary containing market data
        """
        pass

    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Analyze data and determine whether to go long or stay neutral.

        Args:
            data: Dictionary containing market data

        Returns:
            A tuple containing the posistion (long) signal (True or False) and the signal strength (0.0 to 1.0)
        """
        pass

    def execute_strategy(self, data: Dict[str, Any]) -> Optional[Order]:
        """
        This method calls generate_signal() to determine to go long or neutral (True or False),
        then executes an action based on the signal.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            An Order object if a trade is executed, None otherwise
        """

        signal, strength = self.generate_signal(data)
        # Should probably throw an error? 
        current_price = data.get('price', 0.0)
        
        # Determine position size based on signal strength
        size = self.calculate_position_size(strength)
        
        if signal == True and self._position != True:
            if self._position == False:
                self.close()
            self.long(size, current_price)
            order = Order(price = current_price, quantity = size, direction = True)
            self._order_pending = True
            self._pending_order = order
            return order
            
        elif signal == False and self._position != False:
            self.close()
            order = Order(price = current_price, quantity = self._position_size, direction = False)
            self._order_pending = True
            self._pending_order = order
            return order
            
        return None
    
    @abstractmethod
    def calculate_position_size(self, signal_strength: float) -> float:
        """
        Calculate the position size based on signal strength.
        
        Args:
            signal_strength: A value between 0.0 and 1.0 indicating the strength of the signal
            
        Returns:
            The position size to use
        """
        pass
    
    def long(self, size: float, entry_price: float, stop_loss: Optional[float] = None, 
             take_profit: Optional[float] = None) -> None:
        """
        Enter a long position.
        
        Args:
            size: Position size
            entry_price: Entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        """
        self._position = True
        self._position_size = size
        self._entry_price = entry_price
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        print(f"LONG: Size = {size}, Entry = {entry_price}, SL = {stop_loss}, TP = {take_profit}")
    
    @abstractmethod       
    def close(self) -> None:
        """Close the current position."""
        pass
    
    def adjust_stop_loss(self, new_stop_loss: float) -> None:
        """
        Adjust the stop loss level for the current position.
        
        Args:
            new_stop_loss: New stop loss price
        """
        self._stop_loss = new_stop_loss
        print(f"ADJUSTED SL: New stop loss = {new_stop_loss}")
    
    def adjust_take_profit(self, new_take_profit: float) -> None:
        """
        Adjust the take profit level for the current position.
        
        Args:
            new_take_profit: New take profit price
        """
        self._take_profit = new_take_profit
        print(f"ADJUSTED TP: New take profit = {new_take_profit}")
    
    def is_in_position(self) -> bool:
        """Check if the strategy is currently in a position."""
        return self._position != False
    
    def get_position(self) -> bool:
        """Get the current position state."""
        return self._position