from abc import ABC, abstractmethod
from .order import OrderData, createOrder


class StrategyForOrder(ABC):
    """
    An abstract class that will work as a fundemental startegy class with one order.

    The task of this class is to calculate and give signals on what the trader should do. With a 
    F.eks. buy, sell or hold the position.
    """

    def __init__(self) -> None:
        """
        Initialize Strategy class

        Will ask for order information and send it
        """

        self._order = createOrder()


    @property
    def order(self) -> OrderData:
        return self._order
    

    @abstractmethod
    def generate_new_order(self):
        """
        This method will evaluate if you should buy, sell or hold.
        Then it creates and OrderData object based on its evaluation.
        
        Returns:
            A OrderData object that the trader will send to the market.
        """
        pass

