import abc
from enum import Enum


class MarketAction(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3


class MarketDecision():
    """
    Represents a decision to buy or sell a stock
    """
    def __init__(self, symbol: str, action: MarketAction, price: float, quantity: int):
        self.symbol = symbol
        self.action = action
        self.price = price
        self.quantity = quantity


class IStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_signal(self, symbol='AAPL', date=None, position=0, cash=0.0) -> MarketDecision:
        """
        Generate a trading signal for a stock based on a strategy
        :param symbol: Stock ticker symbol
        :param date: Date for the signal generation. The default is the current date.
        The strategy is able to look back any historical data before the date to 
        generate the signal.
        :param position: Current number of shares held
        :param cash: Available cash for trading
        :return: Decision to buy or sell or hold the stock
        """
        raise NotImplementedError

