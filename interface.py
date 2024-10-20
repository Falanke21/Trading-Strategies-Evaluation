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
    def swing_trade(self, symbol='AAPL') -> MarketDecision:
        """
        Swing trade a stock based on a strategy
        :param symbol: Stock ticker symbol
        :return: Decision to buy or sell or hold the stock
        """
        raise NotImplementedError
