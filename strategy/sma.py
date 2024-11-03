import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import pandas_ta as ta
from interface import MarketAction, MarketDecision, IStrategy

class SMAStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str,
                 lookback_days: int = 180,
                 trade_quantity: int = 300,
                 short_window: int = 5,
                 long_window: int = 20):
        """
        Simple Moving Average (SMA) Crossover Strategy
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            lookback_days (int): Number of days to look back for historical data
            trade_quantity (int): Fixed quantity to trade
            short_window (int): Short-term SMA period
            long_window (int): Long-term SMA period
        """
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret)
        self._lookback_days = lookback_days
        self._trade_quantity = trade_quantity
        self._short_window = short_window
        self._long_window = long_window

    def _calculate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate short and long-term SMAs"""
        data['SMA_short'] = ta.sma(data['close'], length=self._short_window)
        data['SMA_long'] = ta.sma(data['close'], length=self._long_window)
        return data

    def generate_signal(self, symbol='AAPL', date=None, position=0, cash=0.0) -> MarketDecision:
        """
        Generate trading signals based on SMA crossover strategy
        
        Buy when short-term SMA crosses above long-term SMA
        Sell when short-term SMA crosses below long-term SMA
        """
        if date is None:
            date = datetime.now()
        start_date = date - timedelta(days=self._lookback_days)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=date
        )
        
        try:
            bars = self._client.get_stock_bars(request_params).df
            if len(bars) < self._long_window:
                return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                    price=bars['close'].iloc[-1], quantity=0)
            
            bars = self._calculate_sma(bars)
            
            current = bars.iloc[-1]
            previous = bars.iloc[-2]
            current_price = current['close']
            
            # Check if we can afford to buy
            can_buy = cash >= (current_price * self._trade_quantity)
            
            # Generate signals based on SMA crossover
            if (current['SMA_short'] > current['SMA_long'] and 
                previous['SMA_short'] <= previous['SMA_long'] and
                can_buy):
                # Buy signal
                return MarketDecision(symbol=symbol, action=MarketAction.BUY,
                                    price=current_price, quantity=self._trade_quantity)
                                    
            elif (current['SMA_short'] < current['SMA_long'] and 
                  previous['SMA_short'] >= previous['SMA_long'] and
                  position >= self._trade_quantity):
                # Sell signal
                return MarketDecision(symbol=symbol, action=MarketAction.SELL,
                                    price=current_price, quantity=self._trade_quantity)
            
            # No signal or unable to execute
            return MarketDecision(symbol=symbol, action=MarketAction.HOLD,
                                price=current_price, quantity=0)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate signal for {symbol}: {str(e)}")
        