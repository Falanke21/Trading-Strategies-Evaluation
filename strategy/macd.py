import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import pandas_ta as ta
from interface import MarketAction, MarketDecision, IStrategy

class MACDStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str, 
                 lookback_days: int = 180,
                 trade_quantity: int = 100,  # Fixed quantity to trade
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret)
        self._lookback_days = lookback_days
        self._trade_quantity = trade_quantity
        self._macd_params = (macd_fast, macd_slow, macd_signal)

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate MACD using pandas_ta library
        macd = ta.macd(data['close'])
        if macd is None:
            raise ValueError("Failed to calculate MACD indicators")
            
        # Verify we have valid MACD values
        if macd.iloc[-1].isna().any():
            raise ValueError("MACD calculation resulted in NaN values")
            
        # Concatenate the MACD columns with the original data
        data = pd.concat([data, macd], axis=1)
        return data

    def generate_signal(self, symbol='AAPL', date=None, position=0, cash=0.0) -> MarketDecision:
        """
        Generates trading signals using MACD crossover strategy with position awareness.
        
        Args:
            symbol (str): The stock symbol to analyze
            date (datetime, optional): The date to analyze the trading decision for
            position (int): Current number of shares held
            cash (float): Available cash for trading
        
        Returns:
            MarketDecision: A decision object containing the trading action and quantity
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
            if len(bars) < 2:
                return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                    price=bars['close'].iloc[-1], quantity=0)
            
            # Calculate MACD values for the fetched data
            bars = self._calculate_macd(bars)
            
            # Get the current and previous data points for analysis
            current_bar = bars.iloc[-1]
            previous_bar = bars.iloc[-2]
            current_price = current_bar['close']
            
            # Check if we can afford to buy
            can_buy = cash >= (current_price * self._trade_quantity)
            
            # Generate trading signals based on MACD crossover
            if (current_bar['MACD_12_26_9'] > current_bar['MACDs_12_26_9'] and 
                previous_bar['MACD_12_26_9'] <= previous_bar['MACDs_12_26_9']):
                # Buy signal - use fixed quantity if we have enough cash
                if can_buy:
                    return MarketDecision(symbol=symbol, action=MarketAction.BUY, 
                                        price=current_price, quantity=self._trade_quantity)
                                        
            elif (current_bar['MACD_12_26_9'] < current_bar['MACDs_12_26_9'] and 
                  previous_bar['MACD_12_26_9'] >= previous_bar['MACDs_12_26_9']):
                # Sell signal - use fixed quantity if we have enough position
                if position >= self._trade_quantity:
                    return MarketDecision(symbol=symbol, action=MarketAction.SELL, 
                                        price=current_price, quantity=self._trade_quantity)
            
            # No signal or unable to execute
            return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                price=current_price, quantity=0)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate signal for {symbol}: {str(e)}")
