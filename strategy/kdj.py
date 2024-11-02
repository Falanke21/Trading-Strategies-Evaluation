import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import pandas_ta as ta
from interface import MarketAction, MarketDecision, IStrategy

class KDJStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str,
                 lookback_days: int = 180,
                 trade_quantity: int = 100,
                 k_period: int = 14,
                 d_period: int = 3,
                 smooth_k: int = 3):
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret)
        self._lookback_days = lookback_days
        self._trade_quantity = trade_quantity
        self._kdj_params = (k_period, d_period, smooth_k)

    def _calculate_kdj(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate KDJ using pandas_ta
        k_period, d_period, smooth_k = self._kdj_params
        stoch = ta.stoch(data['high'], data['low'], data['close'], 
                        k=k_period, d=d_period, smooth_k=smooth_k)
        
        if stoch is None:
            raise ValueError("Failed to calculate KDJ indicators")
            
        # Verify we have valid Stochastic values
        if stoch.iloc[-1].isna().any():
            raise ValueError("KDJ calculation resulted in NaN values")
        
        # Add KDJ indicators to the dataframe
        data['%K'] = stoch['STOCHk_14_3_3']
        data['%D'] = stoch['STOCHd_14_3_3']
        data['%J'] = 3 * data['%K'] - 2 * data['%D']
        
        return data

    def generate_signal(self, symbol='AAPL', date=None, position=0, cash=0.0) -> MarketDecision:
        """
        Generates trading signals using KDJ (Stochastic Oscillator) strategy.
        
        Args:
            symbol (str): The stock symbol to analyze
            date (datetime, optional): The date to analyze the trading decision for
            position (int): Current number of shares held (not used in basic KDJ)
            cash (float): Available cash for trading (not used in basic KDJ)
        
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
                
            bars = self._calculate_kdj(bars)
            
            current_bar = bars.iloc[-1]
            previous_bar = bars.iloc[-2]
            
            analysis_price = current_bar['close']
            
            # Check if we can afford to buy
            can_buy = cash >= (analysis_price * self._trade_quantity)
            
            # Generate signals based on K line crossing D line
            if (current_bar['%K'] > current_bar['%D'] and 
                previous_bar['%K'] <= previous_bar['%D']):
                # Buy signal - use fixed quantity if we have enough cash
                if can_buy:
                    return MarketDecision(symbol=symbol, action=MarketAction.BUY, 
                                        price=analysis_price, quantity=self._trade_quantity)
            elif (current_bar['%K'] < current_bar['%D'] and 
                  previous_bar['%K'] >= previous_bar['%D']):
                # Sell signal - use fixed quantity if we have enough position
                if position >= self._trade_quantity:
                    return MarketDecision(symbol=symbol, action=MarketAction.SELL, 
                                        price=analysis_price, quantity=self._trade_quantity)
            
            # No signal or unable to execute
            return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                price=analysis_price, quantity=0)
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate signal for {symbol}: {str(e)}")
