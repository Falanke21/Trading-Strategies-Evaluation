import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import pandas_ta as ta
from interface import MarketAction, MarketDecision, IStrategy

class EnhancedMACDStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str,
                 lookback_days: int = 180,
                 trade_quantity: int = 100,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 rsi_period: int = 14,
                 rsi_overbought: int = 75,
                 rsi_oversold: int = 25,
                 sma_period: int = 50):
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret)
        self._lookback_days = lookback_days
        self._trade_quantity = trade_quantity
        self._macd_params = (macd_fast, macd_slow, macd_signal)
        self._rsi_params = (rsi_period, rsi_overbought, rsi_oversold)
        self._sma_period = sma_period

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate MACD
        macd = ta.macd(data['close'])
        if macd is None or pd.isna(macd.iloc[-1]).any():  # Changed from isna() to pd.isna()
            raise ValueError("Failed to calculate MACD indicators")
        
        # Calculate RSI
        rsi_period, _, _ = self._rsi_params
        rsi = ta.rsi(data['close'], length=rsi_period)
        if rsi is None or pd.isna(rsi.iloc[-1]):  # Changed from isna() to pd.isna()
            raise ValueError("Failed to calculate RSI indicator")
        
        # Calculate 200 SMA
        sma = ta.sma(data['close'], length=self._sma_period)
        if sma is None or pd.isna(sma.iloc[-1]):  # Changed from isna() to pd.isna()
            raise ValueError("Failed to calculate SMA indicator")
        
        # Add all indicators to the dataframe
        data = pd.concat([data, macd, rsi.rename('RSI'), sma.rename('SMA')], axis=1)
        return data

    def generate_signal(self, symbol='AAPL', date=None, position=0, cash=0.0) -> MarketDecision:
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
            if len(bars) < self._sma_period:
                return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                    price=bars['close'].iloc[-1], quantity=0)
            
            bars = self._calculate_indicators(bars)
            
            current_bar = bars.iloc[-1]
            previous_bar = bars.iloc[-2]
            current_price = current_bar['close']
            
            can_buy = cash >= (current_price * self._trade_quantity)
            
            rsi_period, rsi_overbought, rsi_oversold = self._rsi_params
            
            # Modified signal generation with less restrictive conditions
            if (current_bar['MACD_12_26_9'] > current_bar['MACDs_12_26_9'] and 
                previous_bar['MACD_12_26_9'] <= previous_bar['MACDs_12_26_9']):
                # Primary condition: MACD crossover
                
                # Secondary confirmations (need only 1 out of 2)
                confirmations = 0
                if current_bar['RSI'] < rsi_overbought:  # Not overbought
                    confirmations += 1
                if current_price > current_bar['SMA']:  # Above 200 SMA
                    confirmations += 1
                
                if confirmations >= 1 and can_buy:  # Need only one confirmation
                    return MarketDecision(symbol=symbol, action=MarketAction.BUY, 
                                        price=current_price, quantity=self._trade_quantity)
                                        
            elif (current_bar['MACD_12_26_9'] < current_bar['MACDs_12_26_9'] and 
                  previous_bar['MACD_12_26_9'] >= previous_bar['MACDs_12_26_9'] and
                  (current_bar['RSI'] > rsi_oversold or  # Either oversold
                   current_price < current_bar['SMA'])):  # or below 200 SMA
                if position >= self._trade_quantity:
                    return MarketDecision(symbol=symbol, action=MarketAction.SELL, 
                                        price=current_price, quantity=self._trade_quantity)
            
            return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                price=current_price, quantity=0)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate signal for {symbol}: {str(e)}")
