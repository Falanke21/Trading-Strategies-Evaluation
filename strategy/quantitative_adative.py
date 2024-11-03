import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import pandas_ta as ta
from interface import MarketAction, MarketDecision, IStrategy

class QuantitativeAdaptiveStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str,
                 lookback_days: int = 180,
                 trade_quantity: int = 100,
                 volatility_window: int = 20,
                 bollinger_length: int = 20,
                 bollinger_std: float = 2.0,
                 volume_ma_period: int = 20,
                 rsi_period: int = 14,
                 rsi_overbought: int = 70,
                 rsi_oversold: int = 30,
                 atr_period: int = 14,
                 regime_period: int = 50):
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret)
        self._lookback_days = lookback_days
        self._trade_quantity = trade_quantity
        self._volatility_window = volatility_window
        self._bollinger_params = (bollinger_length, bollinger_std)
        self._volume_ma_period = volume_ma_period
        self._rsi_params = (rsi_period, rsi_overbought, rsi_oversold)
        self._atr_period = atr_period
        self._regime_period = regime_period

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate Bollinger Bands
        bb_length, bb_std = self._bollinger_params
        bb = ta.bbands(data['close'], length=bb_length, std=bb_std)
        
        # Calculate RSI
        rsi_period, _, _ = self._rsi_params
        rsi = ta.rsi(data['close'], length=rsi_period)
        
        # Calculate ATR for volatility
        atr = ta.atr(data['high'], data['low'], data['close'], length=self._atr_period)
        
        # Calculate Volume Moving Average
        volume_ma = ta.sma(data['volume'], length=self._volume_ma_period)
        
        # Calculate Market Regime using Hull Moving Average
        hull_ma = ta.hma(data['close'], length=self._regime_period)
        
        # Calculate Rate of Change
        roc = ta.roc(data['close'], length=10)
        
        # Calculate Money Flow Index
        mfi = ta.mfi(data['high'], data['low'], data['close'], data['volume'], length=14)
        
        # Combine all indicators
        data = pd.concat([
            data,
            bb,
            rsi.rename('RSI'),
            atr.rename('ATR'),
            volume_ma.rename('Volume_MA'),
            hull_ma.rename('HMA'),
            roc.rename('ROC'),
            mfi.rename('MFI')
        ], axis=1)
        
        # Calculate volatility regime
        data['Volatility'] = data['close'].pct_change().rolling(self._volatility_window).std()
        
        return data

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        current = data.iloc[-1]
        
        # Determine trend using Hull MA
        trend = "uptrend" if current['close'] > current['HMA'] else "downtrend"
        
        # Determine volatility regime
        vol_percentile = data['Volatility'].quantile(0.7)
        volatility = "high" if current['Volatility'] > vol_percentile else "low"
        
        return f"{trend}_{volatility}"

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
            if len(bars) < self._regime_period:
                return MarketDecision(symbol=symbol, action=MarketAction.HOLD, 
                                    price=bars['close'].iloc[-1], quantity=0)
            
            bars = self._calculate_indicators(bars)
            current = bars.iloc[-1]
            previous = bars.iloc[-2]
            current_price = current['close']
            
            # Determine if we can afford to trade
            can_buy = cash >= (current_price * self._trade_quantity)
            
            # Get market regime
            regime = self._detect_market_regime(bars)
            
            # Define entry/exit thresholds based on regime
            rsi_period, rsi_overbought, rsi_oversold = self._rsi_params
            
            # Adjust thresholds based on market regime
            if regime == "uptrend_low":
                rsi_oversold += 5
                rsi_overbought += 5
            elif regime == "downtrend_high":
                rsi_oversold -= 5
                rsi_overbought -= 5
            
            # Generate trading signals
            buy_signals = [
                current['close'] < current['BBL_20_2.0'],  # Price below lower Bollinger Band
                current['RSI'] < rsi_oversold,  # RSI oversold
                current['volume'] > current['Volume_MA'] * 1.5,  # High volume
                current['MFI'] < 30,  # Money Flow Index oversold
                current['ROC'] > 0  # Positive Rate of Change
            ]
            
            sell_signals = [
                current['close'] > current['BBU_20_2.0'],  # Price above upper Bollinger Band
                current['RSI'] > rsi_overbought,  # RSI overbought
                current['volume'] > current['Volume_MA'] * 1.5,  # High volume
                current['MFI'] > 70,  # Money Flow Index overbought
                current['ROC'] < 0  # Negative Rate of Change
            ]
            
            # Calculate signal strength
            buy_strength = sum(buy_signals) / len(buy_signals)
            sell_strength = sum(sell_signals) / len(sell_signals)
            
            # Make trading decision
            if buy_strength >= 0.6 and can_buy:  # At least 60% of buy signals are true
                return MarketDecision(symbol=symbol, action=MarketAction.BUY,
                                    price=current_price, quantity=self._trade_quantity)
            elif sell_strength >= 0.6 and position >= self._trade_quantity:
                return MarketDecision(symbol=symbol, action=MarketAction.SELL,
                                    price=current_price, quantity=self._trade_quantity)
            
            return MarketDecision(symbol=symbol, action=MarketAction.HOLD,
                                price=current_price, quantity=0)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate signal for {symbol}: {str(e)}")
        