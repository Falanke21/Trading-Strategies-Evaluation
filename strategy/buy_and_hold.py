import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from interface import MarketAction, MarketDecision, IStrategy

class BuyAndHoldStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the Buy and Hold strategy.
        This strategy will buy as many shares as possible with available cash on day one,
        and then hold those shares indefinitely.
        """
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret)
        self._first_trade = True

    def generate_signal(self, symbol='AAPL', date=None, position=0, cash=0.0) -> MarketDecision:
        """
        Generate trading signal for Buy and Hold strategy:
        - On first trade: Buy maximum shares possible with available cash
        - Afterwards: Always hold
        """
        if date is None:
            date = datetime.now()
            
        # Look back a few days to ensure we get data even around weekends/holidays
        start_date = date - timedelta(days=5)
        end_date = date + timedelta(days=1)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        try:
            bars = self._client.get_stock_bars(request_params).df
            if len(bars) == 0:
                raise RuntimeError("No market data available for this date range")
                
            # Get the last price up to our target date
            bars = bars[bars.index.get_level_values(1).date <= date.date()]
            if len(bars) == 0:
                raise RuntimeError("No market data available up to this date")
                
            current_price = bars['close'].iloc[-1]
            
            # If this is our first trade and we have cash, buy maximum shares possible
            if self._first_trade and cash > current_price:
                max_shares = int(cash / current_price)  # Integer number of shares we can afford
                if max_shares > 0:
                    self._first_trade = False
                    return MarketDecision(
                        symbol=symbol,
                        action=MarketAction.BUY,
                        price=current_price,
                        quantity=max_shares
                    )
            
            # After first trade or if we can't afford any shares, just hold
            return MarketDecision(
                symbol=symbol,
                action=MarketAction.HOLD,
                price=current_price,
                quantity=0
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate signal for {symbol}: {str(e)}")
        