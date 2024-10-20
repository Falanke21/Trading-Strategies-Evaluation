import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockLiveDataClient
import pandas_ta as ta
from interface import MarketAction, MarketDecision, IStrategy

class MACDStrategy(IStrategy):
    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self.live_client = StockLiveDataClient(api_key, api_secret)

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate MACD using pandas_ta library
        macd = ta.macd(data['close'])
        # Concatenate the MACD columns with the original data
        data = pd.concat([data, macd], axis=1)
        return data

    def get_latest_price(self, symbol: str) -> float:
        # Fetch the latest price using the live client
        latest_trade = self.live_client.get_latest_trade(symbol)
        return latest_trade.price

    def swing_trade(self, symbol='AAPL') -> MarketDecision:
        # Define the date range for fetching historical data (last 180 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Set up the request parameters for fetching stock bars
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        # Fetch historical stock data
        bars = self.client.get_stock_bars(request_params).df
        # Calculate MACD values for the fetched data
        bars = self.calculate_macd(bars)
        
        # Get the latest and previous data points
        latest_data = bars.iloc[-1]
        previous_data = bars.iloc[-2]
        
        # Check if the market is open and fetch the latest price if it is
        try:
            latest_price = self.get_latest_price(symbol)
        except Exception as e:
            # If fetching the latest price fails, fall back to the close price
            latest_price = latest_data['close']
        
        # Determine the trading action based on MACD crossover
        if latest_data['MACD_12_26_9'] > latest_data['MACDs_12_26_9'] and previous_data['MACD_12_26_9'] <= previous_data['MACDs_12_26_9']:
            action = MarketAction.BUY
        elif latest_data['MACD_12_26_9'] < latest_data['MACDs_12_26_9'] and previous_data['MACD_12_26_9'] >= previous_data['MACDs_12_26_9']:
            action = MarketAction.SELL
        else:
            action = MarketAction.HOLD
        
        # Return the market decision with the determined action, latest price, and a fixed quantity
        return MarketDecision(symbol=symbol, action=action, price=latest_price, quantity=100)

