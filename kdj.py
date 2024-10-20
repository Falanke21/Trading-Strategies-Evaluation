import pandas as pd
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime

# Replace these with your Alpaca API credentials
API_KEY = ""
SECRET_KEY = ""

def calculate_kdj_df(symbol='AAPL', 
                     start_date=datetime.datetime(2023, 1, 1), 
                     end_date=datetime.datetime(2023, 9, 1)):
    # Initialize Alpaca's stock historical data client
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # Request historical stock data (daily bars)
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )

    # Fetch the data and convert it to a pandas DataFrame
    bars = client.get_stock_bars(request_params).df

    # Calculate KDJ using pandas_ta
    # pandas_ta's stoch (Stochastic Oscillator) can give you %K and %D, and we can compute %J
    stoch = ta.stoch(bars['high'], bars['low'], bars['close'], k=14, d=3, smooth_k=3)

    # Assign %K and %D from stoch
    bars['%K'] = stoch['STOCHk_14_3_3']  # %K line
    bars['%D'] = stoch['STOCHd_14_3_3']  # %D line

    # Calculate %J (which is often defined as 3 * %K - 2 * %D)
    bars['%J'] = 3 * bars['%K'] - 2 * bars['%D']

    # Show the last few rows of data including %K, %D, and %J
    # print(bars[['close', '%K', '%D', '%J']].tail())

    # Example logic for generating buy/sell signals using KDJ
    def generate_kdj_signals(df):
        # Buy when %K crosses above %D (bullish signal)
        df['Buy_Signal'] = (df['%K'] > df['%D']) & (df['%K'].shift(1) <= df['%D'].shift(1))
        
        # Sell when %K crosses below %D (bearish signal)
        df['Sell_Signal'] = (df['%K'] < df['%D']) & (df['%K'].shift(1) >= df['%D'].shift(1))
        return df

    # Apply signal generation logic
    bars = generate_kdj_signals(bars)
    return bars

if __name__ == '__main__':
    bars = calculate_kdj_df()
    # Display the buy/sell signals
    print(bars[['close', '%K', '%D', '%J', 'Buy_Signal', 'Sell_Signal']].tail(20))
