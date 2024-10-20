import datetime
import matplotlib.pyplot as plt

from macd import calculate_macd_df
from kdj import calculate_kdj_df

# Backtesting function
def backtest_strategy(df, strategy_name, buy_column, sell_column):
    # Initialize variables
    initial_capital = 10000  # Starting with $10,000
    shares = 0
    capital = initial_capital
    portfolio_value = []
    buy_price = 0
    trade_log = []  # To log trades
    
    # Loop through the DataFrame and simulate trading
    for index, row in df.iterrows():
        if row[buy_column] and capital > 0:  # Buy signal
            shares = capital / row['close']  # Buy as many shares as we can with the available capital
            buy_price = row['close']
            capital = 0  # All capital is now in shares
            trade_log.append((index, 'Buy', buy_price, shares))
        
        elif row[sell_column] and shares > 0:  # Sell signal
            capital = shares * row['close']  # Sell all shares
            sell_price = row['close']
            profit = (sell_price - buy_price) / buy_price * 100  # Percentage profit on this trade
            shares = 0  # Sold all shares
            trade_log.append((index, 'Sell', sell_price, capital, profit))
        
        # Track portfolio value (if holding shares, value = shares * current price; otherwise, cash)
        portfolio_value.append(capital if shares == 0 else shares * row['close'])
    
    # Calculate cumulative returns
    df['Portfolio Value'] = portfolio_value
    cumulative_return = (df['Portfolio Value'].iloc[-1] - initial_capital) / initial_capital * 100

    # Print the strategy performance
    print(f"Strategy: {strategy_name}")
    print(f"Initial Capital: ${initial_capital}")
    print(f"Final Portfolio Value: ${df['Portfolio Value'].iloc[-1]:.2f}")
    print(f"Cumulative Return: {cumulative_return:.2f}%")
    
    # Print trade logs
    print("\nTrade Logs:")
    for trade in trade_log:
        symbol, timestamp = trade[0]
        date = timestamp.strftime('%Y-%m-%d')
        action = trade[1]
        price = "{:.2f}".format(trade[2])
        print(f"{symbol} - {date} - {action} @ ${price}")
        if trade[1] == 'Sell':
            print(f"  Profit on trade: {trade[4]:.2f}%")
    print()
    
    return df, cumulative_return

def main():
    SYMBOL = 'AAPL'
    START_DATE = datetime.datetime(2023, 1, 1)
    END_DATE = datetime.datetime(2023, 9, 1)
    df_macd = calculate_macd_df(SYMBOL, START_DATE, END_DATE)
    df_kdj = calculate_kdj_df(SYMBOL, START_DATE, END_DATE)

    # Backtest MACD strategy
    bars_with_macd, macd_return = backtest_strategy(
        df_macd.copy(), 
        "MACD Strategy", 
        buy_column='Buy_Signal', 
        sell_column='Sell_Signal'
    )

    # Backtest KDJ strategy
    bars_with_kdj, kdj_return = backtest_strategy(
        df_kdj.copy(), 
        "KDJ Strategy", 
        buy_column='Buy_Signal',  # Buy when %K crosses above %D
        sell_column='Sell_Signal'  # Sell when %K crosses below %D
    )

    # Compare performance
    print(f"\nMACD Strategy Cumulative Return: {macd_return:.2f}%")
    print(f"KDJ Strategy Cumulative Return: {kdj_return:.2f}%")

    # bars_with_macd.index is a multi-index with the first level being the symbol 
    # and the second level being the timestamp

    # Plotting the results
    plt.figure(figsize=(10, 6))
    # Plot MACD strategy portfolio value
    plt.plot(bars_with_macd.index.get_level_values(1), bars_with_macd['Portfolio Value'], label='MACD Strategy', color='blue')
    # Plot KDJ strategy portfolio value
    plt.plot(bars_with_kdj.index.get_level_values(1), bars_with_kdj['Portfolio Value'], label='KDJ Strategy', color='green')
    # Add labels and title
    plt.title(f"Portfolio Value Over Time for {SYMBOL}")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.savefig('portfolio_value.png')

if __name__ == '__main__':
    main()
