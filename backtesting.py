import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from macd import MACDStrategy
from interface import MarketAction
import config  # Import the config file
from tqdm import tqdm  # Import tqdm for progress bar

SYMBOL = 'AAPL'

def backtest_macd_strategy():
    # Initialize the MACD strategy
    macd_strategy = MACDStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
    
    # Define the date range for backtesting (last 365 days)
    end_date = datetime.now() - timedelta(minutes=15)  # Exclude the last 15 minutes
    start_date = end_date - timedelta(days=365)
    
    # Initialize portfolio value and cash
    initial_cash = 100000  # Starting with $100,000
    portfolio_value = initial_cash
    cash = initial_cash
    position = 0  # Number of shares held
    
    # List to store portfolio values and stock prices over time
    portfolio_values = []
    stock_prices = []
    
    # Initialize progress bar
    pbar = tqdm(pd.date_range(start=start_date, end=end_date), desc="Backtesting Progress", dynamic_ncols=True)
    
    # Simulate trading
    for single_date in pbar:
        try:
            # Use the swing_trade method to get the trading decision for the current date
            decision = macd_strategy.swing_trade(SYMBOL, end_date=single_date)
            
            if decision.action == MarketAction.BUY and cash >= decision.price * decision.quantity:
                cash -= decision.price * decision.quantity
                position += decision.quantity
            elif decision.action == MarketAction.SELL and position >= decision.quantity:
                cash += decision.price * decision.quantity
                position -= decision.quantity
            
            # Calculate current portfolio value
            portfolio_value = cash + position * decision.price
            portfolio_values.append(portfolio_value)
            stock_prices.append(decision.price)
            
            # Update progress bar description
            profit = portfolio_value - initial_cash
            pbar.set_postfix({
                'Date': single_date.strftime('%Y-%m-%d'),
                'Position': position,
                'Profit': f'${profit:.2f}'
            })
        except Exception as e:
            print(f"Error on {single_date}: {e}")
            portfolio_values.append(portfolio_value)  # Append the last known portfolio value
            stock_prices.append(stock_prices[-1] if stock_prices else 0)  # Append the last known stock price or 0 if empty
    
    # Create a DataFrame for portfolio values and stock prices
    portfolio_df = pd.DataFrame(data={
        'Date': pd.date_range(start=start_date, end=end_date),
        'Portfolio Value': portfolio_values,
        'Stock Price': stock_prices
    })
    
    # Plotting the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot portfolio value on the primary y-axis
    ax1.plot(portfolio_df['Date'], portfolio_df['Portfolio Value'], label='Portfolio Value', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for the stock price
    ax2 = ax1.twinx()
    ax2.plot(portfolio_df['Date'], portfolio_df['Stock Price'], label=f'{SYMBOL} Stock Price', color='orange')
    ax2.set_ylabel(f'{SYMBOL} Stock Price ($)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add title and legend
    plt.title(f"Portfolio Value and {SYMBOL} Stock Price Over Time")
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # Save and show the plot
    plt.savefig('portfolio_value_macd.png')

if __name__ == '__main__':
    backtest_macd_strategy()
