import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from macd import MACDStrategy
from kdj import KDJStrategy
from enhanced_macd import EnhancedMACDStrategy
from interface import IStrategy, MarketAction  # Updated import
import config  # Import the config file
from tqdm import tqdm  # Import tqdm for progress bar

STRATEGY = EnhancedMACDStrategy
SYMBOL = 'AAPL'

def backtest_strategy(strategy: IStrategy):
    # Set the date range for backtesting (last 365 days, ending a week before today)
    end_date = datetime.now() - timedelta(days=7)
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
    pbar = tqdm(pd.date_range(start=start_date, end=end_date), desc=f"Backtesting {STRATEGY.__name__} Progress", dynamic_ncols=True)
    
    # Simulate trading
    for single_date in pbar:
        try:
            # Use the generate_signal method to get the trading decision for the current date
            decision = strategy.generate_signal(SYMBOL, date=single_date, position=position, cash=cash)
            
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
    
    # Normalize both series to start at 100%
    initial_portfolio = portfolio_df['Portfolio Value'].iloc[0]
    initial_stock = portfolio_df['Stock Price'].iloc[0]
    
    portfolio_df['Portfolio Value %'] = (portfolio_df['Portfolio Value'] / initial_portfolio) * 100
    portfolio_df['Stock Price %'] = (portfolio_df['Stock Price'] / initial_stock) * 100
    
    # Plotting the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot both normalized series on the same axis
    ax1.plot(portfolio_df['Date'], portfolio_df['Portfolio Value %'], 
             label='Portfolio Value', color='blue')
    ax1.plot(portfolio_df['Date'], portfolio_df['Stock Price %'], 
             label=f'{SYMBOL} Stock Price', color='orange')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value (% of initial)')
    
    # Add title and legend
    plt.title(f"{STRATEGY.__name__} Strategy: Portfolio Value and {SYMBOL} Stock Price Performance (%)")
    plt.legend(loc='upper left')
    
    # Save and show the plot
    plt.savefig(f'portfolio_value_{STRATEGY.__name__}.png')
    print(f"Plot saved as portfolio_value_{STRATEGY.__name__}.png")

if __name__ == '__main__':
    # Initialize the strategy
    if STRATEGY == MACDStrategy:
        strategy = MACDStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting MACD strategy")
    elif STRATEGY == KDJStrategy:
        strategy = KDJStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting KDJ strategy")
    elif STRATEGY == EnhancedMACDStrategy:
        strategy = EnhancedMACDStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting EnhancedMACD strategy")
    else:
        raise ValueError(f"Unsupported strategy: {STRATEGY}")
    
    backtest_strategy(strategy)
