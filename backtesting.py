import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from strategy.buy_and_hold import BuyAndHoldStrategy
from strategy.sma import SMAStrategy
from strategy.macd import MACDStrategy
from strategy.kdj import KDJStrategy
from strategy.enhanced_macd import EnhancedMACDStrategy
from strategy.quantitative_adative import QuantitativeAdaptiveStrategy
from interface import IStrategy, MarketAction  # Updated import
from metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_beta, calculate_alpha, calculate_win_rate, calculate_profit_factor
import config  # Import the config file
from tqdm import tqdm  # Import tqdm for progress bar
import os

STRATEGY = QuantitativeAdaptiveStrategy
SYMBOL = 'AAPL'

def plot_backtest_results(portfolio_df, strategy_name, symbol):
    """
    Plot the backtest results comparing portfolio value to stock price performance
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot both normalized series on the same axis
    ax1.plot(portfolio_df['Date'], portfolio_df['Portfolio Value %'], 
             label='Portfolio Value', color='blue')
    ax1.plot(portfolio_df['Date'], portfolio_df['Stock Price %'], 
             label=f'{symbol} Stock Price', color='orange')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value (% of initial)')
    
    # Add title and legend
    plt.title(f"{strategy_name} Strategy: Portfolio Value and {symbol} Stock Price Performance (%)")
    plt.legend(loc='upper left')
    
    # Save the plot to the results directory
    plot_path = f'result_backtest/plots/{symbol}_{strategy_name}_portfolio_value.png'
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")

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
    dates = []
    
    # Get all dates in range
    date_range = pd.date_range(start=start_date, end=end_date)
    # Filter to only include Monday to Friday
    trading_days = [d for d in date_range if d.weekday() < 5]
    
    # Initialize progress bar
    pbar = tqdm(trading_days, desc=f"Backtesting {STRATEGY.__name__} Progress", dynamic_ncols=True)
    
    # Lists to store returns
    daily_returns = []
    market_returns = []
    previous_value = initial_cash
    previous_market_price = None
    
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
            dates.append(single_date)
            
            # Calculate daily strategy return
            daily_return = (portfolio_value - previous_value) / previous_value
            
            # Calculate daily market return (using stock price as market proxy)
            if previous_market_price is not None:
                market_return = (decision.price - previous_market_price) / previous_market_price
                market_returns.append(market_return)
                daily_returns.append(daily_return)
            
            previous_value = portfolio_value
            previous_market_price = decision.price
            
            # Update progress bar description
            profit = portfolio_value - initial_cash
            pbar.set_postfix({
                'Date': single_date.strftime('%Y-%m-%d'),
                'Position': position,
                'Profit': f'${profit:.2f}'
            })
        except Exception as e:
            print(f"Error on {single_date}: {e}")
            continue
    
    # Create a DataFrame for portfolio values and stock prices
    portfolio_df = pd.DataFrame(data={
        'Date': dates,
        'Portfolio Value': portfolio_values,
        'Stock Price': stock_prices
    })
    
    # Normalize both series to start at 100%
    initial_portfolio = portfolio_df['Portfolio Value'].iloc[0]
    initial_stock = portfolio_df['Stock Price'].iloc[0]
    
    portfolio_df['Portfolio Value %'] = (portfolio_df['Portfolio Value'] / initial_portfolio) * 100
    portfolio_df['Stock Price %'] = (portfolio_df['Stock Price'] / initial_stock) * 100
    
    # Calculate metrics
    returns_series = pd.Series(daily_returns)
    market_returns_series = pd.Series(market_returns)
    
    # Verify lengths match
    if len(returns_series) != len(market_returns_series):
        print(f"Warning: Returns series lengths don't match: {len(returns_series)} vs {len(market_returns_series)}")
        # Trim to same length if necessary
        min_length = min(len(returns_series), len(market_returns_series))
        returns_series = returns_series[:min_length]
        market_returns_series = market_returns_series[:min_length]
    
    # Calculate all metrics
    sharpe_ratio = calculate_sharpe_ratio(returns_series)
    max_dd, max_dd_duration = calculate_max_drawdown(portfolio_values)
    beta = calculate_beta(returns_series, market_returns_series)
    alpha = calculate_alpha(returns_series, market_returns_series)
    win_rate = calculate_win_rate(returns_series)
    profit_factor = calculate_profit_factor(returns_series)
    
    # Create metrics text
    metrics_text = f"""Performance Metrics for {STRATEGY.__name__}:

Sharpe Ratio: {sharpe_ratio:.2f}
    > 1.0: Good | > 2.0: Very Good | > 3.0: Excellent
    Measures risk-adjusted returns. Higher is better.

Maximum Drawdown: {max_dd:.2%}
Maximum Drawdown Duration: {max_dd_duration} days
    Shows worst peak-to-trough decline in portfolio value
    Example: {max_dd:.2%} means a ${initial_cash:,.2f} portfolio
    would have dropped to ${initial_cash * (1-max_dd):,.2f} at its lowest point
    Shorter drawdown durations and smaller drawdowns are better
    Useful for understanding worst-case scenarios and risk tolerance

Beta: {beta:.2f}
    1.0: Moves with market
    > 1.0: More volatile than market
    < 1.0: Less volatile than market

Alpha: {alpha:.2%}
    > 0: Outperforming the market
    = 0: Matching the market
    < 0: Underperforming the market

Win Rate: {win_rate:.2%}
    > 50%: More winning trades than losing trades
    Note: Should be considered alongside profit factor

Profit Factor: {profit_factor:.2f}
    > 1.0: Profitable
    > 2.0: Good
    > 3.0: Excellent
    Shows how much profit per unit of risk
"""
    
    # Print metrics to console and save to file
    print(metrics_text)
    save_metrics_to_file(metrics_text, STRATEGY.__name__)
    
    # Plot the results using the helper function
    plot_backtest_results(portfolio_df, STRATEGY.__name__, SYMBOL)

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("result_backtest/plots", exist_ok=True)
    os.makedirs("result_backtest/metrics", exist_ok=True)

def save_metrics_to_file(metrics_text: str, strategy_name: str):
    """Save metrics to a text file"""
    filename = f"result_backtest/metrics/{SYMBOL}_{strategy_name}_metrics.txt"
    with open(filename, 'w') as f:
        f.write(f"Symbol: {SYMBOL}\n\n{metrics_text}")
    print(f"Metrics saved to {filename}")

if __name__ == '__main__':
    # Create necessary directories
    ensure_directories()
    
    # Initialize the strategy
    if STRATEGY == BuyAndHoldStrategy:
        strategy = BuyAndHoldStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting Buy and Hold strategy")
    elif STRATEGY == SMAStrategy:
        strategy = SMAStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting SMA strategy")
    elif STRATEGY == MACDStrategy:
        strategy = MACDStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting MACD strategy")
    elif STRATEGY == KDJStrategy:
        strategy = KDJStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting KDJ strategy")
    elif STRATEGY == EnhancedMACDStrategy:
        strategy = EnhancedMACDStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting EnhancedMACD strategy")
    elif STRATEGY == QuantitativeAdaptiveStrategy:
        strategy = QuantitativeAdaptiveStrategy(api_key=config.API_KEY, api_secret=config.SECRET_KEY)
        print("Backtesting QuantitativeAdaptive strategy")
    else:
        raise ValueError(f"Unsupported strategy: {STRATEGY}")
    
    backtest_strategy(strategy)
