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
import config  # Import the config file
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np

STRATEGY = KDJStrategy
SYMBOL = 'AAPL'

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio for a series of returns
    Args:
        returns: pandas Series of daily returns
        risk_free_rate: Annual risk-free rate (default 2%)
    """
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate/252
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate annualized Sharpe Ratio
    if excess_returns.std() == 0:
        return 0
    
    sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    return sharpe

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the Maximum Drawdown of a portfolio
    Args:
        portfolio_values: List of portfolio values over time
    Returns:
        max_dd: Maximum drawdown as a percentage
        max_dd_duration: Duration of the maximum drawdown in days
    """
    peak = portfolio_values[0]
    max_dd = 0
    max_dd_duration = 0
    current_dd_duration = 0
    peak_idx = 0
    
    for i, value in enumerate(portfolio_values):
        if value > peak:
            peak = value
            peak_idx = i
            current_dd_duration = 0
        else:
            current_dd_duration = i - peak_idx
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = current_dd_duration
                
    return max_dd, max_dd_duration

def calculate_beta(strategy_returns, market_returns):
    """
    Calculate Beta (market sensitivity) of the strategy
    Args:
        strategy_returns: pandas Series of strategy returns
        market_returns: pandas Series of market returns
    Returns:
        beta: Strategy's beta coefficient
    """
    # Calculate covariance between strategy and market returns
    covariance = np.cov(strategy_returns, market_returns)[0][1]
    # Calculate market variance
    market_variance = np.var(market_returns)
    # Calculate beta
    return covariance / market_variance if market_variance != 0 else 1.0

def calculate_alpha(strategy_returns, market_returns, risk_free_rate=0.02):
    """
    Calculate Alpha (excess return) of the strategy
    Args:
        strategy_returns: pandas Series of strategy returns
        market_returns: pandas Series of market returns
        risk_free_rate: Annual risk-free rate (default 2%)
    Returns:
        alpha: Strategy's alpha (annualized)
    """
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate/252
    
    # Calculate beta
    beta = calculate_beta(strategy_returns, market_returns)
    
    # Calculate annualized returns
    strategy_return = strategy_returns.mean() * 252
    market_return = market_returns.mean() * 252
    
    # Calculate alpha using CAPM formula
    alpha = strategy_return - (daily_rf * 252 + beta * (market_return - daily_rf * 252))
    return alpha

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
    
    # Save and show the plot
    plt.savefig(f'portfolio_value_{strategy_name}.png')
    print(f"Plot saved as portfolio_value_{strategy_name}.png")

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
    
    sharpe_ratio = calculate_sharpe_ratio(returns_series)
    max_dd, max_dd_duration = calculate_max_drawdown(portfolio_values)
    beta = calculate_beta(returns_series, market_returns_series)
    alpha = calculate_alpha(returns_series, market_returns_series)
    
    # Print metrics
    print(f"\nPerformance Metrics:")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_dd:.2%}")
    print(f"Maximum Drawdown Duration: {max_dd_duration} days")
    print(f"Beta: {beta:.2f}")
    print(f"Alpha: {alpha:.2%}")
    
    # Plot the results using the helper function
    plot_backtest_results(portfolio_df, STRATEGY.__name__, SYMBOL)

if __name__ == '__main__':
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
