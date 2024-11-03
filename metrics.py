import numpy as np

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

def calculate_win_rate(returns):
    """
    Calculate the win rate (percentage of profitable trades)
    Args:
        returns: pandas Series of returns
    Returns:
        win_rate: Percentage of winning trades
    """
    wins = (returns > 0).sum()
    total = len(returns)
    return wins / total if total > 0 else 0.0

def calculate_profit_factor(returns):
    """
    Calculate the profit factor (gross profits / gross losses)
    Args:
        returns: pandas Series of returns
    Returns:
        profit_factor: Ratio of gross profits to gross losses
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses != 0 else float('inf')
