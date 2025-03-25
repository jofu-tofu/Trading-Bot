import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def time_series_performance(portfolio_weights, returns, plot = False, to_bps = 46):
    to = portfolio_weights.diff().abs()
    print('Mean Turnover: ', to.mean())  # Mean turnover in basis points
    returns = returns.loc[portfolio_weights.index]  # Align returns with portfolio weights
    port_ret = portfolio_weights.shift(1) * returns  - to.shift(1)*to_bps*(10**(-4))# Shift weights to avoid lookahead bias
    print('Time Exposed Ratio:', port_ret.count()/len(port_ret))
    print('Time Exposed Sharpe Ratio:', (port_ret.mean() / port_ret.std()) * np.sqrt(365*24))  # Annualized Sharpe Ratio
    downside_std = port_ret[port_ret < 0].std()  # Downside standard deviation
    print('Time Exposed Sortino Ratio:', (port_ret.mean() / downside_std) * np.sqrt(365*24))  # Annualized Sortino Ratio
    port_ret.fillna(0, inplace=True)  # Fill NaN values with 0 for performance calculation
    sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(365*24)  # Annualized Sharpe Ratio
    print('Total Sharpe Ratio:', sharpe)  # Annualized Sharpe Ratio
    sortino = (port_ret.mean() / downside_std) * np.sqrt(365*24)
    print('Total Sortino Ratio:', sortino)  # Annualized Sortino Ratio
    profit = port_ret.where(port_ret > 0, 0).cumsum()
    loss = -port_ret.where(port_ret < 0, 0).cumsum()
    pf = profit/loss
    print('Profit Factor: ', profit.iloc[-1]/loss.iloc[-1] if loss.iloc[-1] != 0 else np.inf)  # Profit factor
    cum_port_ret = port_ret.cumsum()
    cum_max = cum_port_ret.cummax()
    mask = cum_port_ret < cum_max
    drawdown = cum_max - cum_port_ret
    max_drawdown = drawdown.max()
    drawdown_duration = mask.groupby((~mask).cumsum()).cumcount() + 1
    drawdown_duration[~mask] = 0
    max_drawdown_duration = drawdown_duration.max()/24
    print('Maximum Drawdown:', max_drawdown)
    print('Maximum Drawdown Duration (days):', max_drawdown_duration)
    if plot:
        plt.figure(figsize=(12, 6))
        port_ret.cumsum().plot(label='Portfolio Return')
        plt.title('Cumulative Returns')
        plt.legend()
        plt.show()

        # Plot 2: Return distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(port_ret, bins=50, log_scale=(False, True), element='step').set(
            title='Portfolio Return Distribution',
            xlabel='Log Return', ylabel='Frequency')
        plt.show()
        pf_nonzero = pf[pf != 0]
        pf_trimmed = pf.loc[pf_nonzero.index[200:]]
        # Plot 3: Profit factor over time
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=pf_trimmed)
        plt.title('Profit Factor Over Time')
        plt.xlabel('Date')
        plt.ylabel('Profit Factor')
        plt.show()
    return to
    