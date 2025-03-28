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

def compute_entropy(nbins, series):
    # Compute entropy of a series, normalized by the maximum entropy
    series = series.dropna()
    minimum = series.min()
    maximum = series.max()
    bins = np.linspace(minimum-0.0001, maximum+0.0001, nbins)
    digitized = np.digitize(series, bins)
    counts = np.bincount(digitized)[1:]
    probabilities = counts / len(series)
    entropy = -np.sum(probabilities * np.log(probabilities, where = probabilities > 0))
    return entropy/np.log(nbins)


import pandas as pd
import numpy as np

def create_dollar_bars(df, rolling_median_window=20, bars_per_day=50):
    df = df.sort_index()
    
    bars = []
    daily_totals = []
    
    carry_prices = np.array([], dtype=float)
    carry_volumes = np.array([], dtype=float)
    carry_dollar_vols = np.array([], dtype=float)
    carry_timestamps = np.array([], dtype='datetime64[ns]')
    
    for day, group in df.groupby(df.index.normalize()):
        group = group.sort_index()
        curr_prices = group['price'].values
        curr_volumes = group['volume'].values
        curr_dollar_vols = group['dollar_volume'].values
        curr_timestamps = group.index.to_numpy()
 
        day_total = curr_dollar_vols.sum()
        
        # Compute threshold using the rolling median of previous day totals
        if len(daily_totals) == 0:
            threshold = day_total / bars_per_day
        else:
            # Use the median over the last min(window, available days)
            window = daily_totals[-rolling_median_window:]
            median_total = np.median(window)
            threshold = median_total / bars_per_day

        # Adjust carryover timestamps to the start of the current day if any exist.
        if carry_timestamps.size > 0:
            carry_timestamps = np.full(carry_timestamps.shape, curr_timestamps[0])

        combined_prices = np.concatenate([carry_prices, curr_prices])
        combined_volumes = np.concatenate([carry_volumes, curr_volumes])
        combined_dollar_vols = np.concatenate([carry_dollar_vols, curr_dollar_vols])
        combined_timestamps = np.concatenate([carry_timestamps, curr_timestamps])

        cs = combined_dollar_vols.cumsum()
        start = 0 
        
        while start < len(cs):
            baseline = cs[start - 1] if start > 0 else 0
            target = baseline + threshold
            j = np.searchsorted(cs, target, side='left')
            if j >= len(cs):
                break
            bar_prices = combined_prices[start:j+1]
            bar_volumes = combined_volumes[start:j+1]
            bar_dollar_vols = combined_dollar_vols[start:j+1]
            
            open_price = bar_prices[0]
            high_price = bar_prices.max()
            low_price = bar_prices.min()
            close_price = bar_prices[-1]
            vwap = (bar_prices * bar_volumes).sum() / bar_volumes.sum()
            bar_volume = bar_volumes.sum()
            bar_dollar_sum = bar_dollar_vols.sum()
            bar_end_time = combined_timestamps[j]
            
            bars.append({
                'datetime': bar_end_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'vwap': vwap,
                'volume': bar_volume,
                'dollar_volume': bar_dollar_sum
            })
            start = j + 1  
        
        if start < len(cs):
            carry_prices = combined_prices[start:]
            carry_volumes = combined_volumes[start:]
            carry_dollar_vols = combined_dollar_vols[start:]
            carry_timestamps = combined_timestamps[start:]
        else:
            carry_prices = np.array([], dtype=float)
            carry_volumes = np.array([], dtype=float)
            carry_dollar_vols = np.array([], dtype=float)
            carry_timestamps = np.array([], dtype='datetime64[ns]')
        
        # Append the current day's total to the list for future rolling median calculation.
        daily_totals.append(day_total)
    
    if carry_prices.size > 0:
        open_price = carry_prices[0]
        high_price = carry_prices.max()
        low_price = carry_prices.min()
        close_price = carry_prices[-1]
        vwap = (carry_prices * carry_volumes).sum() / carry_volumes.sum()
        bar_volume = carry_volumes.sum()
        bar_dollar_sum = carry_dollar_vols.sum()
        bar_end_time = carry_timestamps[-1]
        bars.append({
            'datetime': bar_end_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'vwap': vwap,
            'volume': bar_volume,
            'dollar_volume': bar_dollar_sum
        })
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('datetime', inplace=True)
    return bars_df

def get_tick_data(symbol):
    data_path = '../../Data/Kraken_Trading_History/'
    coin_name = symbol.upper() + 'USD'
    file_path = data_path + coin_name + '.csv'
    df = pd.read_csv(file_path, header=None)
    df.set_index(0, inplace=True)
    df.columns = ['price', 'volume']
    df['dollar_volume'] = df['price'] * df['volume']
    df.index = pd.to_datetime(df.index, unit='s')
    return df