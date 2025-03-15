import sqlite3
import pandas as pd
import pickle
import requests
from dotenv import load_dotenv
import os
import numpy as np
load_dotenv()
data_path = '../Data/'
COINGECKO_API = os.getenv("COINGECKO_API_KEY")
KRAKEN_API = os.getenv('KRAKEN_API_KEY')
HEADERS = {"x-cg-pro-api-key": COINGECKO_API,
           "accept": "application/json"}
KRAKEN_HEADER = {
  'Accept': 'application/json'
}
def get_tradable_kraken_coins(volume = 100000):
    filepath = data_path + 'tradable_kraken_coins.pkl'
    tradable_coins = []
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            tradable_coins = pickle.load(f)
        return tradable_coins
    url = "https://api.kraken.com/0/public/AssetPairs"
    all_coins = get_exchange_coins(exchange = 'kraken', volume_threshold = volume)
    for coin in all_coins:
        payload={}
        if coin['target'] == 'USD':
            params = {
                "pair": coin['base'] + '/USD',
                "country_code": "US",
            }
            tradable_coins.append(coin['base'])
            try:
                resp = requests.request("GET", url, headers=KRAKEN_HEADER, params = params, data=payload)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"Error fetching tradable coins: {e}")
            if payload:
                tradable_coins.append(coin['base'])
    with open(filepath, 'wb') as f:
        pickle.dump(tradable_coins, f)
    return tradable_coins
            
    
def get_all_coin_names():
    filepath = data_path + 'all_coin_names.pkl'
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            all_coins = pickle.load(f)
        return all_coins
    import requests

    url = "https://pro-api.coingecko.com/api/v3/coins/list"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return data
    

def get_exchange_coins(exchange = 'kraken', volume_threshold = 100000):
    filepath = data_path + f'{exchange}_coins_{volume_threshold}.pkl'
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            all_coins = pickle.load(f)
        return all_coins
    url = f"https://pro-api.coingecko.com/api/v3/exchanges/{exchange}/tickers"
    params = {
        "depth": "true",
        "order": "volume_desc",
        "page": 1,
    }
    
    all_coins = []
    try:
        while(True):
            resp = requests.get(url, params=params, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()['tickers']
            if not data:
                break
            for coin in data:
                if coin['converted_volume']['usd'] < volume_threshold:
                    break
                else:
                    if coin['target'] == 'USD':
                        all_coins.append(coin)
            params["page"] += 1
        with open(filepath, 'wb') as f:
            pickle.dump(all_coins, f)
    except requests.RequestException as e:
        print(f"Error fetching exchange coins: {e}")
    return all_coins
def rolling_mad_df(df, window):
    arr = df.to_numpy()
    T, N = arr.shape
    arr_T = arr.T
    windows = np.lib.stride_tricks.sliding_window_view(arr_T, window_shape=window, axis=1)
    medians = np.median(windows, axis=2)
    abs_dev = np.abs(windows - medians[:, :, None])
    mad = np.median(abs_dev, axis=2)
    mad = mad.T
    mad_full = np.full((T, N), np.nan)
    mad_full[window - 1:, :] = mad
    mad_df = pd.DataFrame(mad_full, index=df.index, columns=df.columns)
    return mad_df

def modify_extreme_ret(ret):
    # Modify extreme returns
    ret = np.log1p(ret)
    rolling_window = 12
    rollingmedian = ret.rolling(rolling_window, min_periods=1).median()
    rollingmad = rolling_mad_df(ret, rolling_window)
    outlier = abs(ret-rollingmedian) > 15*rollingmad
    ret=ret.where(~outlier, rollingmedian)
    return np.expm1(ret)

def get_all_ochl():
    file_path=data_path + 'all_ochl.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        return df
    conn = sqlite3.connect(data_path + "crypto_data.db")
    query = "SELECT time_rank, coin_id, symbol, name, open, high, low, close, volume, market_cap FROM coin_history"
    df = pd.read_sql_query(query, conn)
    df['time_rank'] = pd.to_datetime(df['time_rank'])-pd.Timedelta(hours=1)
    conn.close()
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
    return df


def transform_returns(df = None, volume_threshold = 10000):
    if df is None:
        df = get_all_ochl()
    df = df.rename(columns={'time_rank':'date'})
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    df = df.sort_values(by=['symbol', 'date'])
    
    df['return'] = df.groupby('symbol')['close'].pct_change()
    
    df.loc[df['volume'] < volume_threshold, 'return'] = np.nan
    
    returns_df = df.pivot_table(index='date', columns='symbol', values='return', aggfunc='last')
    min_date = returns_df.index.min()
    max_date = returns_df.index.max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='h')
    returns_df = returns_df.reindex(full_date_range, fill_value=0)
    return returns_df

def get_all_returns(volume_threshold = 10000):
    file_path=data_path + f'all_returns_{volume_threshold}.pkl'
    try:
        with open(file_path, 'rb') as f:
            all_returns = pickle.load(f)
        return all_returns
    except FileNotFoundError:
        print("File not found. Fetching from database.")
        ret = transform_returns(volume_threshold = volume_threshold)
        with open(file_path, 'wb') as f:
            pickle.dump(ret, f)
        return ret

def get_col_pivot_from_all_ochl(type = 'close', volume_threshold = 10000):
    # transform ochl to pivot table of one column, e.g. close, high, low, open
    # Ensure that the time column is in datetime format and set it as the index
    file_path=data_path + f'all_{type}_{volume_threshold}.pkl'
    try:
        with open(file_path, 'rb') as f:
            all_prices = pickle.load(f)
        return all_prices
    except FileNotFoundError:
        print("File not found. Fetching from database.")
        all_ochl = get_all_ochl()
        all_prices = transform_col(all_ochl, type = type, volume_threshold = volume_threshold)
        min_date = all_prices.index.min()
        max_date = all_prices.index.max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='h')
        all_prices = all_prices.reindex(full_date_range)
        with open(file_path, 'wb') as f:
            pickle.dump(all_prices, f)
        return all_prices
    
def transform_col(df = None, type = 'close', volume_threshold = 10000):
    if df is None:
        df = get_all_ochl()
    df = df.rename(columns={'time_rank':'date'})
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    df = df.sort_values(by=['symbol', 'date'])
    df.loc[df['volume'] < volume_threshold, type] = np.nan
    
    # Use pivot_table to aggregate duplicate date-symbol pairs (using the last available value)
    prices_df = df.pivot_table(index='date', columns='symbol', values=type, aggfunc='last')
    return prices_df
      
      
def get_coins_from_category(category = 'layer-1',vs_currency="usd"):
    file_path = data_path + f'{category}_coins.pkl'
    try:
        with open(file_path, 'rb') as f:
           coins = pickle.load(f)
        return coins
    except FileNotFoundError:
        print("File not found. Fetching from database.")
        all_coins = fetch_coins_from_category(category,vs_currency)
        with open(file_path, 'wb') as f:
            pickle.dump(all_coins, f)
        return all_coins
      
def fetch_coins_from_category(category = 'layer-1',vs_currency="usd"):
    """
    Fetch the list of Layer-1 coins from CoinGecko using the "layer-1" category.
    """
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "category": category,  # Category provided by CoinGecko for Layer-1 coins
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
        "locale": "en"
    }
    all_coins = []
    try:
        while True:
            resp = requests.get(url, params=params, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break  # No more pages
            for coin in data:
                # Save CoinGecko coin id, name and symbol for future use.
                coin_id = coin.get("id")
                name = coin.get("name")
                symbol = coin.get("symbol")
                all_coins.append({"id": coin_id, "name": name, "symbol": symbol})
            params["page"] += 1  # Next page
    except requests.RequestException as e:
        print(f"Error fetching {category} coins: {e}")
    return all_coins

def purify_ret(portfolio_ret, btc_returns, window = 48):
    port_series = pd.Series(portfolio_ret)
    btc_series = pd.Series(btc_returns)
    
    # Compute rolling covariance and variance using a window of the past observations
    rolling_cov = port_series.rolling(window=window).cov(btc_series)
    rolling_var = btc_series.rolling(window=window).var()
    
    # Compute rolling beta; shift beta by one period to ensure no lookahead bias.
    beta = (rolling_cov / rolling_var).shift(1)
    
    # Compute decorrelated returns using the beta from prior data
    decorrelated_returns = port_series - beta * btc_series
    
    return decorrelated_returns, beta

def get_positive_columns(df):
    result_df = df.apply(lambda row: row.index[row > 0].tolist(), axis=1)
    result_df = result_df.to_frame(name="positive_columns")  # Convert Series to DataFrame
    return result_df

def to_sharpe(weightings, ret, th = 1, to_off = False, plot = False, return_ret = False, purify = False):
        weightings = weightings.iloc[::th].fillna(0)
        # ret_prod = (1+ret).cumprod()
        # ret = ret_prod.divide(ret_prod.shift(th), axis = 0) - 1
        ret = ret.rolling(th).sum()
        ret = ret[::th].fillna(0)
        to = weightings.diff().abs().sum(1)
        matching_columns = weightings.columns.intersection(ret.columns)
        weightings_ret = weightings[matching_columns].shift(1) * ret[matching_columns]
        port_ret = weightings_ret.sum(1)
        num_pos = port_ret[port_ret > 0].count()
        prop_pos = num_pos / len(port_ret)
        bps = .0046
        if to_off:
            bps = 0
        port_ret = port_ret - bps * to
        if purify and 'btc' in ret.columns:
            if plot:
                print("Corr with BTC (before purification): ", port_ret.corr(ret['btc']))
            port_ret, beta = purify_ret(port_ret, ret['btc'])
            if plot:
                print("BTC Beta: ", beta.mean())
        if return_ret:
            return port_ret
        avg_to = to.mean()
        sharpe = np.sqrt(24*365/th) * port_ret.mean() / port_ret.std()
        cum_port_ret = port_ret.cumsum()
        cum_max = cum_port_ret.cummax()
        mask = cum_port_ret < cum_max
        drawdown = cum_max - cum_port_ret
        max_drawdown = drawdown.max()
        drawdown_duration = mask.groupby((~mask).cumsum()).cumcount() + 1
        drawdown_duration[~mask] = 0
        max_drawdown_duration = drawdown_duration.max()/24
        if plot:
            if 'btc' in ret.columns:
                print("Corr with BTC: ", port_ret.corr(ret['btc']))
            print("Average Turnover: ", avg_to)
            print("Sharpe Ratio: ", sharpe)
            print('Max Drawdown: ', max_drawdown)
            print("Time Exposure Ratio: ", prop_pos)
            print('Max Drawdown Duration: ', max_drawdown_duration, 'days')
            port_ret.cumsum().plot()
        return avg_to, sharpe, max_drawdown, max_drawdown_duration