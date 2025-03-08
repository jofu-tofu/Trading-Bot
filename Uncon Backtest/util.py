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
HEADERS = {"x-cg-pro-api-key": COINGECKO_API,
           "accept": "application/json"}
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
    # Ensure that the time column is in datetime format and set it as the index
    if df is None:
        df = get_all_ochl()
    df = df.rename(columns={'time_rank':'date'})
    # Ensure DataFrame is sorted by symbol and date.
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    df = df.sort_values(by=['symbol', 'date'])
    
    # Compute returns using the 'close' price.
    df['return'] = df.groupby('symbol')['close'].pct_change()
    
    # Mask returns where volume is less than threshold.
    df.loc[df['volume'] < volume_threshold, 'return'] = np.nan
    
    # Use pivot_table to aggregate duplicate date-symbol pairs (using the last available value)
    returns_df = df.pivot_table(index='date', columns='symbol', values='return', aggfunc='last')
    return returns_df

def get_all_returns(volume_threshold = 10000):
    file_path=data_path + f'all_returns_{volume_threshold}.pkl'
    try:
        with open(file_path, 'rb') as f:
            layer1_returns = pickle.load(f)
        return layer1_returns
    except FileNotFoundError:
        print("File not found. Fetching from database.")
        ret = transform_returns(volume_threshold = volume_threshold)
        with open(file_path, 'wb') as f:
            pickle.dump(ret, f)
        return ret

def get_all_prices(type = 'close'):
    file_path=data_path + f'all_prices_{type}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            layer1_prices = pickle.load(f)
        return layer1_prices
    conn = sqlite3.connect(data_path + "crypto_data.db")
    query = "SELECT time_rank, coin_id, symbol, name, open, high, low, close, volume, market_cap FROM coin_history"
    df = pd.read_sql_query(query, conn)
    df['time_rank'] = pd.to_datetime(df['time_rank'])-pd.Timedelta(hours=1)
    df = df.rename(columns={'time_rank':'date'})
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    df = df.sort_values(by=['symbol', 'date'])
    prices_df = df.pivot_table(index='date', columns='symbol', values=type)
    with open(file_path, 'wb') as f:
        pickle.dump(prices_df, f)
    return prices_df
    

      
      
def get_coins_from_category(category = 'layer-1',vs_currency="usd"):
    file_path = data_path + f'{category}_coins.pkl'
    try:
        with open(file_path, 'rb') as f:
            layer1_coins = pickle.load(f)
        return layer1_coins
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
        if purify:
            port_ret, beta = purify_ret(port_ret, ret['btc'])
        if return_ret:
            return port_ret,to, ret, weightings
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