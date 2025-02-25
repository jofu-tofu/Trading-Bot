# util.py
import requests
import pandas as pd
from datetime import datetime
import numpy as np
def get_layer1_universe():
        """
        Retrieve the list of Layer 1 assets from the TWSQ API.

        Returns:
        --------
        list
            A list of Layer 1 asset symbols.
        """
        api_url = f'https://api.coingecko.com/api/v3/coins/markets'
        category = 'smart-contract-platform'  # Adjust based on actual category
        per_page = 250
        page = 1
        vs_currency = 'usd'
        order = 'market_cap_desc'
        sparkline = 'false'

        coin_names = []
        coin_ticker = []

        while True:
            params = {
                'vs_currency': vs_currency,
                'category': category,
                'order': order,
                'per_page': per_page,
                'page': page,
                'sparkline': sparkline
            }

            response = requests.get(api_url, params=params)

            if response.status_code != 200:
                print(f"Error: Unable to fetch data (Status Code: {response.status_code})")
                break

            data = response.json()

            if not data:
                # No more data to fetch
                break
            
            # Extract coin names
            for coin in data:
                coin_names.append(coin['name'])
                coin_ticker.append(coin['symbol'])

            print(f"Fetched page {page} with {len(data)} coins.")

            page += 1
        return coin_ticker
def corr_with_btc(port_ret):
    bitcoin_data_path = '../bitcoin_data.pkl'
    # Load the Bitcoin data
    bitcoin_data = pd.read_pickle(bitcoin_data_path)



    bitcoin_data['open_time'] = pd.to_datetime(bitcoin_data['open_time'])

    if bitcoin_data['open_time'].dt.tz is not None:
        bitcoin_data['open_time'] = bitcoin_data['open_time'].dt.tz_localize(None)

    bitcoin_data.set_index('open_time', inplace=True)
    if isinstance(port_ret.index, pd.DatetimeIndex):
        port_ret.index = port_ret.index.tz_localize(None)

    # Now perform correlation safely
    btc_returns = pd.to_numeric(bitcoin_data['close'], errors='coerce').pct_change().dropna()
    port_ret_clean = port_ret.dropna()

    # Align indices and compute correlation
    aligned_data = port_ret_clean.align(btc_returns, join='inner')
    corr = aligned_data[0].corr(aligned_data[1])
    return corr

def to_sharpe(weightings, ret, th = 1, to_off = False, plot = True, return_ret = False):
        weightings = weightings.iloc[::th].fillna(0)
        # ret_prod = (1+ret).cumprod()
        # ret = ret_prod.divide(ret_prod.shift(th), axis = 0) - 1
        ret = ret.rolling(th).sum()
        ret = ret[::th].fillna(0)
        to = weightings.diff().abs().sum(1)
        matching_columns = weightings.columns.intersection(ret.columns)
        weightings_ret = weightings[matching_columns].shift(1) * ret[matching_columns]
        port_ret = weightings_ret.sum(1)
        bps = .0026
        if to_off:
            bps = 0
        port_ret = port_ret - bps * to
        if return_ret:
            return port_ret,to, ret, weightings
        avg_to = to.mean()
        sharpe = np.sqrt(24*365/th) * port_ret.mean() / port_ret.std()
        drawdown = port_ret.cumsum().cummax() - port_ret.cumsum()
        max_drawdown = drawdown.max()
        drawdown_durations = (drawdown > 0).cumsum()
        durations = drawdown_durations.groupby(drawdown_durations).cumcount()
        max_drawdown_duration = durations.max()*th/24
        if plot:
            if 'BTCUSDT' in ret.columns:
                print("Corr with BTC: ", port_ret.corr(ret['BTCUSDT']))
            print("Average Turnover: ", avg_to)
            print("Sharpe Ratio: ", sharpe)
            port_ret.cumsum().plot()
        return avg_to, sharpe, max_drawdown, max_drawdown_duration