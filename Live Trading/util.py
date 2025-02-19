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