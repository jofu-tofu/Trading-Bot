import sqlite3
import pandas as pd
import pickle
import requests
from dotenv import load_dotenv
import os
load_dotenv()
COINGECKO_API = os.getenv("COINGECKO_API_KEY")
HEADERS = {"x-cg-pro-api-key": COINGECKO_API,
           "accept": "application/json"}

def get_layer1_df():
    conn = sqlite3.connect("../Data/crypto_data.db")
    query = "SELECT time_rank, coin_id, symbol, name, open, high, low, close, volume, market_cap FROM coin_history"
    df = pd.read_sql_query(query, conn)
    df['time_rank'] = pd.to_datetime(df['time_rank'])
    conn.close()
    return df
def get_layer1_coins():
    try:
        with open('../Data/layer1_coins.pkl', 'rb') as f:
            layer1_coins = pickle.load(f)
        return layer1_coins
    except FileNotFoundError:
        print("File not found. Fetching from database.")
        return fetch_layer1_coins()
        
def fetch_layer1_coins(vs_currency="usd"):
    """
    Fetch the list of Layer-1 coins from CoinGecko using the "layer-1" category.
    """
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "category": "layer-1",  # Category provided by CoinGecko for Layer-1 coins
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
        print(f"Error fetching Layer-1 coins: {e}")
    return all_coins