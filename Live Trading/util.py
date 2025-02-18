# util.py
import requests
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