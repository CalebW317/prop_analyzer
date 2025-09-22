# discover_markets.py
import os
import json
import requests

# --- CONFIG ---
# Make sure your ODDS_API_KEY is set as an environment variable
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
SPORT = 'americanfootball_nfl'
API_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
REGIONS = 'us'

if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY environment variable is missing!")

# --- SCRIPT ---
print(f"Discovering available markets for sport '{SPORT}' in region '{REGIONS}'...")

# We intentionally leave out the 'markets' parameter to see everything available.
params = {
    'apiKey': ODDS_API_KEY,
    'regions': REGIONS,
}

try:
    response = requests.get(API_URL, params=params, timeout=15)

    if response.status_code != 200:
        print(f"\nError: API returned status {response.status_code}")
        print(response.text)
    else:
        games_data = response.json()
        if not games_data:
            print("\nNo active games found for this sport. Cannot determine available markets.")
        else:
            # Use a set to store unique market keys
            available_markets = set()
            for game in games_data:
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        available_markets.add(market.get('key'))
            
            print("\n--- Found Available Markets ---")
            if not available_markets:
                print("No markets found in any of the active games.")
            else:
                for market_key in sorted(list(available_markets)):
                    print(f"- {market_key}")
            print("-----------------------------")
            print("\nCopy the exact market keys you want from the list above into your main script.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
