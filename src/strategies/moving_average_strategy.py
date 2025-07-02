import requests
import os
import toml
from datetime import datetime, timedelta
from typing import Tuple
from ..api_getter.order import SideSignal

def load_api_keys(config_file: str = "settings.toml") -> tuple:
    try:
        alpaca_key = os.getenv("ALPACA_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    except:
        pass

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            keys = conf.get("keys", {})
            alpaca_key = keys.get("alpaca_key", alpaca_key)
            alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)
    
    except:
        pass

    return alpaca_key, alpaca_secret

# defining the strategy

def moving_average_strategy(position: dict) -> Tuple[SideSignal, int]:
    symbol = position["symbol"]
    alpaca_key, alpaca_secret = load_api_keys()

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

# Fetch 200 days of historical prices (1-day intervals)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=250)
    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.isoformat()}Z"
        f"&end={end_date.isoformat()}Z"
        f"&timeframe=1Day&limit=250"
    )

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch data for {symbol}: {response.text}")
        return SideSignal.HOLD, 0

    bars = response.json().get("bars", [])
    if len(bars) < 200:
        print(f"Not enough data for {symbol}")
        return SideSignal.HOLD, 0

    closes = [bar["c"] for bar in bars]

    current_price = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200

    print(f"[{symbol}] Price: {current_price:.2f}, MA20: {ma20:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")

    if current_price > ma20 > ma50 > ma200:
        return SideSignal.BUY, 1
    elif current_price < ma20 < ma50 < ma200:
        return SideSignal.SELL, int(float(position["qty"]))
    else: 
        return SideSignal.HOLD, 0