from ..api_getter.order import SideSignal

import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple
from config import load_api_keys

def calculate_rsi(closes, period: int = 14) -> float:
    gains = []
    losses = []

    for i in range(1, period + 1):
        change = closes[-(i)] - closes[-(i + 1)]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change)) 

    average_gain = sum(gains) / period if gains else 0
    average_loss = sum(losses) / period if losses else 1e-10  # Avoid division by zero

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(position_data: dict) -> Tuple[SideSignal, int]:
    """
    RSI Strategy:
    - Buy if RSI < 30 (oversold)
    - Sell if RSI > 70 (overbought and position held)
    - Otherwise, hold
    """
    symbol = position_data.get("symbol")
    if not symbol:
        print("Missing 'symbol' in position_data")
        return SideSignal.HOLD, 0

    alpaca_key, alpaca_secret = load_api_keys()

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=100)

    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.isoformat()}Z"
        f"&end={end_date.isoformat()}Z"
        f"&timeframe=1Day&limit=100"
    )

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return SideSignal.HOLD, 0

    bars = response.json().get("bars", [])
    if len(bars) < 15:
        print(f"Not enough data to calculate RSI for {symbol}")
        return SideSignal.HOLD, 0

    closes = [bar["c"] for bar in bars]
    rsi = calculate_rsi(closes[-15:])  #15 closes to calculate 14-period RSI

    print(f"[{symbol}] RSI: {rsi:.2f}")

    qty = int(float(position_data.get("qty", 0)))

    if rsi < 30:
        return SideSignal.BUY, 1
    elif rsi > 70 and qty > 0:
        return SideSignal.SELL, qty
    else:
        return SideSignal.HOLD, 0
