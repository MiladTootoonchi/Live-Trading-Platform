from ..api_getter.order import SideSignal

import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple
from config import load_api_keys

def exponential_moving_average(data, period):
    if not data or period <= 0:
        return []
    ema = []
    k = 2 / (period + 1)
    for i, price in enumerate(data):
        if i == 0:
            ema.append(price)
        else:
            ema.append(price * k + ema[i - 1] * (1-k))
    return ema

def calculate_macd(closes):
    ema12 = exponential_moving_average(closes, 12)
    ema26 = exponential_moving_average(closes, 26)
    min_len = min(len(ema12), len(ema26))
    ema12 = ema12[-min_len:]
    ema26 = ema26[-min_len:]
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = exponential_moving_average(macd_line, 9)
    return macd_line, signal_line

def fetch_price_data(symbol: str, days: int = 100):
    """Fetches price data from Alpaca API"""
    alpaca_key, alpaca_secret = load_api_keys()
    
    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.isoformat()}Z"
        f"&end={end_date.isoformat()}Z"
        f"&timeframe=1Day&limit={days}"
    )

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("bars", [])
    except requests.RequestException as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return []

def macd_strategy(position_data: dict) -> Tuple[SideSignal, int]:
    """
    MACD-based trading strategy that can be used with AlpacaTrader.update()
    
    Args:
        position_data: Dict with position information from Alpaca
        
    Returns:
        Tuple[SideSignal, int]: Signal (BUY/SELL/HOLD) and number of shares
    """
    symbol = position_data.get("symbol")
    if not symbol:
        print("Missing 'symbol' in position_data")
        return SideSignal.HOLD, 0

    # Fetch price data
    bars = fetch_price_data(symbol, days=100)
    if len(bars) < 35:
        print(f"Not enough data to calculate MACD for {symbol}")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    macd_line, signal_line = calculate_macd(closes)

    if len(macd_line) < 2 or len(signal_line) < 2:
        return SideSignal.HOLD, 0

    prev_macd, curr_macd = macd_line[-2], macd_line[-1]
    prev_signal, curr_signal = signal_line[-2], signal_line[-1]

    print(f"[{symbol}] MACD: {curr_macd:.4f}, Signal: {curr_signal:.4f}")

    qty = int(float(position_data.get("qty", 0)))

    # Bullish crossover
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        print(f"[{symbol}] Bullish MACD crossover - BUY signal")
        return SideSignal.BUY, 1
    # Bearish crossover  
    elif prev_macd >= prev_signal and curr_macd < curr_signal and qty > 0:
        print(f"[{symbol}] Bearish MACD crossover - SELL signal")
        return SideSignal.SELL, qty
    else:
        return SideSignal.HOLD, 0
