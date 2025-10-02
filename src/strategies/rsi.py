from ..alpaca_trader.order import SideSignal
import requests
from datetime import datetime, timezone, timedelta
from typing import Tuple
from config import load_api_keys, make_logger

logger = make_logger()

def calculate_rsi(closes, period: int = 14) -> float:
    gains = []
    losses = []

    for i in range(1, period + 1):
        change = closes[-i] - closes[-(i + 1)]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))

    average_gain = sum(gains) / period if gains else 0
    average_loss = sum(losses) / period if losses else 1e-10  # avoid div by zero

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_price_data(symbol: str, limit: int = 100):
    """Fetch latest 1-minute bars from Alpaca"""
    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys")
        return []

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Min&limit={limit}"
    logger.info(f"Fetching 1-minute bars from: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        bars = response.json().get("bars", [])
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        return bars
    except requests.RequestException as e:
        logger.error(f"Error fetching bars for {symbol}: {e}")
        return []

def rsi_strategy(position_data: dict) -> Tuple[SideSignal, int]:
    """
    RSI Strategy:
    - Buy if RSI < 30 (oversold)
    - Sell if RSI > 70 (overbought and position held)
    - Otherwise, hold
    Returns qty = 0 for main to decide quantity.
    """
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("Missing 'symbol' in position_data")
        return SideSignal.HOLD, 0

    bars = fetch_price_data(symbol)
    if len(bars) < 15:
        logger.info(f"Not enough bars to calculate RSI for {symbol}")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    rsi = calculate_rsi(closes[-15:])

    logger.info(f"[{symbol}] RSI: {rsi:.2f}")

    qty = int(float(position_data.get("qty", 0)))

    if rsi < 30:
        return SideSignal.BUY, 0 
    elif rsi > 70 and qty > 0:
        return SideSignal.SELL, 0  
    else:
        return SideSignal.HOLD, 0

