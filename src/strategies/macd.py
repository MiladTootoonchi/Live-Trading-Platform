from ..alpaca_trader.order import SideSignal
import requests
from typing import Tuple
from config import load_api_keys, make_logger

logger = make_logger()

def exponential_moving_average(data, period):
    if not data or period <= 0:
        return []
    ema = []
    k = 2 / (period + 1)
    for i, price in enumerate(data):
        if i == 0:
            ema.append(price)
        else:
            ema.append(price * k + ema[i - 1] * (1 - k))
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

def fetch_price_data(symbol: str):
    """Fetches recent intraday price data from Alpaca (1Min bars)."""
    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys")
        return []

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?timeframe=1Min&limit=100"
    )

    logger.info(f"Fetching data from: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return []

    data = response.json()
    bars = data.get("bars", [])
    logger.info(f"Fetched {len(bars)} bars for {symbol}")
    return bars

def macd_strategy(position_data: dict) -> Tuple[SideSignal, int]:
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("Missing 'symbol' in position_data")
        return SideSignal.HOLD, 0

    #MACD strategy requires at least 35 historical bars to compute reliable EMA and signal line values.
    bars = fetch_price_data(symbol)
    if len(bars) < 35:
        logger.info(f"Not enough data to calculate MACD for {symbol}")
        return SideSignal.HOLD, 0

    closes = [float(bar["c"]) for bar in bars]
    macd_line, signal_line = calculate_macd(closes)

    if len(macd_line) < 2 or len(signal_line) < 2:
        return SideSignal.HOLD, 0

    prev_macd, curr_macd = macd_line[-2], macd_line[-1]
    prev_signal, curr_signal = signal_line[-2], signal_line[-1]

    if prev_macd <= prev_signal and curr_macd > curr_signal:
        logger.info(f"[{symbol}] BUY signal - bullish MACD crossover")
        return SideSignal.BUY, 0
    elif prev_macd >= prev_signal and curr_macd < curr_signal:
        logger.info(f"[{symbol}] SELL signal - bearish MACD crossover")
        return SideSignal.SELL, 0
    else:
        logger.info(f"[{symbol}] HOLD - no crossover signal")
        return SideSignal.HOLD, 0

