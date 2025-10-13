from ..alpaca_trader.order import SideSignal
from typing import Tuple
from config import load_api_keys, make_logger
import requests

logger = make_logger()

def fetch_price_data(symbol: str, limit: int = 100):
    """Fetches recent intraday price data from Alpaca (1Min bars)."""
    alpaca_key, alpaca_secret = load_api_keys()
    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys")
        return []

    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Min&limit={limit}"
    logger.info(f"Fetching data from: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return []

    data = response.json()
    bars = data.get("bars", [])
    if not bars:
        logger.warning(f"No bars returned for {symbol}. Check API keys and market hours.")
    return bars