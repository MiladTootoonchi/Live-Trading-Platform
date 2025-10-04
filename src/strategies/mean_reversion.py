from ..alpaca_trader.order import SideSignal
from config import load_api_keys, make_logger
import requests

logger = make_logger()

def fetch_price_data(symbol: str):
    """Fetches latest 1-minute bars from Alpaca"""
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
        data = response.json()
        bars = data.get("bars", [])
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        return bars
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return []

def mean_reversion_strategy(position_data: dict, moving_avg_price: float = 0, min_data_points: int = 1) -> tuple[SideSignal, int]:
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("[Mean Reversion] No symbol provided")
        return SideSignal.HOLD, 0

    bars = fetch_price_data(symbol)
    if bars and len(bars) > 0:
        current_price = float(bars[-1]["c"])
    else:
        current_price = float(position_data.get("current_price", 0))

    if moving_avg_price == 0:
        logger.info("[Mean Reversion] Not enough data to calculate moving average")
        return SideSignal.HOLD, 0

    deviation = (current_price - moving_avg_price) / moving_avg_price * 100

    logger.info(f"[Mean Reversion] Current: {current_price:.2f}, SMA: {moving_avg_price:.2f}, Deviation: {deviation:.2f}%")

    if deviation < -6:
        logger.info(f"[Mean Reversion] SELL signal - stop-loss triggered at {deviation:.2f}%")
        return SideSignal.SELL, 0
    elif deviation < -3:
        logger.info(f"[Mean Reversion] BUY signal - price {deviation:.2f}% below SMA")
        return SideSignal.BUY, 0
    elif deviation > 4:
        logger.info(f"[Mean Reversion] SELL signal - price {deviation:.2f}% above SMA")
        return SideSignal.SELL, 0
    else:
        logger.info("[Mean Reversion] HOLD - price within acceptable range")
        return SideSignal.HOLD, 0


      

   

