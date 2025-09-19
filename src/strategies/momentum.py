from src.alpaca_trader.order import SideSignal
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

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Min&limit=100"
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

def momentum_strategy(position_data: dict) -> tuple[SideSignal, int]:
    symbol = position_data.get("symbol")
    if not symbol:
        logger.error("[Momentum] No symbol provided")
        return SideSignal.HOLD, 0

    bars = fetch_price_data(symbol)
    if not bars:
        return SideSignal.HOLD, 0

    current_price = float(bars[-1]["c"])
    open_price_today = float(bars[-1]["o"])
    change_today = (current_price - open_price_today) / open_price_today * 100

    qty = int(float(position_data.get("qty", 0)))
    avg_entry_price = float(position_data.get("avg_entry_price") or current_price)
    unrealized_return_pct = (current_price - avg_entry_price) / avg_entry_price * 100

    logger.info(f"[Momentum] {symbol} Price: {current_price:.2f}, Change today: {change_today:.2f}%")

    # Buy: strong positive momentum
    if qty == 0 and change_today > 2:
        logger.info("[Momentum] BUY signal triggered")
        return SideSignal.BUY, 10

    # Sell: take profit, negative momentum, or stop loss
    if unrealized_return_pct > 5:
        logger.info("[Momentum] SELL signal - take profit")
        return SideSignal.SELL, qty

    if change_today < -1 and unrealized_return_pct < 2:
        logger.info("[Momentum] SELL signal - negative momentum with small gain")
        return SideSignal.SELL, qty

    if unrealized_return_pct < -3:
        logger.info("[Momentum] SELL signal - stop loss")
        return SideSignal.SELL, qty

    return SideSignal.HOLD, 0
