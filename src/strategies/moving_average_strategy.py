from ..alpaca_trader.order import SideSignal
import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple
import os
from dotenv import load_dotenv
from config import make_logger

logger = make_logger()

load_dotenv()

def moving_average_strategy(position: dict) -> Tuple[SideSignal, int]:
    """
    Moving Average Crossover Strategy using Alpaca Historical data.
    Logic:
    - Buy if current price > MA20 > MA50 > MA200
    - Sell if current price < MA20 < MA50 < MA200 and we have qty
    - Hold otherwise
    """
    symbol = position.get("symbol")
    if not symbol:
        logger.error("No symbol provided in position")
        return SideSignal.HOLD, 0
    
    alpaca_key = os.getenv("alpaca_key")
    alpaca_secret = os.getenv("alpaca_secret_key")

    if not alpaca_key or not alpaca_secret:
        logger.error("Missing API keys")
        return SideSignal.HOLD, 0
    
    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=300)
    
    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?start={start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&end={end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&timeframe=1Day&limit=300"
    )
    
    logger.info(f"Fetching data from: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 401:
            logger.error(f"Unauthorized (401) - Check API keys for {symbol}")
            return SideSignal.HOLD, 0
        elif response.status_code == 403:
            logger.error(f"Forbidden (403) - Check API permissions for {symbol}")
            return SideSignal.HOLD, 0
        elif response.status_code == 422:
            logger.error(f"Unprocessable Entity (422) - Invalid parameters for {symbol}")
            logger.error(f"Response: {response.text}")
            return SideSignal.HOLD, 0
        elif response.status_code == 429:
            logger.error(f"Rate limited (429) for {symbol}")
            return SideSignal.HOLD, 0
        
        response.raise_for_status()
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return SideSignal.HOLD, 0
    
    try:
        data = response.json()
        bars = data.get("bars", [])
        
        logger.info(f"Response keys: {list(data.keys())}")
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        
        if not bars:
            logger.warning(f"No bars returned for {symbol}")
            logger.info(f"Full response: {data}")
            return SideSignal.HOLD, 0
        
        if len(bars) < 200:
            logger.info(f"Not enough data for {symbol}. Requires at least 200 days of data")
            return SideSignal.HOLD, 0
        
        closes = [float(bar["c"]) for bar in bars]
        current_price = closes[-1]
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        ma200 = sum(closes[-200:]) / 200
        
        logger.info(f"[{symbol}] Price: {current_price:.2f}, MA20: {ma20:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}")
        
        qty = int(float(position.get("qty", 0)))
        
        # Bullish alignment: Price > MA20 > MA50 > MA200
        if current_price > ma20 > ma50 > ma200:
            logger.info(f"[{symbol}] BUY signal - bullish MA alignment")
            return SideSignal.BUY, 1
        # Bearish alignment: Price < MA20 < MA50 < MA200
        elif current_price < ma20 < ma50 < ma200 and qty > 0:
            logger.info(f"[{symbol}] SELL signal - bearish MA alignment")
            return SideSignal.SELL, qty
        else:
            logger.info(f"[{symbol}] HOLD - no clear MA trend")
            return SideSignal.HOLD, 0
            
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error processing MA data for {symbol}: {e}")
        return SideSignal.HOLD, 0