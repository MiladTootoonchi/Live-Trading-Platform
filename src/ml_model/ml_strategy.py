import asyncio
import pandas as pd
import numpy as np
import datetime as dt
from alpaca.trading.client import TradingClient

from ..alpaca_trader.order import SideSignal
from config import make_logger, load_api_keys
from src.ml_model.data import fetch_data, stock_data_prediction_pipeline, stock_data_feature_engineering, get_one_realtime_bar
from src.ml_model.training import train_model, sequence_split
from src.ml_model.evaluations import evaluate_model

logger = make_logger()

async def AI_strategy(position_data: dict) -> tuple[SideSignal, int]:
    """
    Evaluates a trading position from an Alpaca JSON response and recommends an action.
    This strategy will only buy or sell.

    Args:
        position_data (dict): JSON object from Alpaca API containing position details.

    Returns:
        tuple:
            (SideSignal.BUY or SideSignal.SELL, qty: int)
    """

    symbol = position_data["symbol"]

    yesterday = dt.datetime.now(dt.UTC) - dt.timedelta(days = 1)
    mdip = dt.datetime.now(dt.UTC) - dt.timedelta(days = 100)        # more than 50 days in the past (many days in past)

    # fetching data from 2020 till mdip (mdip till today will be used for prediction)
    df = fetch_data(symbol = symbol, 
                    start_date = (2020, 1, 1), 
                    end_date = (mdip.year, mdip.month, mdip.day))
    
    # Feature engineering for training and test data
    X, y, scaler = stock_data_feature_engineering(df)

    X_train, X_test, y_train, y_test = sequence_split(X, y) # splitting

    model = train_model(X_train, y_train)   # training

    # Evaluation
    f1, accuracy = evaluate_model(model, X_test, y_test)

    # Fetching mdip data till today (realtime data)
    realtime_bar = await get_one_realtime_bar(symbol = symbol)

    hist_df = fetch_data("MSFT", 
                        start_date = (mdip.year, mdip.month, mdip.day), 
                        end_date = (yesterday.year, yesterday.month, yesterday.day))
    
    hist_df = hist_df.reset_index()     # making sure there isnt any dobble index

    pred_df = pd.concat([hist_df.tail(50), realtime_bar])
    
    # Preprocessing pred_data
    X = stock_data_prediction_pipeline(pred_df, scaler)

    # predicting
    X_seq = np.expand_dims(X[-50:], axis=0)
    real_time_prediction = model.predict(X_seq)

    signal = int((real_time_prediction > 0.5).astype(int).item())

    info = f"""
    The stock will go (1 for up, 0 for down) = {signal} tommorow 
    with {(real_time_prediction*100):.4f} % probability.
    The AI - model got a test f1-score on {f1}, and a test accuracy score on {accuracy}.
    """

    logger.info(info)


    # Deciding side and qty
    side = SideSignal.BUY if real_time_prediction > 0.5 else SideSignal.SELL
    qty = compute_trade_qty(position_data, float(real_time_prediction))

    return side, qty





def compute_trade_qty(position_data: dict, prob: float) -> int:
    """
    Calculates an intelligent stock quantity to trade based on model confidence and risk management.

    The function automatically retrieves your Alpaca account equity, then uses a hybrid 
    risk model that combines confidence scaling and fixed risk-per-trade rules. This ensures 
    trades are dynamically sized while respecting account-level risk limits.

    Args:
        position_data (dict): Alpaca position data containing 'symbol' and price info.
        prob (float): Model probability (0.0 - 1.0) that the trade prediction is correct.

    Returns:
        int: Recommended quantity of shares to buy or sell.
    """

    # Load Alpaca API keys
    alpaca_key, alpaca_secret = load_api_keys()

    try:
        client = TradingClient(alpaca_key, alpaca_secret, paper=True)
        account = client.get_account()
        equity = float(account.equity)
    except Exception as e:
        logger.info(f"Failed to fetch account equity: {e}")
        equity = 10000.0  # fallback default for safety

    # Risk parameters
    confidence_threshold = 0.55
    max_position_frac = 0.10   # Max 10% of total equity
    risk_per_trade = 0.01      # Risk 1% of equity per trade
    stop_pct = 0.02            # 2% stop loss assumption

    try:
        price = float(position_data.get("avg_entry_price") or position_data.get("market_price"))
    except Exception:
        logger.info("Invalid price data in position_data.")
        return 0

    # Confidence-based scaling
    confidence_scale = max(0.0, (prob - confidence_threshold) / (1 - confidence_threshold))

    # Hybrid risk model
    max_position_value = equity * max_position_frac
    risk_dollars = equity * risk_per_trade

    hybrid_qty = (max_position_value * confidence_scale) / price
    risk_qty = risk_dollars / (price * stop_pct)

    qty = int(min(hybrid_qty, risk_qty))

    return qty