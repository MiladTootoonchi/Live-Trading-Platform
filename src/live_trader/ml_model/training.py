import numpy as np
from typing import Callable
import pandas as pd
import datetime as dt
from pathlib import Path
import joblib

from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.utils import fetch_data
from live_trader.config import make_logger

from live_trader.ml_model.layers import (Patchify, GraphMessagePassing, ExpandDims, AutoencoderClassifierLite)
from live_trader.ml_model.evaluations import evaluate_model, brier
from live_trader.ml_model.data import (prepare_training_data, prepare_prediction_data, 
                                       get_one_realtime_bar, compute_trade_qty, create_sequences)

from sklearn.isotonic import IsotonicRegression

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ------------------------------------------------------------------------

def sequence_split(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert features into time sequences and split chronologically.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        time_steps (int): Sequence length.
        train_ratio (float): Fraction used for training.
        val_ratio (float): Fraction used for validation.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """

    X_seq, y_seq = create_sequences(X, y, time_steps)

    n = len(X_seq)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]

    X_val = X_seq[train_end:val_end]
    y_val = y_seq[train_end:val_end]

    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test



def calibrate_probabilities(
    model: Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> IsotonicRegression:
    """
    Fit a probability calibrator on validation predictions.

    Args:
        model (Model): Trained classification model.
        X_val (np.ndarray): Validation sequences.
        y_val (np.ndarray): Validation targets.

    Returns:
        IsotonicRegression: Fitted probability calibrator.
    """

    if len(np.unique(y_val)) < 2:
        return None

    raw_probs = model.predict(X_val).ravel()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_val)

    return calibrator



def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Model:
    """
    Train a time-series model using chronological validation.

    Args:
        model (Model): Compiled Keras model.
        X_train (np.ndarray): Training sequences.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation sequences.
        y_val (np.ndarray): Validation targets.

    Returns:
        Model: Trained Keras model.
    """

    early_stop = EarlyStopping(monitor = 'val_brier', mode = "min", patience = 5, restore_best_weights = True)
    
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=2,
    )
    
    return model


logger = make_logger()



# -----------------------------------------------------------------------------------------------------



async def ML_Pipeline(model_builder: Callable[[np.ndarray], Model], symbol: str, position_data: dict) -> tuple[SideSignal, int]:
    """
    Evaluates a trading position from an Alpaca JSON response and recommends an action.
    This strategy will only buy or sell.

    Args:
        model (Model):          A tensorflow build ML model.
        symbol (str):           a string consisting of the symbol og the stock.
        position_data (dict):   JSON object from Alpaca API containing position details.
                                This is only used as a parameter if you have a posistion in that stock.

    Returns:
        tuple:
            (SideSignal.BUY or SideSignal.SELL, qty: int)
    """

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_name = model_builder.__name__

    MODEL_PATH = MODEL_DIR / f"{model_name}_{symbol}_model.keras"
    SCALER_PATH = MODEL_DIR / f"{model_name}_{symbol}_scaler.pkl"
    CALIBRATOR_PATH = MODEL_DIR / f"{model_name}_{symbol}_calibrator.pkl"


    time_steps = 10
    lookback_days = 750

    yesterday = dt.datetime.now(dt.UTC) - dt.timedelta(days = 1)
    INDICATOR_WARMUP = 150      # must cover largest rolling window
    PRED_HISTORY = 150

    mdip = dt.datetime.now(dt.UTC) - dt.timedelta(
        days = INDICATOR_WARMUP + PRED_HISTORY
    )        # more than 50 days in the past (many days in past)

    start_dt = dt.datetime.now(dt.UTC) - dt.timedelta(days=lookback_days)
    start_date = (start_dt.year, start_dt.month, start_dt.day)


    # fetching data from as early as possible till mdip (mdip till today will be used for prediction)
    df = fetch_data(symbol = symbol, 
                    start_date = start_date, 
                    end_date = (mdip.year, mdip.month, mdip.day))
    
    # Feature engineering for training and test data, then training
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        logger.info(f"Loading existing model for {symbol}")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        if CALIBRATOR_PATH.exists():
            calibrator = joblib.load(CALIBRATOR_PATH)
        else:
            calibrator = None

    else:
        logger.info(f"No existing model found for {symbol}, training...")

        X, y, scaler = prepare_training_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = sequence_split(X, y)

        model = model_builder(X_train)

        model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        calibrator = calibrate_probabilities(
            model=model,
            X_val=X_val,
            y_val=y_val,
        )

        if calibrator is not None:
            joblib.dump(calibrator, CALIBRATOR_PATH)

        # Evaluation
        auc_roc, f1 = evaluate_model(model, symbol, X_test, y_test)

        joblib.dump(scaler, SCALER_PATH)
        model.save(MODEL_PATH)
    
    # Fetching mdip data till today (realtime data)

    hist_df = fetch_data(symbol = symbol, 
                        start_date = (mdip.year, mdip.month, mdip.day), 
                        end_date = (yesterday.year, yesterday.month, yesterday.day))
    
    hist_df = hist_df.reset_index()     # making sure there isnt any dobble index

    last_close = hist_df.iloc[-1]['close']
    realtime_bar = await get_one_realtime_bar(symbol = symbol, last_close = last_close)

    # Drop fully-NA rows
    realtime_bar = realtime_bar.dropna(how = "all")

    # Drop fully-NA columns (this is what removes the FutureWarning)
    realtime_bar = realtime_bar.dropna(axis = 1, how = "all")

    PRED_HISTORY = 150  # must exceed max indicator window

    pred_df = pd.concat(
        [hist_df.tail(PRED_HISTORY), realtime_bar],
        ignore_index=True
    )

    
    # Preprocessing pred_data
    X = prepare_prediction_data(pred_df, scaler)

    # predicting
    X_seq = np.expand_dims(X[-time_steps:], axis=0)
    
    raw_prob = model.predict(X_seq).item()

    if calibrator is None:
        prob = raw_prob
    else:
        prob = calibrator.predict([raw_prob])[0]

    signal = int(prob > 0.5)

    direction = "UP" if signal == 1 else "DOWN"
    confidence = prob if signal == 1 else 1 - prob

    info = f"""

    The stock ({symbol}) will go {direction} 
    with {(confidence * 100):.2f} % confidence.
    """

    logger.info(info)


    # Deciding side and qty
    has_position = (
        position_data is not None
        and float(position_data.get("qty", 0)) > 0
    )

    if has_position:
        qty = compute_trade_qty(position_data, float(prob))

    elif signal == 1 and prob > 0.5:
        qty = 1

    else:
        qty = 0

    if has_position:
        side = SideSignal.BUY if signal == 1 else SideSignal.SELL
    else:
        side = SideSignal.BUY if qty > 0 else SideSignal.HOLD

    return side, qty