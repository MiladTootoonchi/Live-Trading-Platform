import numpy as np
from typing import Tuple, Sequence, Union, Callable
import pandas as pd
import numpy as np
import datetime as dt

from live_trader import SideSignal
from config import make_logger
from .data import stock_data_prediction_pipeline, stock_data_feature_engineering, get_one_realtime_bar, compute_trade_qty
from live_trader.strategies.utils import fetch_data
from .evaluations import evaluate_model

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from .data import create_sequences

# ------------------------------------------------------------------------

def sequence_split(X: Union[Sequence, np.ndarray], 
                   y: Union[Sequence, np.ndarray], 
                   time_steps: int = 5
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts input data into sequences and splits them into training and testing sets.

    Args:
        X (array-like): Feature dataset to be converted into sequences.
        y (array-like): Target dataset corresponding to X.
        time_steps (int, optional): Number of time steps in each sequence. Defaults to 5.

    Returns:
        tuple: 
            X_train_seq (array-like): Training sequences for features.
            X_test_seq (array-like): Testing sequences for features.
            y_train_seq (array-like): Training sequences for targets.
            y_test_seq (array-like): Testing sequences for targets.
    """

    X_seq, y_seq = create_sequences(X, y, time_steps)

    # Train/test split
    split = int(0.8 * len(X_seq))
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq


def train_model(X_seq: Union[np.ndarray, list],
                y_seq: Union[np.ndarray, list],
                model: Model
                ) -> Model:
    """
    Builds and trains an LSTM model on the provided sequences.

    Args:
        X_seq (Union[np.ndarray, list]): Input feature sequences for training.
        y_seq (Union[np.ndarray, list]): Target sequences for training.
        model (Model): A ML model architecture build by tensorflow blocks.

    Returns:
        Model: Trained Keras LSTM model.
    """

    early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    
    model.fit(
        X_seq, y_seq, 
        epochs = 20, 
        batch_size = 16, 
        validation_split = 0.1,
        callbacks = [early_stop],
        verbose = 2)
    
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


    yesterday = dt.datetime.now(dt.UTC) - dt.timedelta(days = 1)
    mdip = dt.datetime.now(dt.UTC) - dt.timedelta(days = 100)        # more than 50 days in the past (many days in past)

    # fetching data from 2020 till mdip (mdip till today will be used for prediction)
    df = fetch_data(symbol = symbol, 
                    start_date = (2020, 1, 1), 
                    end_date = (mdip.year, mdip.month, mdip.day))
    
    # Feature engineering for training and test data
    X, y, scaler = stock_data_feature_engineering(df)

    X_train, X_test, y_train, y_test = sequence_split(X, y) # splitting

    model = model_builder(X_train)
    model = train_model(X_train, y_train, model)   # training

    # Evaluation
    auc_roc, f1 = evaluate_model(model, symbol, X_test, y_test)

    # Fetching mdip data till today (realtime data)
    realtime_bar = await get_one_realtime_bar(symbol = symbol)

    hist_df = fetch_data(symbol = symbol, 
                        start_date = (mdip.year, mdip.month, mdip.day), 
                        end_date = (yesterday.year, yesterday.month, yesterday.day))
    
    hist_df = hist_df.reset_index()     # making sure there isnt any dobble index

    # Drop fully-NA rows
    realtime_bar = realtime_bar.dropna(how = "all")

    # Drop fully-NA columns (this is what removes the FutureWarning)
    realtime_bar = realtime_bar.dropna(axis = 1, how = "all")

    # Now safe to concatenate
    pred_df = pd.concat(
        [hist_df.tail(50), realtime_bar],
        ignore_index = True
    )
    
    # Preprocessing pred_data
    X = stock_data_prediction_pipeline(pred_df, scaler)

    # predicting
    X_seq = np.expand_dims(X[-50:], axis=0)
    real_time_prediction = model.predict(X_seq)

    signal = int((real_time_prediction > 0.5).astype(int).item())

    # Getting probability for prediction
    if signal == 0:
        prob = 1 - real_time_prediction.item()
    else:
        prob = real_time_prediction.item()

    info = f"""

    The stock will go (1 for up, 0 for down) = {signal} tommorow 
    with {(prob * 100):.4f} % probability.
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