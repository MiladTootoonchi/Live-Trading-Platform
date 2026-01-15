import numpy as np
from typing import Callable, Optional, Any
import pandas as pd
import datetime as dt
from pathlib import Path
import joblib

from live_trader.alpaca_trader.order import SideSignal
from live_trader.strategies.utils import fetch_data
from live_trader.config import make_logger
from live_trader.ml_model.utils import (
    ProbabilisticClassifier, ModelArtifact, extract_positive_class_probability,
    is_sequence_model, get_expected_feature_dim
)

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
    model: ProbabilisticClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Optional[IsotonicRegression]:
    """
    Fits an isotonic probability calibrator using validation data.

    This function is framework-agnostic and supports:
    - scikit-learn / XGBoost / LightGBM (predict_proba)
    - Keras / TensorFlow (predict)

    Calibration is skipped if the validation targets
    contain only a single class.

    Args:
        model (ProbabilisticClassifier):
            Trained probabilistic classifier.
        X_val (np.ndarray):
            Validation feature matrix.
        y_val (np.ndarray):
            Binary validation labels.

    Returns:
        Optional[IsotonicRegression]:
            Fitted isotonic calibrator, or None if calibration
            is not possible.
    """

    # Cannot calibrate if only one class is present
    if len(np.unique(y_val)) < 2:
        return None

    # Get raw probabilities (positive class)
    
    if is_sequence_model(model):
        X_val_ = X_val
    else:
        X_val_ = X_val[:, -1, :]

    raw_probs = extract_positive_class_probability(model, X_val_)

    # Defensive reshaping
    raw_probs = np.asarray(raw_probs).reshape(-1)
    y_val = np.asarray(y_val).reshape(-1)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_val)

    return calibrator



def train_model(
    model: ProbabilisticClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Model:
    """
    Train a time-series model using chronological validation.

    Args:
        model (ProbabilisticClassifier): Compiled Keras model.
        X_train (np.ndarray): Training sequences.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation sequences.
        y_val (np.ndarray): Validation targets.

    Returns:
        Model:
            Trains a model using the appropriate strategy depending on framework.

            - Keras models: uses epochs, callbacks, validation
            - sklearn-style models: calls fit(X, y)
    """

    
    # --- reshape inputs if needed ---
    if is_sequence_model(model):
        X_train_ = X_train
        X_val_ = X_val
    else:
        X_train_ = X_train[:, -1, :]
        X_val_ = X_val[:, -1, :]


    # --- Keras-style training ---
    if isinstance(model, Model):
        early_stop = EarlyStopping(
            monitor="val_brier",
            mode="min",
            patience=5,
            restore_best_weights=True,
        )

        model.fit(
            X_train_,
            y_train,
            validation_data=(X_val_, y_val),
            epochs=20,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=2,
        )
        return model

    # --- sklearn / XGB / LGBM / CatBoost ---
    fit_kwargs = {}

    if "eval_set" in model.fit.__code__.co_varnames:
        fit_kwargs["eval_set"] = [(X_val_, y_val)]

    if "verbose" in model.fit.__code__.co_varnames:
        fit_kwargs["verbose"] = False

    model.fit(X_train_, y_train, **fit_kwargs)
    return model

logger = make_logger()



# -----------------------------------------------------------------------------------------------------



async def ML_Pipeline(model_builder: Callable[[np.ndarray], ProbabilisticClassifier], 
                      symbol: str, position_data: dict) -> tuple[SideSignal, int]:
    """
    Evaluates a trading position from an Alpaca JSON response and recommends an action.
    This strategy will only buy or sell.

    Args:
        model_builder (Callable[[np.ndarray], ProbabilisticClassifier]):
            Function that constructs and returns an untrained model.

        symbol (str):   
            a string consisting of the symbol og the stock.

        position_data (dict):   
            JSON object from Alpaca API containing position details.
            This is only used as a parameter if you have a posistion in that stock.

    Returns:
        tuple:
            (SideSignal.BUY or SideSignal.SELL, qty: int)
    """

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_name = getattr(model_builder, "__name__", model_builder.__class__.__name__)

    MODEL_PATH = MODEL_DIR / f"{model_name}_{symbol}.joblib"

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
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        model = artifact.model
        scaler = artifact.scaler
        calibrator = artifact.calibrator


    else:
        logger.info(f"No existing model found for {symbol}, training...")

        X, y, scaler = prepare_training_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = sequence_split(X, y)

        # Build (architecture only)
        model = model_builder(X_train)

        # Train
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

        # Evaluation
        auc_roc, f1 = evaluate_model(model, symbol, X_test, y_test)

        artifact = ModelArtifact(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
        )

        joblib.dump(artifact, MODEL_PATH)


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
    X_flat = prepare_prediction_data(pred_df, scaler)

    expected_features = get_expected_feature_dim(model)

    if expected_features is not None:
        if X_flat.shape[1] != expected_features:
            raise RuntimeError(
                f"Feature mismatch at prediction time. "
                f"Model expects {expected_features} features, "
                f"but got {X_flat.shape[1]}"
            )

    TIME_STEPS = 50

    if is_sequence_model(model):
        # --- sequence model ---
        if len(X_flat) < TIME_STEPS:
            raise RuntimeError(
                f"Not enough data for LSTM prediction: "
                f"need {TIME_STEPS}, got {len(X_flat)}"
            )

        X_seq, _ = create_sequences(
            X_flat,
            np.zeros(len(X_flat)),
            TIME_STEPS
        )

        if len(X_seq) == 0:
            raise RuntimeError(
                f"create_sequences returned empty array. "
                f"X_flat shape={X_flat.shape}, TIME_STEPS={TIME_STEPS}"
            )

        X_last = X_seq[-1:]   # (1, T, F)

    else:
        # --- classical ML ---
        X_last = X_flat[-1:]       # (1, F)

    raw_prob = extract_positive_class_probability(model, X_last)[0]

    if calibrator is not None:
        prob = calibrator.predict([raw_prob])[0]
    else:
        prob = raw_prob

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