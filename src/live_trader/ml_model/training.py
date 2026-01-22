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
from live_trader.ml_model.evaluations import evaluate_model
from live_trader.ml_model.data import (prepare_training_data, prepare_prediction_data, ensure_clean_timestamp,
                                       get_one_realtime_bar, compute_trade_qty, create_sequences, 
                                       MIN_LOOKBACK, TIME_STEPS, SAFETY_MARGIN)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.isotonic import IsotonicRegression

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

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



def _calibrate_probabilities(
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

def _sanitize_time_index(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Validates and normalizes a DataFrame's DatetimeIndex.

    This function ensures that the DataFrame index is a valid
    'pandas.DatetimeIndex', localizes naive timestamps to UTC,
    removes rows with invalid (NaT) timestamps, and sorts the
    DataFrame by time.

    If the index is not a DatetimeIndex or if the DataFrame becomes
    empty after cleanup, a RuntimeError is raised. Any dropped
    timestamps are logged with contextual information to aid debugging.

    Args:
        df (pd.DataFrame):
            The input DataFrame whose index is expected to represent time.
        context (str):
            A descriptive label used in log messages and exception text
            to identify the caller or data source.

    Returns:
        pd.DataFrame:
            A cleaned DataFrame with a timezone-aware UTC DatetimeIndex
            and sorted in ascending time order.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(f"{context}: index is not DatetimeIndex")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    bad = df.index.isna()
    if bad.any():
        logger.error(f"{context}: dropping {bad.sum()} invalid timestamps")
        df = df.loc[~bad]

    if df.empty:
        raise RuntimeError(f"{context}: dataframe empty after timestamp cleanup")

    return df.sort_index()



def _check_model_existence(
    model_builder: Callable[[np.ndarray], ProbabilisticClassifier],
    symbol: str,
    df: pd.DataFrame,
) -> ModelArtifact:
    """
    Loads a persisted model artifact or trains and saves a new one.

    If a model artifact exists for the given model builder and symbol,
    it is loaded from disk. Otherwise, a new model is trained using the
    provided historical data, evaluated, calibrated, and persisted.

    Args:
        model_builder:
            Callable that builds an untrained probabilistic classifier.
        symbol:
            Trading symbol associated with the model.
        df:
            Historical market data used for training if no model exists.

    Returns:
        ModelArtifact
    """

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    model_id = f"{model_builder.__name__}_{symbol}"
    model_path = model_dir / f"{model_id}.joblib"

    if model_path.exists():
        artifact = joblib.load(model_path)
        model = artifact.model
        scaler = artifact.scaler
        calibrator = artifact.calibrator

    else:
        logger.info(f"No existing model found for {symbol}, training...")

        X, y, scaler = prepare_training_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = sequence_split(X, y, TIME_STEPS)

        # Build (architecture only)
        model = model_builder(X_train)

        # Train
        model = train_model(
            model = model,
            X_train = X_train,
            y_train = y_train,
            X_val = X_val,
            y_val = y_val,
        )

        calibrator = _calibrate_probabilities(
            model = model,
            X_val = X_val,
            y_val = y_val,
        )

        # Evaluation
        _, _ = evaluate_model(model, symbol, X_test, y_test)

        artifact = ModelArtifact(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
        )

        joblib.dump(artifact, model_path)

    return artifact



async def _build_prediction_dataframe(
    symbol: str,
    hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetches and prepares historical and realtime market data for prediction.

    Combines historical bars from `pred_start_date` to now with the latest
    realtime bar, normalizes timestamps, and ensures required raw columns exist.

    Args:
        symbol (str):
            Trading symbol (e.g. "AAPL").
        hist_df (pd.DataFrame):
            The data you want to change to a prediction dataframe

    Returns:
        pd.DataFrame:
            Cleaned, time-ordered DataFrame ready for feature preprocessing.

    Raises:
        RuntimeError:
            If no valid historical data is available after cleanup.
    """

    hist_df = _sanitize_time_index(hist_df, "PREDICTION DATA")
    hist_df = hist_df.reset_index()     # ensure no double index

    if hist_df.empty:
        raise RuntimeError("Prediction history empty after timestamp normalization")

    last_close = hist_df.iloc[-1]["close"]
    realtime_bar = await get_one_realtime_bar(
        symbol=symbol,
        last_close=last_close,
    )

    # Drop fully-NA rows/columns
    realtime_bar = realtime_bar.dropna(how="all")
    realtime_bar = realtime_bar.dropna(axis=1, how="all")

    INDICATOR_WARMUP = MIN_LOOKBACK

    PRED_HISTORY = TIME_STEPS + INDICATOR_WARMUP

    pred_df = pd.concat(
        [hist_df.tail(PRED_HISTORY + 1), realtime_bar],
        ignore_index=True,
    ).copy()

    pred_df = ensure_clean_timestamp(pred_df)

    for col in ["vwap", "trade_count"]:
        if col not in pred_df.columns:
            pred_df.loc[:, col] = 0.0

    return pred_df



def _prepare_model_input(
    pred_df: pd.DataFrame,
    scaler: Any,
    model: ProbabilisticClassifier,
    time_steps: int = 50,
) -> np.ndarray:
    """
    Converts prediction data into a model-ready input tensor.

    Validates feature columns, applies scaling, enforces expected feature
    dimensions, and adapts the shape for sequence or non-sequence models.

    Args:
        pred_df (pd.DataFrame):
            Cleaned prediction DataFrame containing raw features.
        scaler (Any):
            Fitted feature scaler used during training.
        model (ProbabilisticClassifier):
            Trained model used to determine input shape requirements.
        time_steps (int):
            Sequence length for sequence-based models.

    Returns:
        np.ndarray:
            Final input tensor suitable for direct model inference.

    Raises:
        RuntimeError:
            If required features are missing or data is insufficient.
    """
    EXPECTED_COLS = [
    "open", "high", "low", "close",
    "volume", "trade_count", "vwap",
    ]

    missing = [c for c in EXPECTED_COLS if c not in pred_df.columns]
    if missing:
        raise RuntimeError(f"Missing prediction features: {missing}")

    X_flat = prepare_prediction_data(pred_df, scaler)

    if X_flat.shape[0] == 0:
        raise RuntimeError(
            "prepare_prediction_data returned empty array "
            "(insufficient history after indicators)"
        )

    expected_features = get_expected_feature_dim(model)
    if expected_features is not None and X_flat.shape[1] != expected_features:
        raise RuntimeError(
            f"Feature mismatch: expected {expected_features}, got {X_flat.shape[1]}"
        )

    if is_sequence_model(model):
        if len(X_flat) < time_steps:
            raise RuntimeError(
                f"Not enough data for sequence prediction: "
                f"need {time_steps}, got {len(X_flat)}"
            )

        X_seq, _ = create_sequences(
            X_flat,
            np.zeros(len(X_flat)),
            time_steps,
        )

        if len(X_seq) == 0:
            raise RuntimeError(
                f"create_sequences returned empty array "
                f"(X_flat = {X_flat.shape}, time_steps = {time_steps})"
            )


        X_last = X_seq[-1:]   # (1, T, F)

    else:
        X_last = X_flat[-1:]  # (1, F)

    return X_last



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

    is_backtest = position_data.get("backtest", False)
    if is_backtest:
        # last available bar timestamp
        current_ts = pd.to_datetime(position_data["history"][-1]["t"], utc=True)
    else:
        current_ts = dt.datetime.now(dt.UTC)


    # ml_training_lookback >= PRED_HISTORY
    ml_training_lookback = 750

    INDICATOR_WARMUP = MIN_LOOKBACK
    PRED_HISTORY = TIME_STEPS + INDICATOR_WARMUP + SAFETY_MARGIN

    # pred_start_date must be more than 50 days in the past
    pred_start_date = current_ts - dt.timedelta(days = PRED_HISTORY)

    start_dt = pred_start_date - dt.timedelta(days = ml_training_lookback)
    start_date = (start_dt.year, start_dt.month, start_dt.day)

    end_dt = pred_start_date - dt.timedelta(days = 1)
    end_date = (end_dt.year, end_dt.month, end_dt.day)

    # fetching data from as early as possible till pred_start_date (pred_start_date till today will be used for prediction)
    df = fetch_data(symbol = symbol, 
                    start_date = start_date, 
                    end_date = end_date)
    df = ensure_clean_timestamp(df)
    
    df = _sanitize_time_index(df, "TRAINING DATA")
    
    artifact = _check_model_existence(model_builder, symbol, df)
    scaler = artifact.scaler
    model = artifact.model
    calibrator = artifact.calibrator

    if is_backtest:
        hist = pd.DataFrame(position_data["history"])
        hist["timestamp"] = pd.to_datetime(hist["t"], utc=True)
        hist = hist.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }).set_index("timestamp")

        pred_df = hist.tail(PRED_HISTORY).copy()

        for col in ["vwap", "trade_count"]:
            if col not in pred_df.columns:
                pred_df[col] = 0.0


    else:
        pred_df = await _build_prediction_dataframe(
            symbol=symbol,
            hist_df = df,
        )

    X_last = _prepare_model_input(
        pred_df=pred_df,
        scaler=scaler,
        model=model,
        time_steps=TIME_STEPS,
    )


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