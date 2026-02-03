import numpy as np
from typing import Callable, Optional, Any
import pandas as pd
from pathlib import Path
import joblib
from alpaca.trading.client import TradingClient

from live_trader.alpaca_trader.order import SideSignal
from live_trader.config import Config
from live_trader.ml_model.utils import (
    ProbabilisticClassifier, ModelArtifact, extract_positive_class_probability,
    is_sequence_model, get_expected_feature_dim
)

from live_trader.ml_model.layers import (Patchify, GraphMessagePassing, ExpandDims, AutoencoderClassifierLite)
from live_trader.ml_model.evaluations import evaluate_model
from live_trader.ml_model.data import MLDataPipeline


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


def _compute_trade_qty(self, position_data: dict, prob: float) -> int:
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

    try:
        client = TradingClient(self._key, self._secret, paper=True)
        account = client.get_account()
        equity = float(account.equity)
    except Exception as e:
        self._config.log_info(f"Failed to fetch account equity: {e}")
        equity = 10000.0  # fallback default for safety

    # Risk parameters
    confidence_threshold = 0.55
    max_position_frac = 0.10        # Max 10% of total equity
    risk_per_trade = 0.01           # Risk 1% of equity per trade
    stop_pct = 0.02                 # 2% stop loss assumption

    try:
        price = float(position_data.get("avg_entry_price") or position_data.get("market_price"))
    except Exception:
        self._config.log_info("Invalid price data in position_data.")
        return 0

    # Confidence-based scaling
    confidence_scale = max(0.0, (prob - confidence_threshold) / (1 - confidence_threshold))

    # Hybrid risk model
    max_position_value = equity * max_position_frac
    risk_dollars = equity * risk_per_trade

    hybrid_qty = (max_position_value * confidence_scale) / price
    risk_qty = risk_dollars / (price * stop_pct)

    qty = int(min(hybrid_qty, risk_qty))

    # If model wants to SELL (negative qty), cap by position size
    try:
        position_qty = int(float(position_data.get("qty", 0)))

        if qty < 0:
            qty = max(qty, -position_qty)
    except Exception:
        self._config.log_info("Invalid position quantity data.\n")
        return 0

    return qty


def _check_model_existence(
    model_builder: Callable[[np.ndarray], ProbabilisticClassifier],
    symbol: str,
    data_pipeline: MLDataPipeline,
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

    else:
        logger.info(f"No existing model found for {symbol}, training...")

        X, y, scaler = data_pipeline.prepare_training_data(data_pipeline.df)
        X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline.sequence_split(X, y, data_pipeline.time_steps)

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



def _prepare_model_input(
    data_pipeline,
    scaler: Any,
    model: ProbabilisticClassifier,
) -> np.ndarray:
    """
    Converts prediction data into a model-ready input tensor.

    Validates feature columns, applies scaling, enforces expected feature
    dimensions, and adapts the shape for sequence or non-sequence models.

    Args:
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

    missing = [c for c in EXPECTED_COLS if c not in data_pipeline.pred_df.columns]
    if missing:
        raise RuntimeError(f"Missing prediction features: {missing}")

    X_flat = data_pipeline.prepare_prediction_data(data_pipeline.pred_df, scaler)

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
        if len(X_flat) < data_pipeline.time_steps:
            raise RuntimeError(
                f"Not enough data for sequence prediction: "
                f"need {data_pipeline.time_steps}, got {len(X_flat)}"
            )

        X_seq, _ = data_pipeline.create_sequences(
            X_flat,
            np.zeros(len(X_flat)),
            data_pipeline.time_steps,
        )

        if len(X_seq) == 0:
            raise RuntimeError(
                f"create_sequences returned empty array "
                f"(X_flat = {X_flat.shape}, time_steps = {data_pipeline.time_steps})"
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

    conf = Config()
    data_pipeline = MLDataPipeline(conf, position_data)
    data_pipeline.run()
    
    artifact = _check_model_existence(model_builder, symbol, data_pipeline)
    scaler = artifact.scaler
    model = artifact.model
    calibrator = artifact.calibrator

    X_last = _prepare_model_input(
        data_pipeline=data_pipeline,
        scaler=scaler,
        model=model,
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
        qty = _compute_trade_qty(position_data, float(prob))

    elif signal == 1 and prob > 0.5:
        qty = 1

    else:
        qty = 0

    if has_position:
        side = SideSignal.BUY if signal == 1 else SideSignal.SELL
    else:
        side = SideSignal.BUY if qty > 0 else SideSignal.HOLD

    return side, qty