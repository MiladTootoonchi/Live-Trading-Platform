from typing import Protocol, Optional, Any
from dataclasses import dataclass
import numpy as np

import warnings

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from tensorflow.keras import Model

class ProbabilisticClassifier(Protocol):
    """
    Structural typing interface for probabilistic binary classifiers.

    This protocol is intentionally framework-agnostic and supports:
    - scikit-learn / XGBoost / LightGBM models via 'predict_proba'
    - Keras / TensorFlow models via direct forward pass (model(X))

    A conforming model must implement **at least one** of:
    - 'predict_proba(X)' → array of shape (n_samples, n_classes)
    - 'predict(X)' → array of probabilities in [0, 1]

    The positive class is assumed to be **class = 1**.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for the positive class.

        Typically implemented by Keras / TensorFlow models where
        'model.predict' returns probabilities directly.

        Args:
            X (np.ndarray):
                Feature matrix or tensor used for inference.
                Shape depends on the model (e.g. 2D or 3D).

        Returns:
            np.ndarray:
                Array of probabilities with shape (n_samples,)
                or (n_samples, 1).
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Typically implemented by scikit-learn compatible models.

        Args:
            X (np.ndarray):
                Feature matrix used for inference.

        Returns:
            np.ndarray:
                Array of shape (n_samples, n_classes), where:
                - [:, 1] corresponds to the probability of class = 1
        """
        ...


@dataclass
class ModelArtifact:
    """
    Serializable container holding everything required for inference.

    This object is designed to be saved and loaded atomically using 'joblib'
    and ensures reproducible inference across sessions.

    Attributes:
        model (ProbabilisticClassifier):
            Trained probabilistic classification model.

        scaler (Optional[Any]):
            Optional feature scaler or transformer applied
            before inference (e.g. StandardScaler).

        calibrator (Optional[Any]):
            Optional probability calibration model
            (e.g. CalibratedClassifierCV or isotonic regression).
    """

    model: ProbabilisticClassifier
    scaler: Optional[Any] = None
    calibrator: Optional[Any] = None


def extract_positive_class_probability(
    model: ProbabilisticClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """
    Extract the probability of the positive class (class = 1)
    from a trained probabilistic classifier.

    This function provides a **unified inference interface** across
    multiple ML frameworks by handling their API differences:

    - scikit-learn / XGBoost / LightGBM:
        Uses 'predict_proba(X)[:, 1]'
    - Keras / TensorFlow:
        Uses 'predict(X)' directly

    Args:
        model (ProbabilisticClassifier):
            Trained classification model implementing either
            'predict_proba' or 'predict'.

        X (np.ndarray):
            Input feature matrix or tensor for inference.

    Returns:
        np.ndarray:
            Array of probabilities for the positive class with
            shape (n_samples,).

    Raises:
        ValueError:
            If the model returns outputs with invalid shape or range.

        TypeError:
            If the model does not implement a compatible prediction method.
    """

    # sklearn-style API
    if callable(getattr(model, "predict_proba", None)):
        probs = np.asarray(model.predict_proba(X))

        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(
                "predict_proba must return array of shape (n_samples, n_classes)"
            )

        return probs[:, 1]

    # Keras-style API
    if isinstance(model, Model):
        # Direct forward pass – avoids Keras predict() bug
        preds = model(X, training=False)

        # Convert to numpy safely
        preds = np.asarray(preds).reshape(-1)

        if np.any((preds < 0) | (preds > 1)):
            raise ValueError(
                "Model.predict() must return probabilities in the range [0, 1]"
            )

        return preds

    raise TypeError(
        "Model must implement either predict_proba or predict"
    )



def get_model_name(model: Any) -> str:
    """
    Returns a stable, human-readable name for a model
    across different ML frameworks.
    """

    # Keras
    if hasattr(model, "name"):
        return model.name

    # sklearn / XGB / LGBM / CatBoost
    return model.__class__.__name__



def adapt_X_for_model(
    model: ProbabilisticClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """
    Adapts input feature shape to match model expectations.

    - Sequence models (Keras): expect 3D input → returned unchanged
    - Tabular models (sklearn, XGBoost, CatBoost): expect 2D input

    Policy for sequence → tabular:
    - Use last timestep (most recent information)

    Args:
        model:
            Trained or untrained model.
        X:
            Feature array of shape (N, T, F) or (N, F)

    Returns:
        np.ndarray:
            Adapted feature array.
    """

    # If already 2D, nothing to do
    if X.ndim == 2:
        return X

    # If sequence input
    if X.ndim == 3:
        # Keras models can handle 3D
        if isinstance(model, Model):
            return X

        # sklearn-style models → reduce sequence
        # Policy: last timestep
        return X[:, -1, :]

    raise ValueError(f"Unsupported input shape: {X.shape}")



def is_sequence_model(model) -> bool:
    """
    True if the model expects (N, T, F) input.
    """

    # Explicit opt-in (best practice)
    if getattr(model, "expects_sequence", False):
        return True

    # Wrapped Keras model
    if hasattr(model, "model") and isinstance(model.model, Model):
        shape = getattr(model.model, "input_shape", None)

    # Raw Keras model
    elif isinstance(model, Model):
        shape = getattr(model, "input_shape", None)

    else:
        return False

    if shape is None:
        return False

    return len(shape) == 3



def get_expected_feature_dim(model) -> int | None:
        """
        Returns expected feature dimension for a model if safely available.
        """

        # Wrapped Keras model (AutoencoderClassifierLite, etc.)
        if hasattr(model, "model") and isinstance(model.model, Model):
            if model.model.input_shape is not None:
                return model.model.input_shape[-1]
            return None

        # Raw Functional / Sequential Keras model
        if isinstance(model, Model):
            if hasattr(model, "input_shape") and model.input_shape is not None:
                return model.input_shape[-1]
            return None

        # Classical ML
        return None