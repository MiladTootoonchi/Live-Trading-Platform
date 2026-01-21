import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from live_trader.config import make_logger
from live_trader.ml_model.utils import get_model_name, extract_positive_class_probability, adapt_X_for_model

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras.models import Model
import tensorflow as tf

# --------------------------------------------------------------------------------------------

logger = make_logger()


def evaluate_model(model: Model,
                   symbol: str,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   save_dir: str = "logfiles/evaluations"
                   ) -> Tuple[float, float]:
    """
    Evaluates the trained model on unseen test data and logs performance metrics.

    This function assesses how well the model generalizes to new data by 
    calculating the F1 score and accuracy. It also saves a confusion matrix 
    and a classification report in the log directory for later inspection. 
    The function avoids console output and logs all activity instead.

    Args:
        model:
            Trained model (Keras, sklearn, XGBoost, etc.)
        symbol (str):
            The symbol of the stock that the ML-model is predicting on.
        X_test (np.ndarray): 
            Feature sequences used for testing the model.
        y_test (np.ndarray): 
            True target values corresponding to the test set.
        save_dir (str, optional): 
            Directory where evaluation files will be stored. Defaults to "logfiles".

    Returns:
        Tuple[float, float]: 
            The model's F1 score and accuracy on the test data.
    """

    # Ensure log directory exists
    model_name = get_model_name(model)
    save_dir = os.path.join(save_dir, model_name)

    os.makedirs(save_dir, exist_ok = True)

    # Reshape target if needed
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Predict probabilities and apply threshold
    try:
        X_test_ = adapt_X_for_model(model, X_test)
        y_pred_prob = extract_positive_class_probability(model, X_test_)
        y_pred = (y_pred_prob > 0.5).astype(int)
    except Exception as e:
        logger.error(f"Model prediction failed during evaluation: {e}")
        raise

    # Compute performance metrics
    try:
        auc_roc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division = 0)
        brier = brier_score(y_test, y_pred)

        evaluation_score_text = f"""

    Model evaluation for {symbol} completed:
    AUC-ROC         =   {auc_roc:.4f}
    f1-score:       =   {f1:.4f}
    brier-score     =   {brier:.4f}
        """

        logger.info(evaluation_score_text)
    except Exception as e:
        logger.error(f"Error while computing evaluation metrics: {e}")
        raise

    # Generate and save classification report
    try:
        report = classification_report(y_test, y_pred, digits = 4)
        report_path = os.path.join(save_dir, f"classification_report_{model_name}_{symbol}.txt")
        with open(report_path, "w") as f:
            f.write(report)
            f.write(evaluation_score_text)
        logger.info(f"Classification report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")

    # Generate and save confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", cbar = False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{symbol}.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        logger.error(f"Failed to create or save confusion matrix: {e}")

    return auc_roc, f1



def brier_score(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """
    Computes the Brier score for a binary
    stock direction classification model.

    The Brier score evaluates the calibration quality of
    predicted probabilities. USE THIS FOR TESTING EVALUASTION ONLY

    Args:
        y_true (array-like):
            Ground-truth binary labels (0 = down, 1 = up).
            Shape: (n_samples,)

        y_prob (array-like):
            Predicted probabilities for class 1 (up).
            Shape: (n_samples,)

    Returns:
        Tuple[float, float]:
            brier_score:
                Mean squared error between predicted probabilities
                and true binary outcomes. Lower is better.
    """

    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true, dtype = np.float32)
    y_prob = np.asarray(y_prob, dtype = np.float32)

    # -------------------------
    # Brier Score
    # -------------------------
    brier_score = np.mean((y_prob - y_true) ** 2)

    return brier_score



@tf.keras.utils.register_keras_serializable(package="LiveTrader")
def brier(
    y_true: tf.Tensor,
    y_pred: tf.Tensor
) -> tf.Tensor:
    """
    TensorFlow Brier score metric for binary classification.

    Args:
        y_true (tf.Tensor):
            Ground-truth binary labels.

        y_pred (tf.Tensor):
            Predicted probabilities for the positive class.

    Returns:
        tf.Tensor:
            Mean squared error between probabilities and labels.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(tf.square(y_pred - y_true))