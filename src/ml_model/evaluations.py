import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Ignoring info + warning + errors: the user do not need to see this
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras.models import Model

from config import make_logger

logger = make_logger()


def evaluate_model(model: Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   save_dir: str = "logfiles"
                   ) -> Tuple[float, float]:
    """
    Evaluates the trained model on unseen test data and logs performance metrics.

    This function assesses how well the model generalizes to new data by 
    calculating the F1 score and accuracy. It also saves a confusion matrix 
    and a classification report in the log directory for later inspection. 
    The function avoids console output and logs all activity instead.

    Args:
        model (Model): 
            The trained LSTM model to be evaluated.
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
    os.makedirs(save_dir, exist_ok = True)

    # Reshape target if needed
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Predict probabilities and apply threshold
    try:
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
    except Exception as e:
        logger.error(f"Model prediction failed during evaluation: {e}")
        raise

    # Compute performance metrics
    try:
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model evaluation completed — F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error while computing evaluation metrics: {e}")
        raise

    # Generate and save classification report
    try:
        report = classification_report(y_test, y_pred, digits=4)
        report_path = os.path.join(save_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Classification report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")

    # Generate and save confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        logger.error(f"Failed to create or save confusion matrix: {e}")

    return f1, accuracy
