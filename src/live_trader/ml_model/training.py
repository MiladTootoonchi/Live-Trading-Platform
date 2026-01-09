import numpy as np
from typing import Tuple, Sequence, Union

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from .modelling import create_sequences, build_lstm, build_attention_bilstm



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
                y_seq: Union[np.ndarray, list]
                ) -> Model:
    """
    Builds and trains an LSTM model on the provided sequences.

    Args:
        X_seq (Union[np.ndarray, list]): Input feature sequences for training.
        y_seq (Union[np.ndarray, list]): Target sequences for training.

    Returns:
        Model: Trained Keras LSTM model.
    """

    lstm_model = build_lstm(X_seq)
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    
    lstm_model.fit(
        X_seq, y_seq, 
        epochs = 20, 
        batch_size = 16, 
        validation_split = 0.1,
        callbacks = [early_stop],
        verbose = 2)
    
    return lstm_model