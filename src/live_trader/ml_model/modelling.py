import numpy as np
from typing import Union, Tuple, Sequence

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Attention, LayerNormalization, Add, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC



def build_lstm(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds and compiles an LSTM model for sequence data.

    Args:
        X_train_seq (Union[np.ndarray, list]): Input feature sequences used to determine input shape.
            Expected shape is (num_samples, time_steps, num_features).

    Returns:
        Model: Compiled Keras LSTM model ready for training.
    """
    # Determine number of features dynamically
    n_features = X_train_seq.shape[2]

    # Use functional API with Input layer (instead of input_shape in LSTM)
    inputs = Input(shape=(None, n_features))  # variable time steps allowed¨
 
    x = LSTM(50, return_sequences = False)(inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs, name = "basic_lstm")

    model.compile(
        loss = 'binary_crossentropy', 
        optimizer = Adam(learning_rate=1e-3), 
        metrics = [
            AUC(name = "roc_auc")
            ]
    )
    
    return model


def build_attention_bilstm(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds and compiles an Attention-enhanced BiLSTM model for binary stock movement prediction.

    Args:
        X_train_seq (Union[np.ndarray, list]): Input feature sequences used to determine input shape.
            Expected shape is (num_samples, time_steps, num_features).

    Returns:
        Model: Compiled Keras BiLSTM-Attention model.
    """

    n_features = X_train_seq.shape[2]

    # Input Layer
    inputs = Input(shape = (None, n_features))

    # Bidirectional LSTM (returns full sequence for attention)
    x = Bidirectional(LSTM(64, return_sequences = True))(inputs)
    x = Dropout(0.3)(x)

    # Attention mechanism
    attn = Attention()([x, x])               # Self-attention over time steps
    x = Add()([x, attn])                     # Residual connection
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Binary output (up/down)
    outputs = Dense(1, activation='sigmoid')(x)

    # Build and compile
    model = Model(inputs, outputs, name = "attention_bilstm")

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = Adam(learning_rate = 1e-3),
        metrics = [
            AUC(name = "roc_auc")
            ]
    )

    return model