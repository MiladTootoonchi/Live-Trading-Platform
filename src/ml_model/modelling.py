import numpy as np
from typing import Union, Tuple, Sequence

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Attention, LayerNormalization, Add, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam


def create_sequences(X: Union[Sequence, np.ndarray],
                        y: Union[Sequence, np.ndarray],
                        time_steps: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sequences from input features and targets for time series modeling.

    Args:
        X (Union[Sequence, np.ndarray]): Feature dataset.
        y (Union[Sequence, np.ndarray]): Target dataset.
        time_steps (int): Number of time steps in each sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Xs: Array of feature sequences of shape (num_sequences, time_steps, num_features).
            ys: Array of target values corresponding to each sequence.
    """

    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])

        if hasattr(y, "iloc"):
            ys.append(y.iloc[i + time_steps])
        else:
            ys.append(y[i + time_steps])
            
    return np.array(Xs), np.array(ys)



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

    model = Model(inputs, outputs)

    model.compile(
        loss = 'binary_crossentropy', 
        optimizer = 'adam', 
        metrics = ['accuracy']
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
    inputs = Input(shape=(None, n_features))

    # Bidirectional LSTM (returns full sequence for attention)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
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
    model = Model(inputs, outputs)

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = Adam(learning_rate=1e-3),
        metrics = ['accuracy']
    )

    return model