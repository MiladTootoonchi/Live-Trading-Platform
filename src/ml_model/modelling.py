import numpy as np
from typing import Union, Tuple, Sequence

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import numpy as np



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
    model = Sequential()
    model.add(LSTM(50, input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
