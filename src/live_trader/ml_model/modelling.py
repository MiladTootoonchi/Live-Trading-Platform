import numpy as np
from typing import Union

from .training import brier

# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Attention, LayerNormalization, Add, GlobalAveragePooling1D, 
    Conv1D, MultiHeadAttention, Lambda, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import keras
from keras import ops

# --------------------  Basic LSTM  --------------------

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
            AUC(name = "roc_auc"),
            brier
            ]
    )
    
    return model



# --------------------  Attention BiLSTM  --------------------



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
            AUC(name = "roc_auc"),
            brier
            ]
    )

    return model



# --------------------  TCN-lite  --------------------



def build_tcn_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds a lightweight Temporal Convolutional Network (TCN-style)
    for noisy financial time series classification.

    Designed to be robust to non-stationarity and overfitting.

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    x = Conv1D(
        filters=32,
        kernel_size=3,
        padding="causal",
        activation="relu"
    )(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(
        filters=16,
        kernel_size=3,
        padding="causal",
        activation="relu"
    )(x)
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="tcn_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            brier
        ]
    )

    return model



# --------------------  PatchTST-lite --------------------



def build_patchtst_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds a lightweight PatchTST-style Transformer model for
    noisy financial time series classification.

    The model splits the time dimension into patches, embeds them,
    and applies a Transformer encoder for temporal modeling.

    Designed for robustness to non-stationarity and overfitting.

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    patch_len: int = 16
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.3

    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    # Patch embedding
    # Split time dimension into non-overlapping patches
    def patchify(x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]

        # Ensure at least one patch
        pad_len = tf.maximum(0, patch_len - time_steps)
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])

        # Recompute after padding
        time_steps = tf.shape(x)[1]
        n_patches = time_steps // patch_len

        x = x[:, :n_patches * patch_len, :]
        x = tf.reshape(x, (batch_size, n_patches, patch_len * n_features))
        return x

    x = tf.keras.layers.Lambda(patchify, name="patchify")(inputs)

    x = Dense(d_model, activation="linear")(x)
    x = LayerNormalization()(x)

    # Transformer Encoder Block
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)

    x = LayerNormalization()(x + attn_out)

    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dropout(dropout)(ff_out)
    ff_out = Dense(d_model)(ff_out)

    x = LayerNormalization()(x + ff_out)

    # Pooling & Head
    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="patchtst_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            brier
        ]
    )

    return model



# --------------------  GNN-lite --------------------



class GraphMessagePassing(tf.keras.layers.Layer):
    """
    Simple graph message-passing layer with learned adjacency.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def build(self, input_shape):
        # input_shape: (B, N_nodes, D)
        n_nodes = input_shape[1]

        self.adjacency = Dense(
            n_nodes,
            activation="tanh",
            name="learned_adjacency"
        )
        self.node_update = Dense(self.hidden_dim, activation="relu")
        self.norm = LayerNormalization()
        self.drop = Dropout(self.dropout)

        super().build(input_shape)

    def call(self, x):
        # x: (B, N, D)
        A = self.adjacency(x)          # (B, N, N)
        messages = tf.matmul(A, x)    # (B, N, D)
        x = self.node_update(messages)
        x = self.norm(x)
        return self.drop(x)



def build_gnn_lite(X_train_seq: Union[np.ndarray, list],) -> Model:
    """
    Builds a lightweight Graph Neural Network (GNN-style) model
    for noisy financial time series classification.

    Nodes represent features (indicators).
    Edges are learned implicitly via feature interactions.

    Designed for robustness to:
    - Non-stationarity
    - Variable-length sequences
    - Small batch sizes

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    hidden_dim: int = 32
    gnn_layers: int = 2
    dropout: float = 0.3

    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    # Temporal aggregation
    # (B, T, F) → (B, F)
    x = GlobalAveragePooling1D(name="temporal_pool")(inputs)

    # Treat features as nodes
    # (B, F) → (B, F, 1)
    x = Lambda(lambda t: tf.expand_dims(t, axis=-1))(x)

    # GNN layers
    for i in range(gnn_layers):
        x = GraphMessagePassing(
            hidden_dim=hidden_dim,
            dropout=dropout,
            name=f"gnn_layer_{i}"
        )(x)

    # Graph pooling
    x = Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)

    # Head
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="gnn_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[AUC(name="auc"), brier]
    )

    return model



# --------------------  Neural Anomaly Detector (NAD-lite)  --------------------



class AutoencoderClassifierLite(keras.Model):
    """
    Autoencoder + Classifier with internal reconstruction loss.

    Keras 3-safe implementation using subclassed Model.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        recon_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.recon_weight = recon_weight

        # -------- Pooling --------
        self.pool = GlobalAveragePooling1D()

        # -------- Encoder --------
        self.enc_dense = Dense(hidden_dim, activation="relu")
        self.enc_norm = LayerNormalization()
        self.enc_drop = Dropout(dropout)
        self.latent = Dense(latent_dim, activation="linear")

        # -------- Decoder --------
        self.dec_dense = Dense(hidden_dim, activation="relu")
        self.dec_drop = Dropout(dropout)
        self.reconstruction = Dense(n_features, activation="linear")

        # -------- Classifier --------
        self.cls_dense = Dense(32, activation="relu")
        self.cls_drop = Dropout(dropout)
        self.output_head = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # -------------------------
        # Pool input
        # -------------------------
        pooled = self.pool(inputs)

        # -------------------------
        # Encode
        # -------------------------
        x = self.enc_dense(pooled)
        x = self.enc_norm(x)
        x = self.enc_drop(x, training=training)

        latent = self.latent(x)

        # -------------------------
        # Decode (reconstruction)
        # -------------------------
        d = self.dec_dense(latent)
        d = self.dec_drop(d, training=training)
        recon = self.reconstruction(d)

        # -------------------------
        # Reconstruction loss
        # -------------------------
        diff = pooled - recon
        recon_loss = ops.mean(ops.square(diff))
        self.add_loss(self.recon_weight * recon_loss)

        # -------------------------
        # Classification
        # -------------------------
        c = self.cls_dense(latent)
        c = self.cls_drop(c, training=training)
        return self.output_head(c)



def build_autoencoder_classifier_lite(X_train_seq: Union[np.ndarray, list]) -> keras.Model:
    """
    Builds an Autoencoder + Classifier model for
    neural anomaly detection in time series.

    Fully compatible with Keras 3 and existing pipelines.

    Args:
        X_train_seq (array-like):
            Training sequences of shape (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """

    latent_dim: int = 16,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    recon_weight: float = 0.3

    n_features = X_train_seq.shape[2]

    model = AutoencoderClassifierLite(
        n_features=n_features,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        recon_weight=recon_weight,
        name="autoencoder_classifier_lite"
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[AUC(name="auc"), brier]
    )

    return model



# --------------------  CNN-GRU-lite --------------------



def build_cnn_gru_lite(X_train_seq: Union[np.ndarray, list]) -> Model:
    """
    Builds a lightweight CNN-GRU model for
    noisy financial time series classification.

    Combines shallow temporal convolutions for
    local pattern extraction with a compact GRU
    layer for sequence modeling.

    Designed to be robust to non-stationarity
    and overfitting.

    Args:
        X_train_seq (array-like):
            Training sequences of shape
            (n_samples, time_steps, n_features)

    Returns:
        Compiled Keras Model
    """
    n_features = X_train_seq.shape[2]

    inputs = Input(shape=(None, n_features))

    # CNN block
    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)
    x = LayerNormalization()(x)

    # GRU block
    x = GRU(
        units=32,
        dropout=0.3
    )(x)

    # Head
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="cnn_gru_lite")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            brier
        ]
    )

    return model