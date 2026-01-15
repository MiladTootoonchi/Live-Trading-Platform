# Ignoring info + warning + errors: the user do not need to see this
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Completely disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Reduces backend logs

import tensorflow as tf

from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
)

from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="LiveTrader")
class Patchify(tf.keras.layers.Layer):
    """
    Splits a temporal sequence into non-overlapping patches.

    This layer converts a time-series tensor of shape
    `(batch_size, time_steps, n_features)` into a patch-based
    representation suitable for Transformer-style models such as
    PatchTST.

    If the number of time steps is not divisible by `patch_len`,
    the sequence is zero-padded along the time dimension to ensure
    an integer number of patches.
    """

    def __init__(self, patch_len: int, **kwargs):
        """
        Initializes the Patchify layer.

        Args:
            patch_len (int):
                Length of each temporal patch (number of time steps
                per patch).
            **kwargs:
                Additional keyword arguments passed to the base
                `tf.keras.layers.Layer` class.
        """
        super().__init__(**kwargs)
        self.patch_len = patch_len

    def call(self, inputs, training: bool = False):
        """
        Applies patchification to the input time series.

        Args:
            inputs (tf.Tensor):
                Input tensor of shape
                `(batch_size, time_steps, n_features)`.
            training (bool, optional):
                Whether the layer is being executed in training mode.
                This argument is included for API consistency and does
                not alter behavior. Defaults to False.

        Returns:
            tf.Tensor:
                Patchified tensor of shape
                `(batch_size, n_patches, patch_len * n_features)`,
                where `n_patches = ceil(time_steps / patch_len)`.
        """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        n_features = tf.shape(inputs)[2]

        # Pad sequence to ensure full patches
        pad_len = tf.maximum(0, self.patch_len - time_steps)
        x = tf.pad(inputs, [[0, 0], [0, pad_len], [0, 0]])

        # Recompute after padding
        time_steps = tf.shape(x)[1]
        n_patches = time_steps // self.patch_len

        # Truncate and reshape into patches
        x = x[:, :n_patches * self.patch_len, :]
        x = tf.reshape(
            x,
            (batch_size, n_patches, self.patch_len * n_features)
        )

        return x

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict:
                Serializable configuration dictionary containing
                the patch length.
        """
        config = super().get_config()
        config.update({
            "patch_len": self.patch_len,
        })
        return config



@register_keras_serializable(package="LiveTrader")
class GraphMessagePassing(tf.keras.layers.Layer):
    """
    Lightweight graph message-passing layer with learned adjacency.

    Each feature is treated as a node, and edges are learned dynamically
    via a dense projection. The layer aggregates messages across nodes
    and applies normalization and dropout for stability.

    Designed for:
    - Small graphs (feature-level nodes)
    - Noisy financial indicators
    - End-to-end differentiable adjacency learning
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Initializes the graph message-passing layer.

        Args:
            hidden_dim (int):
                Dimensionality of the node embeddings after message passing.
            dropout (float, optional):
                Dropout rate applied after normalization.
                Defaults to 0.0.
            **kwargs:
                Additional keyword arguments passed to the base Layer.
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def build(self, input_shape):
        """
        Creates the layer weights based on input shape.

        Args:
            input_shape (tuple):
                Shape of the input tensor.
                Expected: (batch_size, n_nodes, node_dim)
        """
        n_nodes = input_shape[1]

        self.adjacency = Dense(
            units=n_nodes,
            activation="tanh",
            name="learned_adjacency"
        )
        self.node_update = Dense(
            units=self.hidden_dim,
            activation="relu",
            name="node_update"
        )
        self.norm = LayerNormalization(name="node_norm")
        self.drop = Dropout(self.dropout, name="node_dropout")

        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        """
        Applies graph message passing to the input nodes.

        Args:
            inputs (tf.Tensor):
                Node feature tensor of shape (batch_size, n_nodes, node_dim).
            training (bool, optional):
                Whether the layer is being run in training mode.
                Controls dropout behavior. Defaults to False.

        Returns:
            tf.Tensor:
                Updated node embeddings of shape
                (batch_size, n_nodes, hidden_dim).
        """
        # Learn adjacency matrix
        adjacency = self.adjacency(inputs)          # (B, N, N)

        # Aggregate messages
        messages = tf.matmul(adjacency, inputs)     # (B, N, D)

        # Node update
        x = self.node_update(messages)
        x = self.norm(x)

        return self.drop(x, training=training)

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict:
                Serializable configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        })
        return config



@register_keras_serializable(package="LiveTrader")
class ExpandDims(tf.keras.layers.Layer):
    """
    Layer wrapper around `tf.expand_dims` for safe model serialization.

    Used to replace Lambda layers in functional models to ensure
    compatibility with Keras model saving and loading.
    """

    def __init__(self, axis: int = -1, **kwargs):
        """
        Initializes the ExpandDims layer.

        Args:
            axis (int, optional):
                Axis along which to insert a new dimension.
                Defaults to -1.
            **kwargs:
                Additional keyword arguments passed to the base Layer.
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """
        Expands the input tensor along the specified axis.

        Args:
            inputs (tf.Tensor):
                Input tensor.

        Returns:
            tf.Tensor:
                Tensor with an added dimension.
        """
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
    



@register_keras_serializable(package="LiveTrader")
class AutoencoderClassifierLite(tf.keras.Model):
    """
    Autoencoder + Classifier with internal reconstruction loss.

    Keras 3-safe implementation using subclassed Model.
    """

    expects_sequence = True

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

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
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
        # -------- Pool --------
        pooled = self.pool(inputs)

        # -------- Encoder --------
        x = self.enc_dense(pooled)
        x = self.enc_norm(x)
        x = self.enc_drop(x, training=training)
        z = self.latent(x)

        # -------- Decoder --------
        d = self.dec_dense(z)
        d = self.dec_drop(d, training=training)
        recon = self.reconstruction(d)

        # -------- Classifier --------
        c = self.cls_dense(z)
        c = self.cls_drop(c, training=training)
        prob = self.output_head(c)

        # -------- Reconstruction loss --------
        if training:
            recon_loss = tf.reduce_mean(tf.square(recon - pooled))
            self.add_loss(self.recon_weight * recon_loss)

        return prob
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "recon_weight": self.recon_weight,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

