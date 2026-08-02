"""
Base model utilities for building classification heads and compilation.

Provides shared functionality for all architecture variants including
the classification head, layer freezing, compilation, and learning
rate scheduling.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None


def build_classification_head(
    base_model,
    num_classes: int = 2,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
) -> "tf.keras.Model":
    """
    Build classification head on top of a base feature extractor.

    Architecture: GAP -> Dense(256) -> Dropout(0.3) -> Softmax(2)

    Args:
        base_model: Pre-trained base model (feature extractor).
        num_classes: Number of output classes.
        dense_units: Number of units in the dense layer.
        dropout_rate: Dropout rate for regularisation.

    Returns:
        Complete Keras model with classification head.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for model building.")

    x = base_model.output

    # Global Average Pooling
    if len(x.shape) == 4:
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense layer
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_head",
    )(x)

    # Dropout
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_head")(x)

    # Output layer with softmax
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model


def freeze_base_layers(
    model: "tf.keras.Model",
    unfreeze_top: int = 20,
) -> "tf.keras.Model":
    """
    Freeze base layers and unfreeze the top N layers for fine-tuning.

    Args:
        model: Keras model with base and head layers.
        unfreeze_top: Number of top layers to leave trainable.

    Returns:
        Model with frozen/unfrozen layers configured.
    """
    if tf is None:
        raise ImportError("TensorFlow is required.")

    # Freeze all layers first
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze top N layers
    if unfreeze_top > 0:
        for layer in model.layers[-unfreeze_top:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    return model


def compile_model(
    model: "tf.keras.Model",
    learning_rate: float = 0.001,
    num_classes: int = 2,
) -> "tf.keras.Model":
    """
    Compile model with Adam optimiser and appropriate loss/metrics.

    Args:
        model: Keras model to compile.
        learning_rate: Initial learning rate for Adam.
        num_classes: Number of classes (for metric selection).

    Returns:
        Compiled Keras model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required.")

    optimiser = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )

    loss = "categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    if num_classes == 2:
        loss = "categorical_crossentropy"  # Using softmax output

    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=metrics,
    )

    return model


def get_cosine_annealing_schedule(
    initial_lr: float = 0.001,
    min_lr: float = 1e-6,
    total_epochs: int = 20,
    warmup_epochs: int = 2,
) -> "tf.keras.callbacks.LearningRateScheduler":
    """
    Create cosine annealing learning rate schedule with warmup.

    Args:
        initial_lr: Maximum learning rate after warmup.
        min_lr: Minimum learning rate at the end.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.

    Returns:
        LearningRateScheduler callback.
    """
    if tf is None:
        raise ImportError("TensorFlow is required.")

    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr + (initial_lr - min_lr) * cosine_decay

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)


def get_model_summary(model: "tf.keras.Model") -> str:
    """
    Get model summary as a string.

    Args:
        model: Keras model.

    Returns:
        String representation of model summary.
    """
    if tf is None:
        return "TensorFlow not available."

    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
