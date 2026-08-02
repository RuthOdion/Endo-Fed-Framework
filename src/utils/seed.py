"""
Reproducibility and GPU configuration utilities.

Ensures deterministic results across experiments by setting seeds
for all random number generators and configuring GPU memory growth.
"""

import os
import random

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if tf is not None:
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def set_gpu_config(memory_growth: bool = True, gpu_index: int = 0) -> None:
    """
    Configure GPU settings for TensorFlow.

    Designed for Amazon SageMaker ml.g4dn.xlarge instances with NVIDIA T4 GPU (16GB).
    Enables memory growth to prevent TensorFlow from allocating all GPU memory at once.

    Args:
        memory_growth: Whether to enable memory growth. Default True.
        gpu_index: Index of the GPU to use. Default 0 (single GPU on ml.g4dn.xlarge).

    Notes:
        - SageMaker ml.g4dn.xlarge: 1x NVIDIA T4 (16GB), 4 vCPUs, 16GB RAM
        - SageMaker ml.g4dn.2xlarge: 1x NVIDIA T4 (16GB), 8 vCPUs, 32GB RAM
        - SageMaker ml.g4dn.4xlarge: 1x NVIDIA T4 (16GB), 16 vCPUs, 64GB RAM
        - For multi-GPU instances (ml.g4dn.12xlarge), adjust gpu_index accordingly.
    """
    if tf is None:
        print("WARNING: TensorFlow not available. GPU configuration skipped.")
        return

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("No GPUs detected. Running on CPU.")
        return

    try:
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        if gpu_index < len(gpus):
            tf.config.set_visible_devices(gpus[gpu_index], "GPU")
            print(f"Using GPU {gpu_index}: {gpus[gpu_index].name}")
        else:
            print(f"GPU index {gpu_index} not available. Using GPU 0.")
            tf.config.set_visible_devices(gpus[0], "GPU")

        print(f"Total GPUs available: {len(gpus)}")

    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("GPU settings must be configured before TensorFlow initialisation.")
