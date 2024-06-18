import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid function for the input array.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid activations on input.
    """
    return 1 / (1 + np.exp(-x))
