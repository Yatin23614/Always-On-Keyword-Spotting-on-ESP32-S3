"""Quantization helpers used for model calibration and firmware parity."""

import numpy as np


def quantize_int8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return np.clip(np.rint(np.asarray(values) / scale) + zero_point, -128, 127).astype(np.int8)


def dequantize_int8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return (np.asarray(values, dtype=np.float32) - zero_point) * scale
