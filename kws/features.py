"""Numpy-only feature extraction shared by training and parity tests."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    # Must match firmware/main/config.h and scripts/train_jarvis.py.
    sample_rate: int = 10_000
    window_samples: int = 28_000
    frame_samples: int = 300
    hop_samples: int = 200
    mel_bins: int = 40
    fft_size: int = 512
    fmin: float = 60.0
    fmax: float = 4_800.0

    @property
    def frame_count(self) -> int:
        return 1 + (self.window_samples - self.frame_samples) // self.hop_samples


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(config: FeatureConfig) -> np.ndarray:
    bins = np.floor((config.fft_size + 1) * _mel_to_hz(np.linspace(
        _hz_to_mel(config.fmin), _hz_to_mel(config.fmax), config.mel_bins + 2
    )) / config.sample_rate).astype(int)
    bank = np.zeros((config.mel_bins, config.fft_size // 2 + 1), dtype=np.float32)
    for band in range(config.mel_bins):
        left, center, right = bins[band:band + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for index in range(left, min(center, bank.shape[1])):
            bank[band, index] = (index - left) / (center - left)
        for index in range(center, min(right, bank.shape[1])):
            bank[band, index] = (right - index) / (right - center)
    return bank


def extract_log_mel(samples: np.ndarray, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Return [frames, mel_bins, 1] float32 features for exactly one audio window."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size < config.window_samples:
        samples = np.pad(samples, (0, config.window_samples - samples.size))
    samples = samples[:config.window_samples]
    frames = np.stack([
        samples[start:start + config.frame_samples]
        for start in range(0, config.window_samples - config.frame_samples + 1, config.hop_samples)
    ])
    frames *= np.hamming(config.frame_samples).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames, n=config.fft_size)) ** 2
    energies = spectrum @ _mel_filterbank(config).T
    features = np.log(np.maximum(energies, 1e-10)).astype(np.float32)
    features -= features.mean()
    features /= max(float(features.std()), 1e-6)
    return features[..., None]
