"""Waveform augmentation — additive white Gaussian noise at a sampled SNR."""

from __future__ import annotations

import numpy as np


def add_awgn(
    signal: np.ndarray,
    num_bits: int = 16,
    snr_low: int = 15,
    snr_high: int = 30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix in white Gaussian noise at a uniformly sampled SNR in dB.

    The scaling factor ``K`` is solved so the mixture hits the target SNR::

        K = sqrt( (P_signal / P_noise) * 10^(-SNR/10) )

    Args:
        signal: 1-D waveform.
        num_bits: bit depth used to normalise powers (16-bit PCM by default).
        snr_low, snr_high: inclusive-exclusive dB range to sample from.
        rng: optional generator, for reproducible augmentation.
    """
    rng = rng or np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64)
    norm = 2.0 ** (num_bits - 1)

    signal_power = np.sum((signal / norm) ** 2) / signal.size
    noise = rng.normal(size=signal.size)
    noise_power = np.sum((noise / norm) ** 2) / noise.size
    if noise_power == 0 or signal_power == 0:
        return signal.astype(np.float32)

    target_snr = rng.integers(snr_low, snr_high)
    k = np.sqrt((signal_power / noise_power) * 10 ** (-target_snr / 10))
    return (signal + k * noise).astype(np.float32)
