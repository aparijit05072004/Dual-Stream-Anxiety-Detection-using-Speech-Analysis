"""Log-mel front end: waveform -> fixed-size (n_mels x n_frames) dB spectrogram."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import AudioConfig


def load_waveform(path: str | Path, cfg: AudioConfig) -> np.ndarray:
    """Load one clip and pad/trim it to exactly ``cfg.n_samples``.

    A fixed 3 s window taken 0.5 s in skips the leading silence common to
    RAVDESS takes while keeping every tensor the same shape.
    """
    import librosa  # imported lazily: keeps `import anxiety_speech` cheap

    audio, _ = librosa.load(str(path), sr=cfg.sample_rate,
                            duration=cfg.duration, offset=cfg.offset)
    return fit_length(audio, cfg.n_samples)


def fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Zero-pad on the right, or trim, to ``n_samples``."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < n_samples:
        return np.pad(audio, (0, n_samples - audio.size), mode="constant")
    return audio[:n_samples]


def mel_spectrogram(audio: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Power mel spectrogram in dB, referenced to the clip maximum."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        window=cfg.window,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmax=cfg.fmax,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def batch_spectrograms(signals: Sequence[np.ndarray], cfg: AudioConfig,
                       progress: bool = False) -> np.ndarray:
    """Stack log-mel spectrograms for a sequence of waveforms -> (N, mels, frames)."""
    out = np.empty((len(signals), cfg.n_mels, cfg.n_frames), dtype=np.float32)
    for i, sig in enumerate(signals):
        out[i] = mel_spectrogram(sig, cfg)
        if progress and (i + 1) % 100 == 0:
            print(f"\r  {i + 1}/{len(signals)} spectrograms", end="", flush=True)
    if progress:
        print(f"\r  {len(signals)}/{len(signals)} spectrograms")
    return out


def load_many(paths: Iterable[str | Path], cfg: AudioConfig,
              progress: bool = False) -> np.ndarray:
    """Load and length-normalise many clips -> (N, n_samples)."""
    paths = list(paths)
    out = np.empty((len(paths), cfg.n_samples), dtype=np.float32)
    for i, p in enumerate(paths):
        out[i] = load_waveform(p, cfg)
        if progress and (i + 1) % 100 == 0:
            print(f"\r  {i + 1}/{len(paths)} clips", end="", flush=True)
    if progress:
        print(f"\r  {len(paths)}/{len(paths)} clips")
    return out


class SpectrogramScaler:
    """Per-feature standardisation, fitted on the training fold only.

    Spectrograms are flattened to ``(N, mels*frames)``, standardised, and
    reshaped back - identical to the notebook's ``StandardScaler`` usage, but
    serialisable so inference reproduces training-time normalisation.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "SpectrogramScaler":
        flat = x.reshape(x.shape[0], -1)
        self.mean_ = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SpectrogramScaler.fit must be called first")
        shape = x.shape
        flat = (x.reshape(shape[0], -1) - self.mean_) / self.scale_
        return flat.reshape(shape).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "scale": self.scale_}

    def load_state_dict(self, state: dict) -> "SpectrogramScaler":
        self.mean_, self.scale_ = state["mean"], state["scale"]
        return self
