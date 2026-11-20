"""Central configuration — every magic number from the notebook, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

# RAVDESS emotion codes (position 3 of the filename) -> label id used by the model.
# The *song* subset of RAVDESS contains six of the eight RAVDESS emotions:
# disgust (07) and surprise (08) are speech-only and never appear here.
RAVDESS_CODE_TO_LABEL: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
EMOTIONS: Dict[int, str] = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
}

#: Weight each emotion contributes to the anxiety index (see ``anxiety.py``).
#: High-arousal / negative-valence states carry the mass. This is a research
#: proxy over the affective circumplex, **not** a clinical instrument.
ANXIETY_WEIGHTS: Dict[int, float] = {
    0: 0.0,   # neutral
    1: 0.0,   # calm
    2: 0.0,   # happy
    3: 0.5,   # sad
    4: 0.8,   # angry
    5: 1.0,   # fearful
}


@dataclass(frozen=True)
class AudioConfig:
    """Waveform loading and log-mel front end (values as published)."""

    sample_rate: int = 48_000
    duration: float = 3.0
    offset: float = 0.5
    n_fft: int = 1024
    win_length: int = 512
    hop_length: int = 256
    n_mels: int = 128
    window: str = "hamming"

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration)

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.hop_length + 1

    @property
    def fmax(self) -> float:
        return self.sample_rate / 2


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation schedule (values as published)."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.8
    weight_decay: float = 1e-3
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    seed: int = 42
    augment: bool = True
    snr_low: int = 15
    snr_high: int = 30


@dataclass(frozen=True)
class ModelConfig:
    """Parallel CNN + BiLSTM-with-attention network."""

    num_classes: int = len(EMOTIONS)
    conv_channels: tuple = (16, 32, 64, 64)
    conv_dropout: float = 0.3
    lstm_hidden: int = 128
    lstm_dropout: float = 0.1
    head_dropout: float = 0.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
