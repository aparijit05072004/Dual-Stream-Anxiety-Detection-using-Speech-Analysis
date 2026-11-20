"""End-to-end smoke tests on a synthetic RAVDESS-shaped corpus.

These run in seconds and need no download: they build a handful of tiny WAV
files with correct RAVDESS filenames (including the duplicate directory layout
the Kaggle mirror ships) and push them through indexing, splitting, the log-mel
front end, the network, and the anxiety mapping.

    pytest -q tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anxiety_speech.anxiety import anxiety_index, assess, band_for
from anxiety_speech.augment import add_awgn
from anxiety_speech.config import EMOTIONS, AudioConfig, ModelConfig
from anxiety_speech.data import build_index, split_by_speaker, split_stratified
from anxiety_speech.features import SpectrogramScaler, fit_length, mel_spectrogram

ACTORS = list(range(1, 13))
EMOTION_CODES = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    """A miniature RAVDESS song tree, duplicated exactly as the Kaggle ZIP is."""
    import soundfile as sf

    root = tmp_path_factory.mktemp("ravdess")
    rng = np.random.default_rng(0)
    sr, n = 8000, 8000
    for actor in ACTORS:
        for code in EMOTION_CODES:
            for intensity in ([1] if code == 1 else [1, 2]):
                name = f"03-02-{code:02d}-{intensity:02d}-01-01-{actor:02d}.wav"
                tone = 0.2 * np.sin(2 * np.pi * (90 + 25 * code) *
                                    np.arange(n) / sr).astype(np.float32)
                tone += 0.01 * rng.normal(size=n).astype(np.float32)
                for parent in (root / f"Actor_{actor:02d}",
                               root / "audio_song_actors_01-24" / f"Actor_{actor:02d}"):
                    parent.mkdir(parents=True, exist_ok=True)
                    sf.write(parent / name, tone, sr)
    return root


def test_build_index_deduplicates(corpus):
    unique = build_index(corpus)
    everything = build_index(corpus, deduplicate=False)
    assert len(everything) == 2 * len(unique), "fixture should contain exact duplicates"
    assert unique["filename"].is_unique
    assert len(unique) == len(ACTORS) * 11        # 1 neutral + 5 emotions x 2 intensities
    assert set(unique["emotion"]) == set(EMOTIONS.values())
    assert set(unique["gender"]) == {"male", "female"}


def test_speaker_split_shares_no_actor(corpus):
    train, val, test = split_by_speaker(build_index(corpus), seed=0)
    sets = [set(df["actor"]) for df in (train, val, test)]
    assert sets[0] & sets[1] == set() and sets[0] & sets[2] == set()
    assert sets[1] & sets[2] == set()
    assert len(train) + len(val) + len(test) == len(build_index(corpus))


def test_stratified_split_covers_every_row(corpus):
    frame = build_index(corpus)
    train, val, test = split_stratified(frame, seed=0)
    assert len(train) + len(val) + len(test) == len(frame)
    assert set(train["label"]) == set(frame["label"])


def test_spectrogram_shape_matches_config():
    cfg = AudioConfig()
    audio = fit_length(np.zeros(1000, dtype=np.float32), cfg.n_samples)
    mel = mel_spectrogram(audio, cfg)
    assert mel.shape == (cfg.n_mels, cfg.n_frames) == (128, 563)


def test_scaler_roundtrip():
    x = np.random.default_rng(1).normal(size=(8, 1, 16, 20)).astype(np.float32)
    scaler = SpectrogramScaler()
    z = scaler.fit_transform(x)
    assert z.shape == x.shape
    assert np.allclose(z.reshape(8, -1).mean(0), 0, atol=1e-5)
    restored = SpectrogramScaler().load_state_dict(scaler.state_dict())
    assert np.allclose(restored.transform(x), z)


def test_awgn_is_reproducible_and_bounded():
    sig = np.sin(np.linspace(0, 60, 4000)).astype(np.float32)
    a = add_awgn(sig, rng=np.random.default_rng(3))
    b = add_awgn(sig, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert a.shape == sig.shape and np.isfinite(a).all()


def test_anxiety_index_bounds_and_ordering():
    fearful = [0, 0, 0, 0, 0, 1.0]
    calm = [0, 1.0, 0, 0, 0, 0]
    assert anxiety_index(fearful) == pytest.approx(1.0)
    assert anxiety_index(calm) == pytest.approx(0.0)
    assert anxiety_index([0, 0, 0, 1.0, 0, 0]) < anxiety_index([0, 0, 0, 0, 1.0, 0])
    assert band_for(0.0) == "low" and band_for(1.0) == "high"
    with pytest.raises(ValueError):
        anxiety_index([0.5, 0.5])


def test_assess_reports_top_emotion():
    out = assess([0.05, 0.05, 0.05, 0.05, 0.1, 0.7])
    assert out.emotion == "fearful" and out.band in {"elevated", "high"}
    assert out.emotion_confidence == pytest.approx(0.7)
    assert sum(out.posterior.values()) == pytest.approx(1.0)


def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    from anxiety_speech.model import build_model

    cfg = AudioConfig()
    model = build_model(ModelConfig(), cfg)
    logits, posterior, attention = model(torch.randn(2, 1, cfg.n_mels, cfg.n_frames))
    assert logits.shape == (2, len(EMOTIONS))
    assert torch.allclose(posterior.sum(1), torch.ones(2), atol=1e-5)
    assert attention.shape[0] == 2 and attention.shape[1] == 1
    assert torch.allclose(attention.sum(-1), torch.ones(2, 1), atol=1e-5)


def test_model_parameter_count_matches_publication():
    pytest.importorskip("torch")
    from anxiety_speech.model import build_model

    # 8-class head, as trained in the published notebook run.
    model = build_model(ModelConfig(num_classes=8), AudioConfig())
    assert model.num_parameters == 263_593
