"""Dataset indexing, de-duplication and splitting.

RAVDESS filenames encode everything we need::

    03-02-06-02-01-02-12.wav
    │  │  │  │  │  │  └── actor      12  (odd = male, even = female)
    │  │  │  │  │  └───── repetition  02
    │  │  │  │  └──────── statement   01 = "Kids are talking by the door"
    │  │  │  └─────────── intensity   01 = normal, 02 = strong
    │  │  └────────────── emotion     06 = fearful
    │  └───────────────── vocal chan. 02 = song
    └──────────────────── modality    03 = audio-only

.. warning::
   The Kaggle mirror ``uwrfkaggler/ravdess-emotional-song-audio`` ships **two
   copies of every recording** - once under ``Actor_XX/`` and once under
   ``audio_song_actors_01-24/Actor_XX/``. A naive ``os.walk`` therefore yields
   2,024 paths for 1,012 unique recordings, and a random split will place a
   file and its byte-identical twin on opposite sides of the split.
   :func:`build_index` de-duplicates on the filename stem by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import EMOTIONS, RAVDESS_CODE_TO_LABEL

LOGGER = logging.getLogger(__name__)

COLUMNS = ["path", "filename", "label", "emotion", "intensity", "statement",
           "repetition", "actor", "gender"]


def parse_filename(path: Path) -> Optional[dict]:
    """Parse one RAVDESS filename. Returns ``None`` if it is not a song clip."""
    parts = path.stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() for p in parts):
        return None
    modality, _channel, emotion, intensity, statement, repetition, actor = (
        int(p) for p in parts
    )
    if modality != 3:                      # audio-only
        return None
    if emotion not in RAVDESS_CODE_TO_LABEL:   # disgust/surprise: speech subset only
        return None
    label = RAVDESS_CODE_TO_LABEL[emotion]
    return {
        "path": str(path),
        "filename": path.name,
        "label": label,
        "emotion": EMOTIONS[label],
        "intensity": "normal" if intensity == 1 else "strong",
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "gender": "male" if actor % 2 else "female",
    }


def build_index(root: str | Path, deduplicate: bool = True) -> pd.DataFrame:
    """Walk ``root`` and return one row per recording.

    Args:
        root: directory containing ``Actor_XX`` folders (at any depth).
        deduplicate: drop repeated filename stems, keeping the shallowest path.
            Leave this on unless you are deliberately reproducing the
            as-published random-split run.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    records = [r for p in sorted(root.rglob("*.wav")) if (r := parse_filename(p))]
    if not records:
        raise RuntimeError(f"no RAVDESS song .wav files found under {root}")

    frame = pd.DataFrame.from_records(records, columns=COLUMNS)
    n_raw = len(frame)
    if deduplicate:
        frame["_depth"] = frame["path"].map(lambda p: p.count("/"))
        frame = (frame.sort_values(["filename", "_depth"])
                      .drop_duplicates("filename", keep="first")
                      .drop(columns="_depth")
                      .sort_values(["actor", "filename"])
                      .reset_index(drop=True))
        if len(frame) < n_raw:
            LOGGER.warning("de-duplicated %d archive entries down to %d unique "
                           "recordings", n_raw, len(frame))
    return frame


def split_by_speaker(
    frame: pd.DataFrame,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Speaker-independent split: no actor appears in more than one fold.

    This is the honest protocol for emotion recognition - a model that has
    heard a speaker during training can recognise *that speaker's* delivery
    rather than the emotion. Use this for any number you intend to report.
    """
    actors = np.array(sorted(frame["actor"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    n = len(actors)
    n_test = max(1, int(round(test_fraction * n)))
    n_val = max(1, int(round(val_fraction * n)))
    test_actors = set(actors[:n_test])
    val_actors = set(actors[n_test:n_test + n_val])

    is_test = frame["actor"].isin(test_actors)
    is_val = frame["actor"].isin(val_actors)
    train, val, test = frame[~is_test & ~is_val], frame[is_val], frame[is_test]
    LOGGER.info("speaker-independent split | train %d (%d actors) | val %d (%d) "
                "| test %d (%d)", len(train), train.actor.nunique(), len(val),
                val.actor.nunique(), len(test), test.actor.nunique())
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def split_stratified(
    frame: pd.DataFrame,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Class-stratified random split - the protocol used in the notebook.

    Actors are shared across folds, so scores from this split are optimistic
    relative to :func:`split_by_speaker`. Provided for reproducibility of the
    published run; prefer the speaker-independent split for new numbers.
    """
    rng = np.random.default_rng(seed)
    parts: List[List[pd.DataFrame]] = [[], [], []]
    for _, group in frame.groupby("label", sort=True):
        idx = rng.permutation(len(group))
        m = len(group)
        cut_train = int((1.0 - val_fraction - test_fraction) * m)
        cut_val = int((1.0 - test_fraction) * m)
        for bucket, sl in zip(parts, (idx[:cut_train], idx[cut_train:cut_val],
                                      idx[cut_val:])):
            bucket.append(group.iloc[sl])
    return tuple(pd.concat(p).reset_index(drop=True) for p in parts)  # type: ignore[return-value]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts per emotion x intensity, for the dataset card."""
    return (frame.pivot_table(index="emotion", columns="intensity",
                              values="filename", aggfunc="count", fill_value=0)
                 .reindex([EMOTIONS[i] for i in sorted(EMOTIONS)]))
