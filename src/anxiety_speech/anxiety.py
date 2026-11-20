"""Emotion posteriors -> a continuous anxiety index.

The classifier is trained on emotion labels because no public corpus of
clinically-labelled anxious speech exists at this scale. The index below maps
the emotion posterior onto the *high-arousal / negative-valence* region of the
affective circumplex, which is the region anxious speech occupies.

.. warning::
   This is a research proxy for screening-style research, **not** a diagnostic
   instrument. It is not validated against any clinical scale (GAD-7, STAI, …),
   it is derived from acted sung speech, and it must never be used to make a
   decision about a person. See ``docs/MODEL_CARD.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .config import ANXIETY_WEIGHTS, EMOTIONS

#: Index bands used for the human-readable label.
BANDS = ((0.20, "low"), (0.50, "moderate"), (0.75, "elevated"), (1.01, "high"))


@dataclass
class AnxietyAssessment:
    """One clip's result."""

    index: float
    band: str
    emotion: str
    emotion_confidence: float
    posterior: Dict[str, float]

    def __str__(self) -> str:
        top = ", ".join(f"{k} {v:.1%}" for k, v in
                        sorted(self.posterior.items(), key=lambda kv: -kv[1])[:3])
        return (f"anxiety index {self.index:.3f} ({self.band})  |  "
                f"emotion {self.emotion} ({self.emotion_confidence:.1%})  |  {top}")


def anxiety_index(posterior: Sequence[float],
                  weights: Mapping[int, float] = ANXIETY_WEIGHTS) -> float:
    """Weighted sum of the emotion posterior: ``Σ wᵢ · p(emotionᵢ)``.

    With the default weights the index lies in ``[0, 1]``: 0 for a confident
    neutral/calm/happy reading, 1 for a confident *fearful* reading.
    """
    p = np.asarray(posterior, dtype=np.float64)
    if p.ndim != 1 or p.size != len(weights):
        raise ValueError(f"expected a posterior of length {len(weights)}, got {p.shape}")
    w = np.array([weights[i] for i in range(len(weights))], dtype=np.float64)
    return float(np.clip((p * w).sum(), 0.0, 1.0))


def band_for(index: float) -> str:
    """Map the index onto a coarse band label."""
    for upper, name in BANDS:
        if index < upper:
            return name
    return BANDS[-1][1]


def assess(posterior: Sequence[float],
           weights: Mapping[int, float] = ANXIETY_WEIGHTS) -> AnxietyAssessment:
    """Full assessment for one clip's emotion posterior."""
    p = np.asarray(posterior, dtype=np.float64)
    idx = anxiety_index(p, weights)
    top = int(p.argmax())
    return AnxietyAssessment(
        index=idx,
        band=band_for(idx),
        emotion=EMOTIONS[top],
        emotion_confidence=float(p[top]),
        posterior={EMOTIONS[i]: float(p[i]) for i in range(len(EMOTIONS))},
    )
