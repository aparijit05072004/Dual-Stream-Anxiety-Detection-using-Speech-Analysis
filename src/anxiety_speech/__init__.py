"""Anxiety detection from speech — the speech stream of the dual-stream model.

Reference implementation of the speech pathway described in:

    Basu, A., Mishra, A., Roy, A., Chakraborty, A., & Hariharan, R. (2026).
    "Dual-Stream Anxiety Detection: LSTM-Based Speech Analysis and
    CNN-Driven Sentiment Analysis." In *Proceedings of the Third Congress on
    Control, Robotics, and Mechatronics*, Smart Innovation, Systems and
    Technologies, pp. 1-15. Springer Nature Singapore.
    https://doi.org/10.1007/978-981-96-8126-6_1

The package refactors ``Speech_sentiment_analysis.ipynb`` into an importable,
CLI-driven pipeline and adds a leakage-safe data protocol (de-duplication and
speaker-independent splitting). See ``docs/METHODOLOGY.md``.
"""

__version__ = "1.0.0"
__all__ = ["config", "data", "features", "augment", "model", "anxiety"]
