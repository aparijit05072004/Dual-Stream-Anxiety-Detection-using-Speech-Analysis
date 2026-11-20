# 📄 Publication

This repository is the reference implementation of the **speech stream** of
the dual-stream model published as **Chapter 1** of the proceedings below.

<div align="center">

### Dual-Stream Anxiety Detection: LSTM-Based Speech Analysis and CNN-Driven Sentiment Analysis

**Adrija Basu · Akanksha Mishra · Aditya Roy · Aparijit Chakraborty · R. Hariharan**

*Proceedings of the Third Congress on Control, Robotics, and Mechatronics*
Smart Innovation, Systems and Technologies · Springer Nature Singapore · 2026 · pp. 1–15

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--981--96--8126--6__1-0b5394?style=for-the-badge)](https://doi.org/10.1007/978-981-96-8126-6_1)
[![Springer](https://img.shields.io/badge/Read_on-SpringerLink-0b7261?style=for-the-badge&logo=springer&logoColor=white)](https://link.springer.com/chapter/10.1007/978-981-96-8126-6_1)

</div>

## Record

| Field | Value |
|---|---|
| **Chapter title** | Dual-Stream Anxiety Detection: LSTM-Based Speech Analysis and CNN-Driven Sentiment Analysis |
| **Authors** | Adrija Basu, Akanksha Mishra, Aditya Roy, Aparijit Chakraborty, R. Hariharan |
| **Book** | Proceedings of the Third Congress on Control, Robotics, and Mechatronics |
| **Series** | Smart Innovation, Systems and Technologies (ISSN 2190-3026) |
| **Publisher** | Springer Nature Singapore |
| **Year** | 2026 |
| **Pages** | 1–15 (Chapter 1) |
| **DOI** | [10.1007/978-981-96-8126-6_1](https://doi.org/10.1007/978-981-96-8126-6_1) |
| **ISBN (online)** | 978-981-96-8126-6 |
| **ISBN (print)** | 978-981-96-8125-9 |

## What the chapter covers, and what this repository implements

The chapter proposes a **dual-stream** approach: an acoustic stream over
speech and a lexical stream over text, combined into a single anxiety
assessment.

| Stream | Model | In this repository |
|---|---|---|
| 🎙️ **Speech** | parallel CNN + BiLSTM with attention over log-mel spectrograms | ✅ full implementation, training, evaluation and inference |
| 📝 **Text** | CNN-driven sentiment analysis over transcripts | ➖ companion work — not implemented here |

Everything in [`src/anxiety_speech`](../src/anxiety_speech) and
[`Speech_sentiment_analysis.ipynb`](../Speech_sentiment_analysis.ipynb)
concerns the speech stream. [`ARCHITECTURE.md`](ARCHITECTURE.md) shows where
the second stream would attach.

## Citing

### BibTeX

```bibtex
@inbook{Basu2026DualStream,
  title     = {Dual-Stream Anxiety Detection: LSTM-Based Speech Analysis
               and CNN-Driven Sentiment Analysis},
  author    = {Basu, Adrija and Mishra, Akanksha and Roy, Aditya
               and Chakraborty, Aparijit and Hariharan, R.},
  booktitle = {Proceedings of the Third Congress on Control, Robotics,
               and Mechatronics},
  series    = {Smart Innovation, Systems and Technologies},
  publisher = {Springer Nature Singapore},
  year      = {2026},
  pages     = {1--15},
  isbn      = {9789819681266},
  issn      = {2190-3026},
  doi       = {10.1007/978-981-96-8126-6_1},
  url       = {https://doi.org/10.1007/978-981-96-8126-6_1}
}
```

### APA

> Basu, A., Mishra, A., Roy, A., Chakraborty, A., & Hariharan, R. (2026).
> Dual-stream anxiety detection: LSTM-based speech analysis and CNN-driven
> sentiment analysis. In *Proceedings of the Third Congress on Control,
> Robotics, and Mechatronics* (pp. 1–15). Smart Innovation, Systems and
> Technologies. Springer Nature Singapore.
> https://doi.org/10.1007/978-981-96-8126-6_1

### IEEE

> A. Basu, A. Mishra, A. Roy, A. Chakraborty, and R. Hariharan,
> "Dual-stream anxiety detection: LSTM-based speech analysis and CNN-driven
> sentiment analysis," in *Proc. 3rd Congr. Control, Robotics, and
> Mechatronics*, Smart Innovation, Systems and Technologies, Springer Nature
> Singapore, 2026, pp. 1–15, doi: 10.1007/978-981-96-8126-6_1.

A machine-readable [`CITATION.cff`](../CITATION.cff) sits at the repository
root — GitHub renders a *Cite this repository* button from it.

## Dataset citation

The corpus has its own citation requirement:

```bibtex
@article{livingstone2018ravdess,
  title   = {The Ryerson Audio-Visual Database of Emotional Speech and Song
             (RAVDESS): A dynamic, multimodal set of facial and vocal
             expressions in North American English},
  author  = {Livingstone, Steven R. and Russo, Frank A.},
  journal = {PLoS ONE},
  volume  = {13}, number = {5}, pages = {e0196391}, year = {2018},
  doi     = {10.1371/journal.pone.0196391}
}
```
