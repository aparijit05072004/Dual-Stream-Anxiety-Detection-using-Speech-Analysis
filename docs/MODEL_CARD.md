# 🧾 Model card

> Following the *Model Cards for Model Reporting* convention
> (Mitchell et al., 2019).

## Model details

| | |
|---|---|
| **Name** | Parallel CNN + BiLSTM-with-attention speech-emotion classifier |
| **Version** | 1.0.0 |
| **Type** | Supervised multi-class classifier over log-mel spectrograms |
| **Parameters** | 263,593 trainable (8-class head, as published) |
| **Framework** | PyTorch |
| **Input** | 3 s of 48 kHz mono audio → 128 × 563 log-mel spectrogram (dB) |
| **Output** | Posterior over 6 emotions, plus a derived anxiety index in [0, 1] |
| **Training data** | RAVDESS Emotional Song Audio (CC BY-NC-SA 4.0) |
| **Published in** | [Springer, SIST, 2026 — DOI 10.1007/978-981-96-8126-6_1](https://doi.org/10.1007/978-981-96-8126-6_1) |
| **Licence of the data** | Non-commercial; any derived model inherits that restriction |

## Intended use

**Intended.** Academic research into acoustic correlates of affect; a
reproducible baseline for speech-emotion recognition on RAVDESS; teaching
material for parallel CNN/RNN architectures and attention over spectrograms;
a component in further research on multimodal affect detection.

**Not intended.** Any use that affects a person. Specifically out of scope:

- 🚫 **Clinical diagnosis or screening.** The model has never been validated
  against a clinical instrument (GAD-7, STAI, HAM-A, or any other). An
  "elevated" index is not a finding about anybody's mental health.
- 🚫 **Employment, admissions, insurance, lending or law-enforcement
  decisions.** Inferring emotional state from voice to make a consequential
  decision about a person is the paradigm case this model must not serve.
- 🚫 **Covert monitoring.** Emotion inference on people who have not given
  informed, specific, revocable consent.
- 🚫 **Deployment on speech unlike the training data** — spontaneous speech,
  other languages or accents, telephone-band or noisy-channel audio, children
  — without new validation on that data.

## The anxiety index is a proxy

The classifier is trained on **acted emotion labels**. No participant in
RAVDESS was assessed for anxiety, and the corpus contains no anxiety labels
at all. The index is a weighted projection of the emotion posterior onto the
high-arousal / negative-valence region of the affective circumplex:

$$\text{anxiety index} = 1.0\,p(\text{fearful}) + 0.8\,p(\text{angry}) + 0.5\,p(\text{sad})$$

Those weights are a **stated modelling hypothesis**, chosen on theoretical
grounds, not estimated from data and not validated against any ground truth
for anxiety. They are configurable in `config.ANXIETY_WEIGHTS` and should be
re-derived and re-justified for any new context. Treat the index as a
research-grade indicator of *affective tone*, nothing more.

## Factors

| Factor | Coverage in training data |
|---|---|
| Speaker gender | 12 male / 11 female actors — balanced |
| Emotion intensity | normal (552) and strong (460) — near balanced |
| Language / accent | North-American English only |
| Age | adult professional actors only |
| Lexical content | two fixed sentences |
| Channel | studio-quality, 48 kHz, single microphone |
| Speech style | **sung**, not spoken |

## Metrics

Accuracy, weighted recall, weighted F1, macro F1, confusion matrix, and
accuracy by speaker gender and emotion intensity. Macro F1 is reported
because *neutral* has half the support of every other class.

## Evaluation data and results

See [`RESULTS.md`](RESULTS.md) for the full discussion. Summary: **95.96 %**
test accuracy, 0.96 weighted F1, on a class-stratified random split of a
**duplicated** archive — a protocol that shares both recordings and speakers
between folds. That figure is an upper bound, not an estimate of
speaker-independent performance.

## Limitations

1. **Acted affect ≠ felt affect.** Performed emotion is more prototypical and
   more separable than spontaneous emotion. Accuracy on acted corpora
   routinely fails to transfer.
2. **Sung speech.** Melody constrains pitch and rhythm — the exact prosodic
   channels anxiety would modulate.
3. **Evaluation leak.** Duplicated archive plus a non-speaker-independent
   split, as above.
4. **Two classes never occur.** The published 8-class head includes *disgust*
   and *surprise*, which do not exist in the song subset; they can never be
   predicted. The refactored package uses a 6-class head.
5. **Tiny evaluation fold.** 198 clips: one error moves accuracy by 0.5 %.
6. **Fixed 3 s window.** Anything expressed outside that window is invisible.
7. **No calibration.** Softmax outputs are not calibrated probabilities, so
   the anxiety index should not be read as a likelihood.
8. **Demographic narrowness.** 23 North-American actors; nothing here
   supports claims about other populations.

## Ethical considerations

Speech-emotion inference is a **dual-use** technology. The same model that
supports research into affective computing enables covert emotional
surveillance. Voice also leaks identity, gender, age, health and emotional
state — audio is biometric data under GDPR Article 9 and comparable regimes,
and requires a lawful basis and explicit consent.

If you build on this work:

- obtain **informed, specific, revocable consent** from every person recorded;
- keep a human in the loop for anything consequential, and let subjects
  contest an output;
- publish **subgroup** metrics, not just an aggregate;
- state plainly that the output is a proxy signal, never a diagnosis;
- re-validate on data that matches your deployment before trusting anything.

## Caveats and recommendations

The right next steps for this model are a speaker-independent re-run, a
cross-corpus test (CREMA-D, TESS, SAVEE, IEMOCAP), calibration, and — before
any claim about anxiety specifically — evaluation against a validated
clinical instrument on consented, spontaneous speech.

## Citation

See [`PUBLICATION.md`](PUBLICATION.md) or [`CITATION.cff`](../CITATION.cff).
