"""Evaluate a checkpoint: overall metrics, confusion matrix, subgroup breakdown.

Example::

    python -m anxiety_speech.evaluate --checkpoint artifacts/best_model.pt \
        --split-csv artifacts/split_test.csv --out artifacts/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .anxiety import anxiety_index
from .config import EMOTIONS, AudioConfig, ModelConfig
from .features import SpectrogramScaler, batch_spectrograms, load_many
from .model import build_model


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Restore model + fitted scaler + the audio config they were trained with."""
    blob = torch.load(path, map_location=device, weights_only=False)
    audio_cfg = AudioConfig(**blob["audio_cfg"])
    model_cfg = ModelConfig(**blob["model_cfg"])
    model = build_model(model_cfg, audio_cfg, device)
    model.load_state_dict(blob["model"])
    model.eval()
    scaler = SpectrogramScaler().load_state_dict(blob["scaler"])
    return model, scaler, audio_cfg


def subgroup_table(frame: pd.DataFrame, correct: np.ndarray, column: str) -> pd.DataFrame:
    """Accuracy broken down by a metadata column (gender, intensity, …)."""
    df = frame.assign(correct=correct)
    out = (df.groupby(column)["correct"]
             .agg(n="count", correct="sum")
             .assign(accuracy=lambda d: (100 * d["correct"] / d["n"]).round(2)))
    return out.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-csv", required=True,
                        help="CSV written by train.py, e.g. artifacts/split_test.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, recall_score)

    device = torch.device(args.device)
    model, scaler, audio_cfg = load_checkpoint(args.checkpoint, device)
    frame = pd.read_csv(args.split_csv)
    y_true = np.ascontiguousarray(frame["label"].to_numpy(dtype=np.int64))

    signals = load_many(frame["path"], audio_cfg, progress=True)
    x = scaler.transform(batch_spectrograms(signals, audio_cfg, progress=True)[:, None])

    loss_fn, losses, preds, probs = nn.CrossEntropyLoss(), [], [], []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            xb = torch.from_numpy(x[start:start + 32]).float().to(device)
            yb = torch.from_numpy(y_true[start:start + 32]).long().to(device)
            logits, posterior, _ = model(xb)
            losses.append(loss_fn(logits, yb).item() * len(xb))
            preds.append(posterior.argmax(1).cpu().numpy())
            probs.append(posterior.cpu().numpy())
    preds, probs = np.concatenate(preds), np.concatenate(probs)
    labels = sorted(EMOTIONS)
    names = [EMOTIONS[i] for i in labels]

    metrics = {
        "n": int(len(y_true)),
        "loss": float(np.sum(losses) / len(y_true)),
        "accuracy": round(100 * float((preds == y_true).mean()), 2),
        "recall_weighted": round(float(recall_score(y_true, preds, average="weighted",
                                                    zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, preds, average="weighted",
                                            zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, preds, average="macro",
                                         zero_division=0)), 4),
    }
    cm = confusion_matrix(y_true, preds, labels=labels)
    correct = (preds == y_true).astype(int)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv")
    frame.assign(predicted=[EMOTIONS[p] for p in preds], correct=correct,
                 anxiety_index=[round(anxiety_index(p), 4) for p in probs]
                 ).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + json.dumps(metrics, indent=2))
    print("\nper-class report\n" + classification_report(
        y_true, preds, labels=labels, target_names=names, zero_division=0))
    print("confusion matrix (rows = truth)\n" +
          pd.DataFrame(cm, index=names, columns=names).to_string())
    for column in ("gender", "intensity"):
        if column in frame:
            print(f"\naccuracy by {column}\n" +
                  subgroup_table(frame, correct, column).to_string(index=False))
    print(f"\nartifacts written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
