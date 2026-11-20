"""Train the parallel CNN + BiLSTM network.

Example::

    python -m anxiety_speech.train --data-root data/ravdess --split speaker \
        --epochs 100 --out artifacts/

The ``--split`` flag chooses the protocol:

``speaker``
    Speaker-independent (default). No actor is shared between folds - the
    honest protocol, and the one to use for any number you report.
``stratified``
    Class-stratified random split, reproducing the notebook run. Actors are
    shared across folds, so scores are optimistic.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .augment import add_awgn
from .config import EMOTIONS, AudioConfig, ModelConfig, TrainConfig
from .data import build_index, split_by_speaker, split_stratified, summarise
from .features import SpectrogramScaler, batch_spectrograms, load_many
from .model import build_model

LOGGER = logging.getLogger("anxiety_speech.train")


def prepare_fold(frame, audio_cfg: AudioConfig, label: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"[{label}] loading {len(frame)} clips")
    signals = load_many(frame["path"], audio_cfg, progress=True)
    return signals, np.ascontiguousarray(frame["label"].to_numpy(dtype=np.int64))


def augment_fold(signals: np.ndarray, labels: np.ndarray, cfg: TrainConfig
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Double the training fold with an AWGN copy of every clip."""
    rng = np.random.default_rng(cfg.seed)
    noisy = np.stack([add_awgn(s, snr_low=cfg.snr_low, snr_high=cfg.snr_high, rng=rng)
                      for s in signals])
    return np.concatenate([signals, noisy]), np.concatenate([labels, labels])


def run_epoch(model, x, y, loss_fn, optimizer, batch_size, device, rng):
    """One training pass. Returns (mean loss, accuracy %)."""
    model.train()
    order = rng.permutation(len(x))
    total_loss = total_correct = 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        xb = torch.from_numpy(x[idx]).float().to(device)
        yb = torch.from_numpy(y[idx]).long().to(device)
        logits, posterior, _ = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * len(idx)
        total_correct += int((posterior.argmax(1) == yb).sum())
    return total_loss / len(order), 100.0 * total_correct / len(order)


@torch.no_grad()
def evaluate_fold(model, x, y, loss_fn, device, batch_size=32):
    """Returns (mean loss, accuracy %, predictions, posteriors)."""
    model.eval()
    losses, preds, probs = [], [], []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).float().to(device)
        yb = torch.from_numpy(y[start:start + batch_size]).long().to(device)
        logits, posterior, _ = model(xb)
        losses.append(loss_fn(logits, yb).item() * len(xb))
        preds.append(posterior.argmax(1).cpu().numpy())
        probs.append(posterior.cpu().numpy())
    preds = np.concatenate(preds)
    return (float(np.sum(losses) / len(x)), 100.0 * float((preds == y).mean()),
            preds, np.concatenate(probs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="folder holding Actor_XX/ directories")
    parser.add_argument("--out", default="artifacts", help="where to write checkpoint + history")
    parser.add_argument("--split", choices=("speaker", "stratified"), default="speaker")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--no-augment", action="store_true", help="skip the AWGN copy")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="do NOT de-duplicate the archive (reproduces the leaky run)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    audio_cfg, model_cfg = AudioConfig(), ModelConfig()
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      learning_rate=args.lr, seed=args.seed,
                      augment=not args.no_augment)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = build_index(args.data_root, deduplicate=not args.keep_duplicates)
    print(f"\nindexed {len(frame)} recordings from {frame.actor.nunique()} actors")
    print(summarise(frame).to_string(), "\n")

    splitter = split_by_speaker if args.split == "speaker" else split_stratified
    train_df, val_df, test_df = splitter(frame, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        df.to_csv(out_dir / f"split_{name}.csv", index=False)

    x_train, y_train = prepare_fold(train_df, audio_cfg, "train")
    if cfg.augment:
        x_train, y_train = augment_fold(x_train, y_train, cfg)
        print(f"[train] augmented to {len(x_train)} clips")
    x_val, y_val = prepare_fold(val_df, audio_cfg, "val")
    x_test, y_test = prepare_fold(test_df, audio_cfg, "test")

    print("\ncomputing log-mel spectrograms")
    x_train = batch_spectrograms(x_train, audio_cfg, progress=True)[:, None]
    x_val = batch_spectrograms(x_val, audio_cfg, progress=True)[:, None]
    x_test = batch_spectrograms(x_test, audio_cfg, progress=True)[:, None]

    scaler = SpectrogramScaler().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(a) for a in (x_train, x_val, x_test))

    device = torch.device(args.device)
    model = build_model(model_cfg, audio_cfg, device)
    print(f"\nmodel on {device} | {model.num_parameters:,} trainable parameters")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay, momentum=cfg.momentum)

    history, best_val, best_epoch = [], -1.0, -1
    started = time.time()
    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = run_epoch(model, x_train, y_train, loss_fn, optimizer,
                                    cfg.batch_size, device, rng)
        va_loss, va_acc, _, _ = evaluate_fold(model, x_val, y_val, loss_fn, device,
                                              cfg.batch_size)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        print(f"epoch {epoch:3d} | loss {tr_loss:.4f} acc {tr_acc:5.2f}% | "
              f"val_loss {va_loss:.4f} val_acc {va_acc:5.2f}%")
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                        "audio_cfg": audio_cfg.__dict__, "model_cfg": model_cfg.__dict__,
                        "epoch": epoch, "val_acc": va_acc, "emotions": EMOTIONS},
                       out_dir / "best_model.pt")

    te_loss, te_acc, preds, _ = evaluate_fold(model, x_test, y_test, loss_fn, device,
                                              cfg.batch_size)
    summary = {
        "split": args.split,
        "deduplicated": not args.keep_duplicates,
        "epochs": cfg.epochs,
        "minutes": round((time.time() - started) / 60, 2),
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "final_test_loss": te_loss,
        "final_test_acc": te_acc,
        "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\ncheckpoint + history written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
