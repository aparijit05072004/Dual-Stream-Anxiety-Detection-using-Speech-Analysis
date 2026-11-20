"""Score one or more audio files with a trained checkpoint.

Example::

    python -m anxiety_speech.predict --checkpoint artifacts/best_model.pt clip.wav
    python -m anxiety_speech.predict -c artifacts/best_model.pt --json clips/*.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .anxiety import assess
from .features import batch_spectrograms, load_waveform


def predict_files(checkpoint: str | Path, paths: list[str], device: str = "cpu") -> list[dict]:
    """Return one assessment dict per input file."""
    from .evaluate import load_checkpoint

    model, scaler, audio_cfg = load_checkpoint(checkpoint, device)
    signals = [load_waveform(p, audio_cfg) for p in paths]
    x = scaler.transform(batch_spectrograms(signals, audio_cfg)[:, None])
    with torch.no_grad():
        _, posterior, attention = model(torch.from_numpy(x).float().to(device))
    posterior = posterior.cpu().numpy()

    results = []
    for path, p in zip(paths, posterior):
        a = assess(p)
        results.append({
            "file": str(path),
            "anxiety_index": round(a.index, 4),
            "band": a.band,
            "emotion": a.emotion,
            "confidence": round(a.emotion_confidence, 4),
            "posterior": {k: round(v, 4) for k, v in a.posterior.items()},
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", help="one or more .wav files")
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    results = predict_files(args.checkpoint, args.audio, args.device)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        width = max(len(Path(r["file"]).name) for r in results)
        print(f"{'file'.ljust(width)}  {'anxiety':>8}  {'band':<9}  emotion")
        for r in results:
            print(f"{Path(r['file']).name.ljust(width)}  {r['anxiety_index']:>8.3f}  "
                  f"{r['band']:<9}  {r['emotion']} ({r['confidence']:.1%})")
        print("\nResearch proxy over acted speech — not a clinical assessment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
