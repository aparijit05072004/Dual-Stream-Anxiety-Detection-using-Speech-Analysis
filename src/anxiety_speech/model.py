"""The parallel CNN + BiLSTM-with-attention network.

Two views of the same log-mel spectrogram are computed side by side and fused:

* **Convolutional branch** - four ``Conv-BN-ReLU-MaxPool-Dropout`` blocks
  squeeze the (128 x 563) spectrogram down to a 256-d texture embedding that
  captures *local* spectro-temporal structure (formant shape, jitter, bursts).
* **Recurrent branch** - the spectrogram is pooled to 64 mel bands x 140 frames,
  read as a sequence by a bidirectional LSTM, and collapsed by additive
  attention into a 256-d embedding that captures *long-range* prosody
  (intonation contour, pacing, tremor).

The two embeddings are concatenated (512-d) and classified by a single linear
layer. 263,593 trainable parameters for the published 6-class configuration.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import AudioConfig, ModelConfig


class ParallelCNNBiLSTM(nn.Module):
    """Parallel convolutional / recurrent speech-emotion classifier.

    Args:
        model_cfg: channel widths, dropout rates, hidden size, class count.
        audio_cfg: used once at construction to infer the flattened conv size,
            so the network adapts to a different mel/frame geometry instead of
            failing with a shape error deep in ``forward``.
    """

    def __init__(self, model_cfg: ModelConfig | None = None,
                 audio_cfg: AudioConfig | None = None) -> None:
        super().__init__()
        self.cfg = model_cfg or ModelConfig()
        self.audio_cfg = audio_cfg or AudioConfig()
        c1, c2, c3, c4 = self.cfg.conv_channels
        p = self.cfg.conv_dropout

        self.conv = nn.Sequential(
            *self._block(1, c1, pool=2, dropout=p),
            *self._block(c1, c2, pool=4, dropout=p),
            *self._block(c2, c3, pool=4, dropout=p),
            *self._block(c3, c4, pool=4, dropout=p),
        )

        self.lstm_pool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        lstm_input = self.audio_cfg.n_mels // 2
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=self.cfg.lstm_hidden,
                            bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(self.cfg.lstm_dropout)
        self.attention = nn.Linear(2 * self.cfg.lstm_hidden, 1)

        conv_dim = self._infer_conv_dim()
        self.classifier = nn.Linear(conv_dim + 2 * self.cfg.lstm_hidden,
                                    self.cfg.num_classes)
        self.head_dropout = nn.Dropout(self.cfg.head_dropout)

    @staticmethod
    def _block(in_ch: int, out_ch: int, pool: int, dropout: float) -> list[nn.Module]:
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
            nn.Dropout(p=dropout),
        ]

    def _infer_conv_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.audio_cfg.n_mels, self.audio_cfg.n_frames)
            return int(self.conv(dummy).flatten(1).shape[1])

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: ``x`` of shape ``(B, 1, n_mels, n_frames)``.

        Returns:
            ``(logits, posterior, attention_weights)`` where the posterior is a
            softmax over classes and the attention weights are ``(B, 1, T)``.
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"expected (B, 1, mels, frames), got {tuple(x.shape)}")

        conv_embedding = self.conv(x).flatten(start_dim=1)

        seq = self.lstm_pool(x).squeeze(1).permute(0, 2, 1)   # (B, T, mels/2)
        lstm_out, _ = self.lstm(seq)                          # (B, T, 2H)
        lstm_out = self.lstm_dropout(lstm_out)

        # Additive attention over time. Equivalent to scoring each step in a
        # Python loop, as in the notebook, but evaluated as one matmul.
        scores = self.attention(lstm_out).transpose(1, 2)     # (B, 1, T)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, lstm_out).squeeze(1)     # (B, 2H)

        logits = self.classifier(torch.cat([conv_embedding, context], dim=1))
        logits = self.head_dropout(logits)
        return logits, torch.softmax(logits, dim=1), weights

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_cfg: ModelConfig | None = None,
                audio_cfg: AudioConfig | None = None,
                device: str | torch.device = "cpu") -> ParallelCNNBiLSTM:
    """Construct the network and move it to ``device``."""
    return ParallelCNNBiLSTM(model_cfg, audio_cfg).to(device)
