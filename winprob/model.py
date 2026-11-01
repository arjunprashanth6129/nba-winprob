"""The win-probability network (PyTorch, used for training)."""
import numpy as np
import torch
from torch import nn

from .features import FEATURES


class WinProbNet(nn.Module):
    """MLP: game state -> logit of P(home team wins)."""

    def __init__(self, n_in: int = len(FEATURES), hidden=(128, 64, 32), dropout=0.05):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def export_weights(checkpoint: dict, path):
    """Save a checkpoint's weights as plain NumPy arrays for `winprob.inference`.

    The server only needs a forward pass, which is a few matrix multiplies, so
    it can run without PyTorch (whose install is too big for serverless hosts).
    """
    sd = checkpoint["state_dict"]
    linear = sorted({k.rsplit(".", 1)[0] for k in sd}, key=lambda k: int(k.split(".")[1]))
    arrays = {}
    for i, name in enumerate(linear):
        arrays[f"w{i}"] = sd[f"{name}.weight"].cpu().numpy()
        arrays[f"b{i}"] = sd[f"{name}.bias"].cpu().numpy()
    np.savez(path, mean=np.asarray(checkpoint["mean"], np.float32), std=np.asarray(checkpoint["std"], np.float32),
             features=np.asarray(checkpoint["features"]), **arrays)
